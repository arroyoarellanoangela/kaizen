"""LoRA (Low-Rank Adaptation) — implemented from scratch.

Math:
    For a pretrained weight matrix W (d_out x d_in), LoRA decomposes the
    update as a low-rank product:

        W' = W + (alpha / rank) * B @ A

    where A is (rank x d_in) and B is (d_out x rank).

    - A is initialized with Kaiming uniform (so the product starts small)
    - B is initialized with zeros (so W' = W at init — no disruption)
    - Only A and B are trainable; W is frozen

    The scaling factor (alpha / rank) controls the magnitude of the LoRA
    update relative to the pretrained weights. With rank=8 and alpha=16,
    scaling = 2.0.

Total trainable params for bge-m3 (12 layers, d=1024, rank=8):
    Per layer: 2 targets * 2 * (1024 * 8) = 32,768
    Total:     12 * 32,768 = 393,216  (0.07% of 568M)
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a low-rank adapter.

    Forward pass:
        y = x @ W^T + (x @ A^T @ B^T) * scaling
          = original_output + lora_output * scaling

    W is frozen. Only A and B receive gradients.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_in = original.in_features
        d_out = original.out_features
        device = original.weight.device
        dtype = original.weight.dtype

        # Freeze the original weight
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

        # Low-rank matrices — same device/dtype as original weight
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device, dtype=dtype))

        # Kaiming uniform init for A (variance ~ 1/d_in)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        base_out = self.original(x)

        # LoRA path: x -> dropout -> A^T -> B^T -> scale
        lora_out = self.dropout(x)
        lora_out = lora_out @ self.lora_A.T  # (batch, seq, rank)
        lora_out = lora_out @ self.lora_B.T  # (batch, seq, d_out)

        return base_out + lora_out * self.scaling


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: tuple[str, ...] = ("query", "value"),
) -> int:
    """Replace matching nn.Linear layers with LoRALinear wrappers.

    Walks the model graph, finds any nn.Linear whose name matches one of
    target_modules, and replaces it in-place with a LoRALinear.

    All non-LoRA parameters are frozen (requires_grad=False).

    Returns the number of injected LoRA adapters.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    injected = 0
    for _module_name, module in list(model.named_modules()):
        for target in target_modules:
            if not hasattr(module, target):
                continue
            original = getattr(module, target)
            if not isinstance(original, nn.Linear):
                continue

            lora_layer = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, target, lora_layer)
            injected += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA injected: %d adapters, %d trainable / %d total params (%.2f%%)",
        injected,
        trainable,
        total,
        100 * trainable / total if total else 0,
    )

    return injected


def get_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters (for the optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model: nn.Module) -> dict[str, int]:
    """Return trainable and frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def merge_lora(model: nn.Module) -> None:
    """Fold LoRA weights into the base weights for zero-overhead inference.

    After merging, the LoRALinear modules still exist but their A and B
    contributions are absorbed into W. This means inference has no extra
    cost compared to the original model.

    W_merged = W + (alpha / rank) * B @ A
    """
    merged = 0
    for module in model.modules():
        for name, child in list(module._modules.items()):
            if not isinstance(child, LoRALinear):
                continue

            with torch.no_grad():
                # W += scaling * B @ A
                child.original.weight.add_(child.scaling * (child.lora_B @ child.lora_A))

            # Replace LoRALinear with the merged original
            module._modules[name] = child.original
            merged += 1

    logger.info("Merged %d LoRA adapters into base weights.", merged)


def save_lora_weights(model: nn.Module, path: Path) -> None:
    """Save only the LoRA A and B matrices (~300KB for MiniLM + rank 8)."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            state[f"{name}.lora_B"] = module.lora_B.data.cpu()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

    size_kb = path.stat().st_size / 1024
    logger.info("Saved LoRA weights: %s (%.1f KB, %d adapters)", path, size_kb, len(state) // 2)


def load_lora_weights(model: nn.Module, path: Path) -> int:
    """Load saved LoRA weights into an already-injected model.

    The model must have LoRA adapters injected (via inject_lora) before
    calling this function.

    Returns the number of loaded adapters.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    loaded = 0

    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        a_key = f"{name}.lora_A"
        b_key = f"{name}.lora_B"
        if a_key in state and b_key in state:
            module.lora_A.data.copy_(state[a_key])
            module.lora_B.data.copy_(state[b_key])
            loaded += 1

    logger.info("Loaded LoRA weights from %s (%d adapters)", path, loaded)
    return loaded
