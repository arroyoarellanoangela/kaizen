"""Training loop for LoRA fine-tuning — raw PyTorch, no Trainer.

Demonstrates:
  - MultipleNegativesRankingLoss (InfoNCE) from scratch
  - Mixed precision (torch.amp) with GradScaler
  - Gradient accumulation (effective batch = batch_size * accumulation_steps)
  - Cosine LR with linear warmup
  - GPU profiling via pynvml

Usage:
    python -m finetune.train
    python -m finetune.train --epochs 5 --lr 3e-5
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from suyven_rag.finetune.config import TrainConfig
from suyven_rag.finetune.dataset import ContrastivePairsDataset, TripletDataset, train_eval_split
from suyven_rag.finetune.lora import (
    count_params,
    get_lora_params,
    inject_lora,
    merge_lora,
    save_lora_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss function: MultipleNegativesRankingLoss (InfoNCE) from scratch
# ---------------------------------------------------------------------------


def compute_mnrl_loss(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE / MultipleNegativesRankingLoss.

    Given a batch of (query, positive) embedding pairs:
      1. Compute cosine similarity matrix: sim[i][j] = cos(query_i, positive_j)
      2. Labels = diagonal (each query_i should match positive_i)
      3. Loss = CrossEntropy(sim / temperature, labels)

    In-batch negatives: for query_i, all positive_j (j != i) act as negatives.
    With batch_size=64, each query gets 63 negatives for free.

    Args:
        query_embeds: (batch, dim) normalized query embeddings
        positive_embeds: (batch, dim) normalized positive embeddings
        temperature: scaling factor (lower = sharper distribution)

    Returns:
        Scalar loss.
    """
    # Normalize (in case model output isn't already L2-normed)
    q = F.normalize(query_embeds, p=2, dim=1)
    p = F.normalize(positive_embeds, p=2, dim=1)

    # Cosine similarity matrix: (batch, batch)
    sim = q @ p.T / temperature

    # Labels: query_i should match positive_i (diagonal)
    labels = torch.arange(sim.size(0), device=sim.device)

    return F.cross_entropy(sim, labels)


def compute_triplet_loss(
    query_embeds: torch.Tensor,
    positive_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Triplet margin loss with hard negatives.

    Pushes query closer to positive and away from negative:
        loss = max(0, d(q, pos) - d(q, neg) + margin)

    where d is cosine distance (1 - cosine_similarity).
    """
    q = F.normalize(query_embeds, p=2, dim=1)
    p = F.normalize(positive_embeds, p=2, dim=1)
    n = F.normalize(negative_embeds, p=2, dim=1)

    d_pos = 1.0 - (q * p).sum(dim=1)  # cosine distance to positive
    d_neg = 1.0 - (q * n).sum(dim=1)  # cosine distance to negative

    return F.relu(d_pos - d_neg + margin).mean()


# ---------------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------------


def encode_texts(model, tokenizer, texts: list[str], max_length: int, device: str) -> torch.Tensor:
    """Tokenize and encode texts through the model, return mean-pooled embeddings."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)

    # Mean pooling over non-padding tokens
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    token_embeddings = outputs.last_hidden_state
    summed = (token_embeddings * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


# ---------------------------------------------------------------------------
# LR scheduler: cosine with linear warmup
# ---------------------------------------------------------------------------


class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """Cosine decay with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        import math

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# GPU profiling
# ---------------------------------------------------------------------------


def gpu_snapshot() -> dict | None:
    """Quick GPU metrics snapshot for training log."""
    try:
        from suyven_rag.rag.monitoring import gpu_metrics

        return gpu_metrics()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: TrainConfig, triplets_path: Path | None = None) -> dict:
    """Run the full training loop. Returns training summary dict."""
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # 1. Load base model
    # sentence-transformers uses short names, but HF transformers needs the full path
    hf_model_id = f"sentence-transformers/{config.base_model}"
    logger.info("Loading base model: %s (HF: %s)", config.base_model, hf_model_id)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModel.from_pretrained(hf_model_id)
    model.to(device)

    # 2. Inject LoRA
    inject_lora(
        model,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.target_modules,
    )
    params = count_params(model)
    logger.info("Params: %s", params)

    # 3. Load dataset
    logger.info("Loading training data: %s", config.train_data_path)
    full_dataset = ContrastivePairsDataset(config.train_data_path)
    train_ds, eval_ds = train_eval_split(full_dataset, eval_ratio=config.eval_split)
    logger.info("Train: %d pairs, Eval: %d pairs", len(train_ds), len(eval_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # 3b. Load triplets (hard negatives) if provided
    triplet_loader = None
    if triplets_path and triplets_path.exists():
        triplet_ds = TripletDataset(triplets_path)
        triplet_loader = DataLoader(
            triplet_ds,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        logger.info("Triplets: %d hard negative triplets", len(triplet_ds))

    # 4. Optimizer: AdamW on LoRA params only
    lora_params = get_lora_params(model)
    optimizer = torch.optim.AdamW(lora_params, lr=config.learning_rate)

    # 5. Scheduler: cosine with warmup
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = CosineWithWarmup(optimizer, warmup_steps, total_steps)

    # 6. Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.fp16 and device == "cuda" else None
    use_amp = scaler is not None

    logger.info(
        "Training: %d epochs, %d steps/epoch, %d total steps, warmup=%d, amp=%s",
        config.epochs,
        steps_per_epoch,
        total_steps,
        warmup_steps,
        use_amp,
    )

    # 7. Training loop
    history = {"train_loss": [], "eval_loss": [], "lr": [], "gpu": []}
    global_step = 0
    t_start = time.time()

    model.train()
    optimizer.zero_grad()

    triplet_iter = None

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Reset triplet iterator each epoch
        if triplet_loader:
            triplet_iter = iter(triplet_loader)

        for step, (queries, positives) in enumerate(train_loader):
            # --- InfoNCE loss on pairs ---
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                q_emb = encode_texts(model, tokenizer, list(queries), config.max_seq_length, device)
                p_emb = encode_texts(
                    model, tokenizer, list(positives), config.max_seq_length, device
                )
                loss = compute_mnrl_loss(q_emb, p_emb, config.temperature)

                # --- Triplet loss on hard negatives (interleaved) ---
                if triplet_iter is not None:
                    try:
                        t_queries, t_positives, t_negatives = next(triplet_iter)
                        tq_emb = encode_texts(
                            model, tokenizer, list(t_queries), config.max_seq_length, device
                        )
                        tp_emb = encode_texts(
                            model, tokenizer, list(t_positives), config.max_seq_length, device
                        )
                        tn_emb = encode_texts(
                            model, tokenizer, list(t_negatives), config.max_seq_length, device
                        )
                        t_loss = compute_triplet_loss(tq_emb, tp_emb, tn_emb)
                        loss = loss + t_loss  # combined loss
                    except StopIteration:
                        triplet_iter = None  # exhausted for this epoch

                loss = loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_steps += 1

                # Log every 10 optimizer steps
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / (epoch_steps * config.gradient_accumulation_steps)
                    lr = scheduler.get_last_lr()[0]
                    gpu = gpu_snapshot()
                    history["train_loss"].append({"step": global_step, "loss": round(avg_loss, 4)})
                    history["lr"].append({"step": global_step, "lr": lr})
                    if gpu:
                        history["gpu"].append({"step": global_step, **gpu})
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e vram=%.1fGB",
                        epoch + 1,
                        global_step,
                        avg_loss,
                        lr,
                        gpu["vram_used_gb"] if gpu else 0,
                    )

        # Epoch eval
        eval_loss = evaluate(model, tokenizer, eval_loader, config, device, use_amp)
        history["eval_loss"].append({"epoch": epoch + 1, "loss": round(eval_loss, 4)})
        logger.info("Epoch %d complete: eval_loss=%.4f", epoch + 1, eval_loss)

    t_total = time.time() - t_start

    # 8. Save LoRA weights
    config.output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = config.output_dir / "lora_weights.pt"
    save_lora_weights(model, lora_path)

    # 9. Merge and save full model
    merge_lora(model)
    merged_path = config.output_dir / "merged_model"
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    logger.info("Saved merged model to %s", merged_path)

    # 10. Loss curve
    plot_loss_curves(history, config.loss_plot_path)

    # 11. Summary
    summary = {
        "base_model": config.base_model,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "trainable_params": params["trainable"],
        "frozen_params": params["frozen"],
        "trainable_pct": round(100 * params["trainable"] / params["total"], 2),
        "train_pairs": len(train_ds),
        "eval_pairs": len(eval_ds),
        "epochs": config.epochs,
        "total_steps": global_step,
        "final_train_loss": history["train_loss"][-1]["loss"] if history["train_loss"] else None,
        "final_eval_loss": history["eval_loss"][-1]["loss"] if history["eval_loss"] else None,
        "training_time_s": round(t_total, 1),
        "lora_checkpoint_kb": round(lora_path.stat().st_size / 1024, 1),
        "device": device,
    }

    summary_path = config.output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary: %s", json.dumps(summary, indent=2))

    return summary


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, tokenizer, eval_loader, config, device, use_amp) -> float:
    """Run eval set, return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for queries, positives in eval_loader:
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            q_emb = encode_texts(model, tokenizer, list(queries), config.max_seq_length, device)
            p_emb = encode_texts(model, tokenizer, list(positives), config.max_seq_length, device)
            loss = compute_mnrl_loss(q_emb, p_emb, config.temperature)

        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------


def plot_loss_curves(history: dict, output_path: Path) -> None:
    """Save training loss curve as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Train loss
        if history["train_loss"]:
            steps = [h["step"] for h in history["train_loss"]]
            losses = [h["loss"] for h in history["train_loss"]]
            ax1.plot(steps, losses, "b-", linewidth=1.5)
            ax1.set_xlabel("Optimizer Step")
            ax1.set_ylabel("Train Loss")
            ax1.set_title("Training Loss (InfoNCE)")
            ax1.grid(True, alpha=0.3)

        # Eval loss per epoch
        if history["eval_loss"]:
            epochs = [h["epoch"] for h in history["eval_loss"]]
            losses = [h["loss"] for h in history["eval_loss"]]
            ax2.bar(epochs, losses, color="steelblue", alpha=0.8)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Eval Loss")
            ax2.set_title("Validation Loss per Epoch")
            ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info("Loss curve saved: %s", output_path)
    except ImportError:
        logger.warning("matplotlib not available, skipping loss curve plot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for embedding model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--data", type=Path, default=None, help="Path to training data JSONL")
    parser.add_argument(
        "--triplets", type=Path, default=None, help="Path to triplets JSONL for hard negatives"
    )
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        gradient_accumulation_steps=args.accum_steps,
    )
    if args.data:
        config.train_data_path = args.data
    train(config, triplets_path=args.triplets)


if __name__ == "__main__":
    main()
