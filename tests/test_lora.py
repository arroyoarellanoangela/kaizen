"""Tests for finetune/lora.py — LoRA implementation from scratch."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from suyven_rag.finetune.lora import (
    LoRALinear,
    count_params,
    get_lora_params,
    inject_lora,
    load_lora_weights,
    merge_lora,
    save_lora_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ToyTransformerLayer(nn.Module):
    """Minimal transformer-like module with query/key/value projections."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.output(q + k + v)


class ToyModel(nn.Module):
    """Two-layer toy transformer for testing."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.layer1 = ToyTransformerLayer(d_model)
        self.layer2 = ToyTransformerLayer(d_model)

    def forward(self, x):
        return self.layer2(self.layer1(x))


@pytest.fixture
def toy_model():
    return ToyModel(d_model=64)


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(2, 4, 64)  # (batch, seq, d_model)


# ---------------------------------------------------------------------------
# TestLoRALinear
# ---------------------------------------------------------------------------


class TestLoRALinear:
    def test_output_shape_matches_original(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        x = torch.randn(2, 4, 64)
        out = lora(x)
        assert out.shape == (2, 4, 64)

    def test_init_preserves_original_output(self):
        """At init, B=0 so LoRA output should equal original output."""
        original = nn.Linear(64, 64)
        x = torch.randn(2, 4, 64)
        expected = original(x)

        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)
        actual = lora(x)

        torch.testing.assert_close(actual, expected)

    def test_original_weight_frozen(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        assert not lora.original.weight.requires_grad

    def test_lora_params_trainable(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_B_initialized_to_zero(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        assert torch.all(lora.lora_B == 0)

    def test_A_not_zero(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        assert not torch.all(lora.lora_A == 0)

    def test_scaling_factor(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8)
        assert lora.scaling == 2.0  # alpha / rank = 8 / 4

    def test_gradient_flows_to_lora_only(self):
        original = nn.Linear(64, 64)
        lora = LoRALinear(original, rank=4, alpha=8, dropout=0.0)
        # Set B to non-zero so there's signal
        with torch.no_grad():
            lora.lora_B.fill_(0.01)

        x = torch.randn(2, 4, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        assert lora.original.weight.grad is None

    def test_rank_dimensions(self):
        original = nn.Linear(128, 64)
        lora = LoRALinear(original, rank=8, alpha=16)
        assert lora.lora_A.shape == (8, 128)  # (rank, d_in)
        assert lora.lora_B.shape == (64, 8)  # (d_out, rank)


# ---------------------------------------------------------------------------
# TestInjectLoRA
# ---------------------------------------------------------------------------


class TestInjectLoRA:
    def test_injects_into_target_modules(self, toy_model):
        n = inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        assert n == 4  # 2 layers * 2 targets

    def test_leaves_non_targets_untouched(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        # key and output should still be nn.Linear
        assert isinstance(toy_model.layer1.key, nn.Linear)
        assert not isinstance(toy_model.layer1.key, LoRALinear)
        assert isinstance(toy_model.layer1.output, nn.Linear)

    def test_freezes_all_non_lora_params(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        for name, param in toy_model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_forward_still_works(self, toy_model, x):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        out = toy_model(x)
        assert out.shape == x.shape

    def test_single_target(self, toy_model):
        n = inject_lora(toy_model, rank=4, alpha=8, target_modules=("query",))
        assert n == 2  # 2 layers * 1 target


# ---------------------------------------------------------------------------
# TestGetLoRAParams
# ---------------------------------------------------------------------------


class TestGetLoRAParams:
    def test_returns_only_trainable(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        params = get_lora_params(toy_model)
        # 4 adapters * 2 params each = 8
        assert len(params) == 8
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# TestCountParams
# ---------------------------------------------------------------------------


class TestCountParams:
    def test_counts_correct(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        counts = count_params(toy_model)

        # Trainable: 4 adapters * (4*64 + 64*4) = 4 * 512 = 2048
        assert counts["trainable"] == 4 * 2 * (4 * 64)
        assert counts["frozen"] > 0
        assert counts["total"] == counts["trainable"] + counts["frozen"]

    def test_trainable_pct_small(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        counts = count_params(toy_model)
        pct = counts["trainable"] / counts["total"]
        assert pct < 0.15  # LoRA should be a small fraction


# ---------------------------------------------------------------------------
# TestMergeLoRA
# ---------------------------------------------------------------------------


class TestMergeLoRA:
    def test_merge_produces_same_output(self, toy_model, x):
        inject_lora(toy_model, rank=4, alpha=8, dropout=0.0, target_modules=("query", "value"))
        # Set non-zero B so merge has something to fold
        with torch.no_grad():
            for m in toy_model.modules():
                if isinstance(m, LoRALinear):
                    m.lora_B.fill_(0.01)

        toy_model.eval()
        out_before = toy_model(x).clone()

        merge_lora(toy_model)
        out_after = toy_model(x)

        torch.testing.assert_close(out_before, out_after, atol=1e-5, rtol=1e-5)

    def test_merge_removes_lora_modules(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        merge_lora(toy_model)

        for m in toy_model.modules():
            assert not isinstance(m, LoRALinear), "LoRALinear should be replaced after merge"

    def test_merged_model_is_all_linear(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))
        merge_lora(toy_model)
        assert isinstance(toy_model.layer1.query, nn.Linear)
        assert isinstance(toy_model.layer1.value, nn.Linear)


# ---------------------------------------------------------------------------
# TestSaveLoadLoRA
# ---------------------------------------------------------------------------


class TestSaveLoadLoRA:
    def test_save_and_load_roundtrip(self, x):
        # Use fixed seed so both models share the same base weights
        torch.manual_seed(123)
        model1 = ToyModel(d_model=64)
        inject_lora(model1, rank=4, alpha=8, dropout=0.0, target_modules=("query", "value"))
        # Set non-zero LoRA weights
        with torch.no_grad():
            for m in model1.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.fill_(0.5)
                    m.lora_B.fill_(0.02)

        model1.eval()
        out_original = model1(x).clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lora.pt"
            save_lora_weights(model1, path)

            # Verify file exists and is small
            assert path.exists()
            assert path.stat().st_size < 50_000  # should be tiny for toy model

            # Create fresh model with SAME seed, inject LoRA, load weights
            torch.manual_seed(123)
            fresh = ToyModel(d_model=64)
            inject_lora(fresh, rank=4, alpha=8, dropout=0.0, target_modules=("query", "value"))
            loaded = load_lora_weights(fresh, path)
            assert loaded == 4  # 4 adapters

            fresh.eval()
            out_loaded = fresh(x)
            torch.testing.assert_close(out_original, out_loaded)

    def test_save_only_lora_weights(self, toy_model):
        inject_lora(toy_model, rank=4, alpha=8, target_modules=("query", "value"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lora.pt"
            save_lora_weights(toy_model, path)
            state = torch.load(path, weights_only=True)

            # Should only contain lora_A and lora_B keys
            assert all("lora_A" in k or "lora_B" in k for k in state)
            assert len(state) == 8  # 4 adapters * 2 matrices


# ---------------------------------------------------------------------------
# TestMNRLLoss
# ---------------------------------------------------------------------------


class TestMNRLLoss:
    def test_perfect_alignment_low_loss(self):
        """Identical query and positive embeddings should give low loss."""
        from suyven_rag.finetune.train import compute_mnrl_loss

        embeds = torch.eye(8)
        loss = compute_mnrl_loss(embeds, embeds, temperature=0.05)
        assert loss.item() < 0.1  # near zero for perfect alignment

    def test_random_embeddings_higher_loss(self):
        from suyven_rag.finetune.train import compute_mnrl_loss

        torch.manual_seed(42)
        q = torch.randn(8, 64)
        p = torch.randn(8, 64)
        loss = compute_mnrl_loss(q, p, temperature=0.05)
        assert loss.item() > 1.0  # random should have high loss

    def test_loss_is_scalar(self):
        from suyven_rag.finetune.train import compute_mnrl_loss

        q = torch.randn(4, 32)
        p = torch.randn(4, 32)
        loss = compute_mnrl_loss(q, p)
        assert loss.dim() == 0  # scalar

    def test_loss_is_differentiable(self):
        from suyven_rag.finetune.train import compute_mnrl_loss

        q = torch.randn(4, 32, requires_grad=True)
        p = torch.randn(4, 32, requires_grad=True)
        loss = compute_mnrl_loss(q, p)
        loss.backward()
        assert q.grad is not None
        assert p.grad is not None


# ---------------------------------------------------------------------------
# TestDataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_load_and_length(self, tmp_path):
        from suyven_rag.finetune.dataset import ContrastivePairsDataset

        data = tmp_path / "pairs.jsonl"
        data.write_text(
            '{"query":"q1","positive":"p1","source":"s1","category":"c1"}\n'
            '{"query":"q2","positive":"p2","source":"s2","category":"c2"}\n'
        )
        ds = ContrastivePairsDataset(data)
        assert len(ds) == 2
        assert ds[0] == ("q1", "p1")
        assert ds[1] == ("q2", "p2")

    def test_train_eval_split(self, tmp_path):
        from suyven_rag.finetune.dataset import ContrastivePairsDataset, train_eval_split

        data = tmp_path / "pairs.jsonl"
        lines = [
            f'{{"query":"q{i}","positive":"p{i}","source":"s","category":"c"}}\n'
            for i in range(100)
        ]
        data.write_text("".join(lines))

        ds = ContrastivePairsDataset(data)
        train_ds, eval_ds = train_eval_split(ds, eval_ratio=0.2)
        assert len(train_ds) == 80
        assert len(eval_ds) == 20
        # No overlap
        train_queries = {q for q, _ in train_ds.pairs}
        eval_queries = {q for q, _ in eval_ds.pairs}
        assert len(train_queries & eval_queries) == 0

    def test_max_samples(self, tmp_path):
        from suyven_rag.finetune.dataset import ContrastivePairsDataset

        data = tmp_path / "pairs.jsonl"
        lines = [
            f'{{"query":"q{i}","positive":"p{i}","source":"s","category":"c"}}\n' for i in range(50)
        ]
        data.write_text("".join(lines))

        ds = ContrastivePairsDataset(data, max_samples=10)
        assert len(ds) == 10
