"""A/B evaluation: base embeddings vs LoRA fine-tuned embeddings.

Runs bench.py with base model, then swaps to fine-tuned model and re-runs.
Outputs comparison table.

Usage:
    python -m finetune.evaluate
    python -m finetune.evaluate --checkpoint data/finetune/checkpoints/merged_model
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from suyven_rag.finetune.config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


def run_bench(label: str, extra_args: list[str] | None = None) -> Path:
    """Run bench.py and return the report path."""
    cmd = [
        sys.executable,
        "bench.py",
        "--agents",
        "--label",
        label,
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("bench.py failed:\n%s", result.stderr)
        raise RuntimeError(f"bench.py failed with code {result.returncode}")

    # Find the report file
    report_dir = BASE_DIR / "data" / "eval" / "bench"
    reports = sorted(
        report_dir.glob(f"*{label}*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not reports:
        raise FileNotFoundError(f"No report found for label '{label}'")

    return reports[0]


def run_compare(report_a: Path, report_b: Path) -> None:
    """Run bench.py --compare."""
    cmd = [
        sys.executable,
        "bench.py",
        "--compare",
        str(report_a),
        str(report_b),
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR))


def swap_embed_model(checkpoint_path: Path) -> None:
    """Register the fine-tuned model in the model registry at runtime."""
    import torch
    from sentence_transformers import SentenceTransformer

    from suyven_rag.rag.model_registry import ModelInfo, _embed_models, _registry

    logger.info("Loading fine-tuned model from %s", checkpoint_path)

    # Load as SentenceTransformer by creating a model from the HF checkpoint
    # The merged model is a raw HuggingFace model, wrap it in SentenceTransformer
    model = SentenceTransformer(str(checkpoint_path))
    if torch.cuda.is_available():
        model = model.to("cuda").half()

    # Register in model registry
    _registry["finetuned_embed"] = ModelInfo(
        name="finetuned_embed",
        model_id=str(checkpoint_path),
        model_type="embed",
    )
    _embed_models["finetuned_embed"] = model

    # Swap default to point to fine-tuned
    _embed_models["default_embed"] = model

    logger.info("Swapped default embed model to fine-tuned")


def restore_embed_model() -> None:
    """Restore the original embedding model."""
    from suyven_rag.rag.model_registry import _embed_models

    # Remove cached finetuned model, let registry reload default on next access
    _embed_models.pop("default_embed", None)
    _embed_models.pop("finetuned_embed", None)

    logger.info("Restored default embed model")


def evaluate(config: TrainConfig, checkpoint: Path | None = None) -> None:
    """Run full A/B evaluation."""
    ckpt = checkpoint or config.output_dir / "merged_model"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    # Step 1: Baseline
    logger.info("=== Phase 1: Baseline benchmark ===")
    report_base = run_bench("base_embed")
    logger.info("Baseline report: %s", report_base)

    # Step 2: Fine-tuned
    # Note: bench.py runs as a subprocess, so we can't swap models in-process.
    # Instead, we set an env var that the model registry can pick up.
    import os

    os.environ["FINETUNED_EMBED_PATH"] = str(ckpt)

    logger.info("=== Phase 2: Fine-tuned benchmark ===")
    report_ft = run_bench("finetuned_embed")
    logger.info("Fine-tuned report: %s", report_ft)

    # Step 3: Compare
    logger.info("=== Phase 3: Comparison ===")
    run_compare(report_base, report_ft)

    # Cleanup
    os.environ.pop("FINETUNED_EMBED_PATH", None)


def main():
    parser = argparse.ArgumentParser(description="A/B evaluation: base vs fine-tuned embeddings")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to merged model checkpoint",
    )
    args = parser.parse_args()

    config = TrainConfig()
    evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
