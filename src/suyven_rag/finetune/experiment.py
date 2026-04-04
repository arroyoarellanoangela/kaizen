"""Experiment tracking system — inspired by LlamaFactory + Ruflo's self-learning.

Every training run gets:
  - Unique run ID (timestamp-based)
  - Full config snapshot (YAML)
  - Metrics log (train/eval loss per step)
  - Eval results (intrinsic + retrieval)
  - Checkpoint paths

This enables: reproducibility, comparison across runs, learning from past experiments.

Usage:
    tracker = ExperimentTracker("lora_v2_rank8")
    tracker.log_config(config_dict)
    tracker.log_step(step=10, train_loss=0.5, lr=2e-5)
    tracker.log_eval(eval_loss=0.3, accuracy=98.5)
    tracker.save()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "data" / "finetune" / "experiments"


class ExperimentTracker:
    """Track a single fine-tuning experiment with full provenance."""

    def __init__(self, name: str, tags: list[str] | None = None):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.tags = tags or []
        self.run_dir = EXPERIMENTS_DIR / f"{self.run_id}_{name}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.config: dict[str, Any] = {}
        self.steps: list[dict] = []
        self.evals: list[dict] = []
        self.metadata: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "tags": self.tags,
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        logger.info("Experiment started: %s (run_id=%s)", name, self.run_id)

    def log_config(self, config: dict[str, Any]) -> None:
        """Snapshot the full training config."""
        self.config = config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_step(self, step: int, **metrics: float) -> None:
        """Log metrics for a training step."""
        entry = {"step": step, **metrics}
        self.steps.append(entry)

    def log_eval(self, epoch: int | None = None, **metrics: float) -> None:
        """Log evaluation metrics."""
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        self.evals.append(entry)

    def log_artifact(self, name: str, path: Path) -> None:
        """Track an artifact (checkpoint, loss curve, etc.)."""
        self.metadata.setdefault("artifacts", {})[name] = str(path)

    def finish(self, status: str = "completed") -> Path:
        """Finalize and save the experiment."""
        self.metadata["status"] = status
        self.metadata["finished_at"] = datetime.now().isoformat()
        self.metadata["total_steps"] = len(self.steps)
        self.metadata["total_evals"] = len(self.evals)

        if self.steps:
            self.metadata["final_train_loss"] = self.steps[-1].get("train_loss")
        if self.evals:
            self.metadata["final_eval_loss"] = self.evals[-1].get("eval_loss")
            self.metadata["best_eval_loss"] = min(
                e.get("eval_loss", float("inf")) for e in self.evals
            )

        # Save all data
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        with open(self.run_dir / "steps.jsonl", "w") as f:
            for s in self.steps:
                f.write(json.dumps(s) + "\n")

        with open(self.run_dir / "evals.json", "w") as f:
            json.dump(self.evals, f, indent=2)

        logger.info(
            "Experiment saved: %s (%d steps, %d evals)",
            self.run_dir,
            len(self.steps),
            len(self.evals),
        )
        return self.run_dir


def list_experiments() -> list[dict]:
    """List all past experiments with their metadata."""
    experiments = []
    if not EXPERIMENTS_DIR.exists():
        return experiments

    for d in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                experiments.append(json.load(f))

    return experiments


def get_experiment(run_id: str) -> dict | None:
    """Load full experiment data by run ID."""
    for d in EXPERIMENTS_DIR.iterdir():
        if d.name.startswith(run_id):
            result = {}
            for fname in ["metadata.json", "config.json", "evals.json"]:
                fpath = d / fname
                if fpath.exists():
                    with open(fpath) as f:
                        result[fname.replace(".json", "")] = json.load(f)

            steps_path = d / "steps.jsonl"
            if steps_path.exists():
                result["steps"] = []
                with open(steps_path) as f:
                    for line in f:
                        result["steps"].append(json.loads(line))

            return result

    return None


def compare_experiments(run_ids: list[str]) -> list[dict]:
    """Compare multiple experiments side by side."""
    results = []
    for rid in run_ids:
        exp = get_experiment(rid)
        if exp:
            results.append(
                {
                    "run_id": exp["metadata"]["run_id"],
                    "name": exp["metadata"]["name"],
                    "config": exp.get("config", {}),
                    "final_eval_loss": exp["metadata"].get("final_eval_loss"),
                    "best_eval_loss": exp["metadata"].get("best_eval_loss"),
                    "total_steps": exp["metadata"].get("total_steps"),
                }
            )
    return results
