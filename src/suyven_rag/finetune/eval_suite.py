"""Modular evaluation suite — inspired by EleutherAI's lm-evaluation-harness.

Each eval task is a registered plugin with:
  - Name and description
  - Input data (auto-loaded or provided)
  - Metric computation
  - Result formatting

Usage:
    # Run all registered tasks
    python -m finetune.eval_suite --model data/finetune/checkpoints/merged_model

    # Run specific tasks
    python -m finetune.eval_suite --tasks intrinsic,retrieval --model ...

    # List available tasks
    python -m finetune.eval_suite --list
"""

import argparse
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Task registry (harness-style)
# ---------------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type["EvalTask"]] = {}


def register_task(name: str):
    """Decorator to register an eval task."""

    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        cls.task_name = name
        return cls

    return decorator


def list_tasks() -> list[dict]:
    """List all registered eval tasks."""
    return [
        {"name": name, "description": cls.__doc__ or "No description"}
        for name, cls in _TASK_REGISTRY.items()
    ]


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------


class EvalTask(ABC):
    """Base class for evaluation tasks."""

    task_name: str = ""

    def __init__(self, base_model: SentenceTransformer, ft_model: SentenceTransformer):
        self.base = base_model
        self.ft = ft_model

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Run the evaluation, return metrics dict."""
        ...

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Batch cosine similarity."""
        return np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
        )


# ---------------------------------------------------------------------------
# Task 1: Intrinsic discrimination (pos vs neg)
# ---------------------------------------------------------------------------


@register_task("intrinsic")
class IntrinsicDiscrimination(EvalTask):
    """Measures if model better distinguishes correct vs wrong passages."""

    def run(self) -> dict[str, Any]:
        pairs_path = BASE_DIR / "data" / "finetune" / "pairs_v2.jsonl"
        if not pairs_path.exists():
            return {"error": "No training data found"}

        pairs = []
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                pairs.append(json.loads(line))

        random.seed(42)
        random.shuffle(pairs)
        eval_pairs = pairs[-296:]

        queries = [p["query"] for p in eval_pairs]
        positives = [p["positive"] for p in eval_pairs]
        negatives = positives[1:] + positives[:1]  # Shifted as negatives

        bq = self.base.encode(queries, show_progress_bar=False, convert_to_numpy=True)
        bp = self.base.encode(positives, show_progress_bar=False, convert_to_numpy=True)
        bn = self.base.encode(negatives, show_progress_bar=False, convert_to_numpy=True)
        fq = self.ft.encode(queries, show_progress_bar=False, convert_to_numpy=True)
        fp_ = self.ft.encode(positives, show_progress_bar=False, convert_to_numpy=True)
        fn = self.ft.encode(negatives, show_progress_bar=False, convert_to_numpy=True)

        base_margin = self._cos_sim(bq, bp) - self._cos_sim(bq, bn)
        ft_margin = self._cos_sim(fq, fp_) - self._cos_sim(fq, fn)

        base_acc = float(np.mean(self._cos_sim(bq, bp) > self._cos_sim(bq, bn))) * 100
        ft_acc = float(np.mean(self._cos_sim(fq, fp_) > self._cos_sim(fq, fn))) * 100

        return {
            "base_accuracy": round(base_acc, 1),
            "ft_accuracy": round(ft_acc, 1),
            "accuracy_delta": round(ft_acc - base_acc, 1),
            "base_margin": round(float(np.mean(base_margin)), 4),
            "ft_margin": round(float(np.mean(ft_margin)), 4),
            "margin_delta": round(float(np.mean(ft_margin) - np.mean(base_margin)), 4),
            "margin_improvement_pct": round(
                ((float(np.mean(ft_margin)) / max(float(np.mean(base_margin)), 1e-8)) - 1) * 100, 1
            ),
            "n_pairs": len(eval_pairs),
        }


# ---------------------------------------------------------------------------
# Task 2: Retrieval quality (search ChromaDB, score with reranker)
# ---------------------------------------------------------------------------


@register_task("retrieval")
class RetrievalQuality(EvalTask):
    """Measures retrieval quality on real queries against ChromaDB."""

    TEST_QUERIES = [
        "How does AWS Lambda handle cold starts?",
        "What is the difference between SageMaker and EC2 for ML training?",
        "How to set up a VPC with private and public subnets?",
        "What are the best practices for DynamoDB partition key design?",
        "How does Kubernetes horizontal pod autoscaling work?",
        "What is the CAP theorem and how does it apply to distributed databases?",
        "How to implement CI/CD pipeline with GitHub Actions?",
        "What is retrieval augmented generation?",
        "How does gradient descent work in neural networks?",
        "What are the advantages of using Docker containers?",
        "How to handle rate limiting in API design?",
        "What is infrastructure as code and why use Terraform?",
        "How does Apache Kafka guarantee message ordering?",
        "What is the difference between SQL and NoSQL databases?",
        "How to implement authentication with JWT tokens?",
        "What are microservices architecture patterns?",
        "How to optimize query performance in PostgreSQL?",
        "What is event-driven architecture?",
        "How does load balancing work in cloud environments?",
        "What are the SOLID principles in software design?",
    ]

    def run(self) -> dict[str, Any]:
        from suyven_rag.rag.index_registry import get_index
        from suyven_rag.rag.model_registry import get_reranker

        collection = get_index()
        reranker = get_reranker()

        def search_and_score(model, queries, top_k=5):
            embeddings = model.encode(queries, show_progress_bar=False, convert_to_numpy=True)
            all_scores = []
            for i, q in enumerate(queries):
                result = collection.query(
                    query_embeddings=[embeddings[i].tolist()],
                    n_results=top_k,
                    include=["documents"],
                )
                docs = result["documents"][0]
                pairs = [(q, doc) for doc in docs]
                scores = reranker.predict(pairs, show_progress_bar=False)
                all_scores.append(scores.tolist())
            return all_scores

        base_scores = search_and_score(self.base, self.TEST_QUERIES)
        ft_scores = search_and_score(self.ft, self.TEST_QUERIES)

        # Compute MRR@5 (using reranker score as relevance proxy)
        def mrr(scores_list, threshold=0.5):
            """MRR where a doc is 'relevant' if reranker score > threshold."""
            rrs = []
            for scores in scores_list:
                found = False
                for rank, s in enumerate(scores, 1):
                    if s > threshold:
                        rrs.append(1.0 / rank)
                        found = True
                        break
                if not found:
                    rrs.append(0.0)
            return float(np.mean(rrs))

        # Win/tie/loss analysis
        wins, ties, losses = 0, 0, 0
        for i in range(len(self.TEST_QUERIES)):
            b1, f1 = base_scores[i][0], ft_scores[i][0]
            if f1 > b1 + 0.1:
                wins += 1
            elif b1 > f1 + 0.1:
                losses += 1
            else:
                ties += 1

        return {
            "base_avg_top1_score": round(float(np.mean([s[0] for s in base_scores])), 3),
            "ft_avg_top1_score": round(float(np.mean([s[0] for s in ft_scores])), 3),
            "base_mrr5": round(mrr(base_scores), 3),
            "ft_mrr5": round(mrr(ft_scores), 3),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
            "n_queries": len(self.TEST_QUERIES),
        }


# ---------------------------------------------------------------------------
# Task 3: Embedding space quality
# ---------------------------------------------------------------------------


@register_task("embedding_space")
class EmbeddingSpaceQuality(EvalTask):
    """Measures embedding space properties: isotropy, cluster separation."""

    def run(self) -> dict[str, Any]:
        # Sample diverse texts from ChromaDB
        from suyven_rag.rag.index_registry import get_index

        col = get_index()
        result = col.get(limit=500, include=["documents", "metadatas"])
        texts = result["documents"]
        categories = [m.get("category", "unknown") for m in result["metadatas"]]

        base_emb = self.base.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype(
            np.float32
        )
        ft_emb = self.ft.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype(
            np.float32
        )

        def analyze_space(embeddings, cats):
            # Isotropy: how uniformly distributed are embeddings?
            # Perfect isotropy = all singular values equal
            _, s, _ = np.linalg.svd(embeddings - embeddings.mean(axis=0), full_matrices=False)
            isotropy = float(np.min(s) / (np.max(s) + 1e-8))

            # Inter-cluster distance (between categories)
            unique_cats = list(set(cats))
            centroids = {}
            for cat in unique_cats:
                mask = [i for i, c in enumerate(cats) if c == cat]
                if mask:
                    centroids[cat] = embeddings[mask].mean(axis=0)

            if len(centroids) > 1:
                dists = []
                keys = list(centroids.keys())
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        d = np.linalg.norm(centroids[keys[i]] - centroids[keys[j]])
                        dists.append(d)
                inter_cluster = float(np.mean(dists))
            else:
                inter_cluster = 0.0

            # Intra-cluster density (within categories)
            intra_dists = []
            for cat in unique_cats:
                mask = [i for i, c in enumerate(cats) if c == cat]
                if len(mask) > 1:
                    cat_emb = embeddings[mask]
                    centroid = cat_emb.mean(axis=0)
                    dists = np.linalg.norm(cat_emb - centroid, axis=1)
                    intra_dists.append(float(np.mean(dists)))

            intra_cluster = float(np.mean(intra_dists)) if intra_dists else 0.0

            return {
                "isotropy": round(isotropy, 4),
                "inter_cluster_dist": round(inter_cluster, 4),
                "intra_cluster_dist": round(intra_cluster, 4),
                "separation_ratio": round(inter_cluster / max(intra_cluster, 1e-8), 3),
            }

        base_stats = analyze_space(base_emb, categories)
        ft_stats = analyze_space(ft_emb, categories)

        return {
            "base": base_stats,
            "ft": ft_stats,
            "isotropy_delta": round(ft_stats["isotropy"] - base_stats["isotropy"], 4),
            "separation_delta": round(
                ft_stats["separation_ratio"] - base_stats["separation_ratio"], 3
            ),
        }


# ---------------------------------------------------------------------------
# Task 4: Latency benchmark
# ---------------------------------------------------------------------------


@register_task("latency")
class LatencyBenchmark(EvalTask):
    """Measures inference latency for embedding generation."""

    def run(self) -> dict[str, Any]:
        test_texts = [
            "How does AWS Lambda handle cold starts?",
            "What is the difference between batch and stream processing in data engineering?",
            "Explain the concept of containerization and its benefits for deployment.",
        ] * 10  # 30 texts

        # Warmup
        self.base.encode(test_texts[:3], show_progress_bar=False)
        self.ft.encode(test_texts[:3], show_progress_bar=False)

        # Benchmark
        t0 = time.time()
        self.base.encode(test_texts, show_progress_bar=False)
        base_time = time.time() - t0

        t0 = time.time()
        self.ft.encode(test_texts, show_progress_bar=False)
        ft_time = time.time() - t0

        return {
            "base_total_ms": round(base_time * 1000, 1),
            "ft_total_ms": round(ft_time * 1000, 1),
            "base_per_text_ms": round(base_time / len(test_texts) * 1000, 2),
            "ft_per_text_ms": round(ft_time / len(test_texts) * 1000, 2),
            "overhead_pct": round((ft_time / max(base_time, 1e-8) - 1) * 100, 1),
            "n_texts": len(test_texts),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_eval_suite(
    model_path: str | Path,
    tasks: list[str] | None = None,
    base_model_name: str = "BAAI/bge-m3",
) -> dict[str, Any]:
    """Run evaluation suite and return results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading base model: %s", base_model_name)
    base = SentenceTransformer(base_model_name).to(device)
    if device == "cuda":
        base = base.half()

    logger.info("Loading fine-tuned model: %s", model_path)
    ft = SentenceTransformer(str(model_path)).to(device)
    if device == "cuda":
        ft = ft.half()

    task_names = tasks or list(_TASK_REGISTRY.keys())
    results = {}

    for name in task_names:
        if name not in _TASK_REGISTRY:
            logger.warning("Unknown task: %s (available: %s)", name, list(_TASK_REGISTRY.keys()))
            continue

        logger.info("Running eval task: %s", name)
        task = _TASK_REGISTRY[name](base, ft)

        t0 = time.time()
        result = task.run()
        elapsed = time.time() - t0

        result["_elapsed_s"] = round(elapsed, 2)
        results[name] = result
        logger.info("  %s: %.1fs", name, elapsed)

    return results


def print_results(results: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUITE RESULTS")
    print("=" * 70)

    for task_name, metrics in results.items():
        print(f"\n--- {task_name} ---")
        for k, v in metrics.items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Embedding model evaluation suite")
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_DIR / "data" / "finetune" / "checkpoints" / "merged_model",
    )
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task names")
    parser.add_argument("--list", action="store_true", help="List available tasks")
    parser.add_argument("--output", type=Path, default=None, help="Save results to JSON")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable eval tasks:")
        for task in list_tasks():
            print(f"  {task['name']}: {task['description'].strip().split(chr(10))[0]}")
        return

    task_list = args.tasks.split(",") if args.tasks else None
    results = run_eval_suite(args.model, tasks=task_list)
    print_results(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
