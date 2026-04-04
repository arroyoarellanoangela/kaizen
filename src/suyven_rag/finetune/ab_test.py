"""Quick A/B retrieval test: base vs fine-tuned embeddings.

Embeds test queries with both models, searches ChromaDB, compares results.

Usage:
    python -m finetune.ab_test
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = BASE_DIR / "data" / "finetune" / "checkpoints" / "merged_model"

# Test queries: mix of real-world questions users might ask
TEST_QUERIES = [
    "How does AWS Lambda handle cold starts?",
    "What is the difference between SageMaker and EC2 for ML training?",
    "How to set up a VPC with private and public subnets?",
    "What are the best practices for DynamoDB partition key design?",
    "How does Kubernetes horizontal pod autoscaling work?",
    "What is the CAP theorem and how does it apply to distributed databases?",
    "How to implement CI/CD pipeline with GitHub Actions?",
    "What is the difference between batch and stream processing?",
    "How to optimize Python code for large dataset processing?",
    "What are transformer attention mechanisms?",
    "How to configure S3 bucket policies for cross-account access?",
    "What is retrieval augmented generation?",
    "How does gradient descent work in neural networks?",
    "What are the advantages of using Docker containers?",
    "How to handle rate limiting in API design?",
    "What is infrastructure as code and why use Terraform?",
    "How does Apache Kafka guarantee message ordering?",
    "What is the difference between SQL and NoSQL databases?",
    "How to implement authentication with JWT tokens?",
    "What are microservices architecture patterns?",
]


def load_models():
    """Load base and fine-tuned sentence-transformer models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading base model...")
    base = SentenceTransformer("BAAI/bge-m3")
    base = base.to(device)
    if device == "cuda":
        base = base.half()

    logger.info("Loading fine-tuned model from %s...", CHECKPOINT)
    finetuned = SentenceTransformer(str(CHECKPOINT))
    finetuned = finetuned.to(device)
    if device == "cuda":
        finetuned = finetuned.half()

    return base, finetuned


def search_with_model(model, queries, collection, top_k=5):
    """Embed queries with model and search ChromaDB."""
    embeddings = model.encode(queries, show_progress_bar=False, convert_to_numpy=True)

    results = []
    for i, query in enumerate(queries):
        result = collection.query(
            query_embeddings=[embeddings[i].tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results.append(
            {
                "query": query,
                "docs": result["documents"][0],
                "metadatas": result["metadatas"][0],
                "distances": result["distances"][0],
            }
        )
    return results


def score_with_reranker(queries, docs_list):
    """Score query-doc pairs with cross-encoder reranker."""
    from suyven_rag.rag.model_registry import get_reranker

    reranker = get_reranker()
    all_scores = []

    for query, docs in zip(queries, docs_list, strict=False):
        pairs = [(query, doc) for doc in docs]
        scores = reranker.predict(pairs, show_progress_bar=False)
        all_scores.append(scores.tolist())

    return all_scores


def compare(base_results, ft_results, base_scores, ft_scores):
    """Compare base vs fine-tuned retrieval quality."""
    print("\n" + "=" * 80)
    print("A/B RETRIEVAL COMPARISON: Base vs Fine-tuned")
    print("=" * 80)

    # Aggregate metrics
    base_avg_top1 = np.mean([s[0] for s in base_scores])
    ft_avg_top1 = np.mean([s[0] for s in ft_scores])
    base_avg_top5 = np.mean([np.mean(s) for s in base_scores])
    ft_avg_top5 = np.mean([np.mean(s) for s in ft_scores])
    base_avg_max = np.mean([max(s) for s in base_scores])
    ft_avg_max = np.mean([max(s) for s in ft_scores])

    print(f"\n{'Metric':<30} {'Base':>10} {'FineTuned':>10} {'Delta':>10}")
    print("-" * 60)
    print(
        f"{'Avg reranker score @1':<30} {base_avg_top1:>10.3f} {ft_avg_top1:>10.3f} {ft_avg_top1 - base_avg_top1:>+10.3f}"
    )
    print(
        f"{'Avg reranker score @5':<30} {base_avg_top5:>10.3f} {ft_avg_top5:>10.3f} {ft_avg_top5 - base_avg_top5:>+10.3f}"
    )
    print(
        f"{'Avg best score in top-5':<30} {base_avg_max:>10.3f} {ft_avg_max:>10.3f} {ft_avg_max - base_avg_max:>+10.3f}"
    )

    # Per-query comparison
    wins = 0
    ties = 0
    losses = 0

    print(f"\n{'Query':<55} {'Base@1':>8} {'FT@1':>8} {'Winner':>8}")
    print("-" * 80)

    for i, query in enumerate(TEST_QUERIES):
        b1 = base_scores[i][0]
        f1 = ft_scores[i][0]
        if f1 > b1 + 0.1:
            winner = "FT"
            wins += 1
        elif b1 > f1 + 0.1:
            winner = "Base"
            losses += 1
        else:
            winner = "Tie"
            ties += 1

        q_short = query[:52] + "..." if len(query) > 55 else query
        print(f"{q_short:<55} {b1:>8.2f} {f1:>8.2f} {winner:>8}")

    print(f"\nWins: {wins}  Ties: {ties}  Losses: {losses}")
    print(f"Win rate: {wins}/{wins + losses} = {100 * wins / max(wins + losses, 1):.0f}%")

    # Show example where fine-tuned improved most
    deltas = [ft_scores[i][0] - base_scores[i][0] for i in range(len(TEST_QUERIES))]
    best_i = int(np.argmax(deltas))
    worst_i = int(np.argmin(deltas))

    print(f"\nBiggest improvement: {TEST_QUERIES[best_i][:60]}")
    print(f"  Base top result: {base_results[best_i]['docs'][0][:100]}...")
    print(f"  FT top result:   {ft_results[best_i]['docs'][0][:100]}...")

    if deltas[worst_i] < -0.1:
        print(f"\nBiggest regression: {TEST_QUERIES[worst_i][:60]}")
        print(f"  Base top result: {base_results[worst_i]['docs'][0][:100]}...")
        print(f"  FT top result:   {ft_results[worst_i]['docs'][0][:100]}...")

    return {
        "base_avg_top1": round(float(base_avg_top1), 3),
        "ft_avg_top1": round(float(ft_avg_top1), 3),
        "base_avg_top5": round(float(base_avg_top5), 3),
        "ft_avg_top5": round(float(ft_avg_top5), 3),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def main():
    from suyven_rag.rag.index_registry import get_index

    base_model, ft_model = load_models()
    collection = get_index()

    logger.info("Searching with base model...")
    t0 = time.time()
    base_results = search_with_model(base_model, TEST_QUERIES, collection, top_k=5)
    t_base = time.time() - t0

    logger.info("Searching with fine-tuned model...")
    t0 = time.time()
    ft_results = search_with_model(ft_model, TEST_QUERIES, collection, top_k=5)
    t_ft = time.time() - t0

    logger.info("Scoring with cross-encoder reranker...")
    base_scores = score_with_reranker(TEST_QUERIES, [r["docs"] for r in base_results])
    ft_scores = score_with_reranker(TEST_QUERIES, [r["docs"] for r in ft_results])

    logger.info("Search time: base=%.2fs, fine-tuned=%.2fs", t_base, t_ft)

    summary = compare(base_results, ft_results, base_scores, ft_scores)

    # Save results
    output = BASE_DIR / "data" / "finetune" / "ab_results.json"
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
