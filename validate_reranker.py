#!/usr/bin/env python3
"""
Reranker Quality Validation — Bi-Encoder vs Bi-Encoder + Cross-Encoder
========================================================================
Validates that the cross-encoder reranker improves retrieval quality
over bi-encoder-only (cosine similarity from ChromaDB).

Methodology:
- Manual ground-truth: 23 queries with expected relevant sources
- NDCG@5 and NDCG@10 for both pipelines
- Side-by-side comparison showing rank changes
- Per-query quality breakdown

Usage:
    python validate_reranker.py
    python validate_reranker.py --side-by-side    # verbose output
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows (cp1252 can't handle box-drawing/emoji chars)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from sentence_transformers import CrossEncoder

from rag.config import EMBED_MODEL, TOP_K
from rag.store import get_collection

# ---------------------------------------------------------------------------
# Ground truth — queries with expected relevant sources
# ---------------------------------------------------------------------------
# Each entry: (query, [list of source stems that SHOULD rank high])
# These are manually selected based on the knowledge base content.

GROUND_TRUTH = [
    (
        "What is a star schema?",
        ["dimensional-modeling", "star-schema", "data-warehouse", "redshift"],
    ),
    (
        "How does medallion architecture work?",
        ["medallion-architecture", "data-lakehouse", "bronze-silver-gold", "delta-lake"],
    ),
    (
        "What is RAG and how does retrieval-augmented generation work?",
        ["rag", "01_RAG_FUNDAMENTALS", "02_RAG_ADVANCED", "07_RAG_ORCHESTRATION"],
    ),
    (
        "How to optimize GPU inference for transformers?",
        ["10_LLM_INFERENCE_OPTIMIZATION", "06_PARALLEL_ASYNC_RAG_AND_LLM_INFERENCE", "cuda", "gpu"],
    ),
    (
        "Explain data pipeline design patterns",
        ["pipeline", "etl", "elt", "data-engineering", "airflow"],
    ),
    (
        "What are embedding models and how do they work?",
        ["embeddings", "03_EMBEDDINGS", "vector", "sentence-transformers"],
    ),
    (
        "How does chunking work for RAG pipelines?",
        ["chunking", "04_CHUNKING_STRATEGIES", "text-splitting", "overlap"],
    ),
    (
        "What is schema evolution in data engineering?",
        ["schema-evolution", "schema", "migration", "data-warehouse"],
    ),
    (
        "How does vector database indexing work?",
        ["vector-database", "05_VECTOR_STORES", "hnsw", "chromadb", "qdrant"],
    ),
    (
        "What is dimensional modeling and its benefits?",
        ["dimensional-modeling", "kimball", "star-schema", "fact-table", "dimension"],
    ),
    (
        "How to handle data quality issues?",
        ["data-quality", "missing-data", "outlier", "validation", "imputation"],
    ),
    (
        "What are sampling methods in statistics?",
        ["sampling", "stratified", "cluster", "random-sampling", "sample-size"],
    ),
    (
        "Explain the Claude Opus agent architecture",
        ["11_OPUS_AGENT_ARCHITECTURE", "opus", "claude", "agent", "subagent"],
    ),
    (
        "What is LangGraph and how does it work for AI orchestration?",
        ["langgraph", "07_RAG_ORCHESTRATION", "graph", "orchestration", "state"],
    ),
    (
        "How does Redshift optimize query performance?",
        ["redshift", "distribution-key", "sort-key", "columnar", "spectrum"],
    ),
    (
        "What is Apache Spark and how does it process data?",
        ["spark", "pyspark", "rdd", "dataframe", "partition"],
    ),
    (
        "What are the best practices for AWS data architecture?",
        ["aws", "s3", "glue", "athena", "data-lake", "redshift"],
    ),
    (
        "Explain fine-tuning techniques for LLMs",
        ["fine-tuning", "lora", "qlora", "08_FINE_TUNING", "09_FINE_TUNING"],
    ),
    # -- Tricky queries: ambiguous, indirect, similar-doc discrimination --
    (
        # AMBIGUOUS: "model" could be ML model, data model, statistical model
        "What is a model and how should I design one?",
        ["dimensional-modeling", "data-model", "star-schema", "modeling"],
    ),
    (
        # INDIRECT: asking about the concept without using the technical term
        "How do I make sure my AI gives answers based on my own documents instead of hallucinating?",
        ["rag", "01_RAG_FUNDAMENTALS", "retrieval", "grounding", "knowledge"],
    ),
    (
        # DISTINGUISHING SIMILAR: star vs snowflake — both are dimensional models
        "When should I use a normalized snowflake schema instead of a star schema?",
        ["dimensional-modeling", "snowflake", "star-schema", "normalization"],
    ),
    (
        # LONG + SPECIFIC: very detailed question
        "I have a Spark job that reads from S3, applies transformations, and writes to Redshift — what partitioning and distribution strategy should I use for optimal performance?",
        ["spark", "redshift", "partition", "distribution-key", "s3"],
    ),
    (
        # CROSS-DOMAIN: touches both stats and data engineering
        "How do I calculate sample sizes for A/B testing in a data pipeline?",
        ["sampling", "sample-size", "ab-testing", "statistics", "hypothesis"],
    ),
]


# ---------------------------------------------------------------------------
# NDCG calculation
# ---------------------------------------------------------------------------

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG at k. 1.0 = perfect ranking."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def compute_relevance(result: dict, expected_sources: list[str]) -> float:
    """
    Score a single result against expected sources.
    Returns 1.0 for match, 0.0 for no match.
    Uses fuzzy matching: the result's source/category must contain
    any of the expected source keywords (case-insensitive).
    """
    source = result.get("source", "").lower()
    category = result.get("category", "").lower()
    text_preview = result.get("text", "")[:200].lower()

    for expected in expected_sources:
        expected_lower = expected.lower()
        if expected_lower in source or expected_lower in category or expected_lower in text_preview:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OVERFETCH = 4


def retrieve_bi_encoder_only(
    query: str,
    collection,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Stage 1 only: bi-encoder cosine retrieval from ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=min(n, collection.count()),
    )
    if not results or not results["documents"]:
        return []

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "category": meta.get("category", ""),
            "source": meta.get("source", ""),
            "score": round(1.0 - dist, 4),
        })
    return output


def retrieve_with_reranker(
    query: str,
    collection,
    reranker: CrossEncoder,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Stage 1 + Stage 2: bi-encoder overfetch → cross-encoder rerank."""
    fetch_n = min(n * OVERFETCH, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
    )
    if not results or not results["documents"]:
        return []

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "text": doc,
            "category": meta.get("category", ""),
            "source": meta.get("source", ""),
            "bi_score": round(1.0 - dist, 4),
        })

    # Cross-encoder reranking
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    for c, score in zip(candidates, scores):
        c["score"] = round(float(score), 4)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side-by-side", action="store_true", help="Show detailed comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("  RERANKER QUALITY VALIDATION")
    print("  Bi-Encoder vs Bi-Encoder + Cross-Encoder")
    print("=" * 60)

    col = get_collection()
    if col.count() == 0:
        print("\n  ❌ Knowledge base empty. Run: python ingest.py")
        return

    print(f"\n  Knowledge base: {col.count():,} chunks")
    print(f"  Test queries: {len(GROUND_TRUTH)}")
    print(f"  Reranker: {RERANKER_MODEL}")

    # Load reranker
    print("\n  Loading reranker (FP16 GPU)...")
    reranker = CrossEncoder(RERANKER_MODEL)
    if torch.cuda.is_available():
        reranker.model = reranker.model.to("cuda").half()

    # -- Run both pipelines -----------------------------------------
    print(f"\n{'-'*60}")
    print("  Running retrieval...")
    print(f"{'-'*60}")

    query_results = []

    for query, expected_sources in GROUND_TRUTH:
        # Bi-encoder only (top-5 and top-10)
        bi_results_5 = retrieve_bi_encoder_only(query, col, n=5)
        bi_results_10 = retrieve_bi_encoder_only(query, col, n=10)

        # Bi-encoder + reranker (top-5 and top-10)
        rr_results_5 = retrieve_with_reranker(query, col, reranker, n=5)
        rr_results_10 = retrieve_with_reranker(query, col, reranker, n=10)

        # Compute relevance scores
        bi_rel_5 = [compute_relevance(r, expected_sources) for r in bi_results_5]
        bi_rel_10 = [compute_relevance(r, expected_sources) for r in bi_results_10]
        rr_rel_5 = [compute_relevance(r, expected_sources) for r in rr_results_5]
        rr_rel_10 = [compute_relevance(r, expected_sources) for r in rr_results_10]

        # NDCG
        bi_ndcg5 = ndcg_at_k(bi_rel_5, 5)
        bi_ndcg10 = ndcg_at_k(bi_rel_10, 10)
        rr_ndcg5 = ndcg_at_k(rr_rel_5, 5)
        rr_ndcg10 = ndcg_at_k(rr_rel_10, 10)

        # Precision@k (fraction of relevant in top-k)
        bi_p5 = sum(bi_rel_5) / len(bi_rel_5) if bi_rel_5 else 0
        rr_p5 = sum(rr_rel_5) / len(rr_rel_5) if rr_rel_5 else 0

        entry = {
            "query": query,
            "expected": expected_sources,
            "bi_ndcg5": bi_ndcg5,
            "bi_ndcg10": bi_ndcg10,
            "rr_ndcg5": rr_ndcg5,
            "rr_ndcg10": rr_ndcg10,
            "bi_p5": bi_p5,
            "rr_p5": rr_p5,
            "bi_results_5": bi_results_5,
            "rr_results_5": rr_results_5,
            "delta_ndcg5": rr_ndcg5 - bi_ndcg5,
        }
        query_results.append(entry)

        # Status
        delta = entry["delta_ndcg5"]
        if delta > 0.05:
            icon = "🟢"  # reranker improved
        elif delta < -0.05:
            icon = "🔴"  # reranker worse
        else:
            icon = "⚪"  # no change

        print(f"  {icon} NDCG@5: {bi_ndcg5:.2f} → {rr_ndcg5:.2f} ({delta:+.2f}) | {query[:50]}")

    # -- Aggregate metrics ------------------------------------------
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*60}")

    def mean(xs): return sum(xs) / len(xs) if xs else 0

    metrics = {
        "bi_ndcg5": mean([q["bi_ndcg5"] for q in query_results]),
        "bi_ndcg10": mean([q["bi_ndcg10"] for q in query_results]),
        "rr_ndcg5": mean([q["rr_ndcg5"] for q in query_results]),
        "rr_ndcg10": mean([q["rr_ndcg10"] for q in query_results]),
        "bi_p5": mean([q["bi_p5"] for q in query_results]),
        "rr_p5": mean([q["rr_p5"] for q in query_results]),
    }

    print(f"\n  {'Metric':<25} {'Bi-Encoder':>12} {'+ Reranker':>12} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Mean NDCG@5':<25} {metrics['bi_ndcg5']:>11.3f} {metrics['rr_ndcg5']:>11.3f} {metrics['rr_ndcg5']-metrics['bi_ndcg5']:>+9.3f}")
    print(f"  {'Mean NDCG@10':<25} {metrics['bi_ndcg10']:>11.3f} {metrics['rr_ndcg10']:>11.3f} {metrics['rr_ndcg10']-metrics['bi_ndcg10']:>+9.3f}")
    print(f"  {'Mean Precision@5':<25} {metrics['bi_p5']:>11.3f} {metrics['rr_p5']:>11.3f} {metrics['rr_p5']-metrics['bi_p5']:>+9.3f}")

    improved = sum(1 for q in query_results if q["delta_ndcg5"] > 0.05)
    same = sum(1 for q in query_results if abs(q["delta_ndcg5"]) <= 0.05)
    worse = sum(1 for q in query_results if q["delta_ndcg5"] < -0.05)
    print(f"\n  Improved: {improved}  |  Same: {same}  |  Worse: {worse}  (of {len(query_results)} queries)")

    # -- Side-by-side -----------------------------------------------
    if args.side_by_side:
        print(f"\n\n{'='*60}")
        print("  SIDE-BY-SIDE COMPARISON (top-5)")
        print(f"{'='*60}")

        for q in query_results:
            print(f"\n  Query: {q['query']}")
            print(f"  Expected sources: {', '.join(q['expected'])}")
            print(f"  NDCG@5: {q['bi_ndcg5']:.2f} → {q['rr_ndcg5']:.2f} ({q['delta_ndcg5']:+.2f})")
            print(f"  {'-'*56}")
            print(f"  {'Rank':<5} {'Bi-Encoder':<25} {'+ Reranker':<25}")
            print(f"  {'-'*5} {'-'*25} {'-'*25}")
            for i in range(5):
                bi = q["bi_results_5"][i] if i < len(q["bi_results_5"]) else None
                rr = q["rr_results_5"][i] if i < len(q["rr_results_5"]) else None
                bi_src = f"{bi['source'][:20]}" if bi else "—"
                rr_src = f"{rr['source'][:20]}" if rr else "—"
                bi_rel = "✓" if bi and compute_relevance(bi, q["expected"]) > 0 else "✗"
                rr_rel = "✓" if rr and compute_relevance(rr, q["expected"]) > 0 else "✗"
                print(f"  {i+1:<5} {bi_rel} {bi_src:<22} {rr_rel} {rr_src:<22}")

    # -- Generate report --------------------------------------------
    report = f"""# Reranker Quality Validation — Bi-Encoder vs Cross-Encoder

## Summary

| Metric | Bi-Encoder Only | + Cross-Encoder Reranker | Delta |
|--------|:-----------:|:-------------------:|:-----:|
| **Mean NDCG@5** | {metrics['bi_ndcg5']:.3f} | **{metrics['rr_ndcg5']:.3f}** | {metrics['rr_ndcg5']-metrics['bi_ndcg5']:+.3f} |
| **Mean NDCG@10** | {metrics['bi_ndcg10']:.3f} | **{metrics['rr_ndcg10']:.3f}** | {metrics['rr_ndcg10']-metrics['bi_ndcg10']:+.3f} |
| **Mean Precision@5** | {metrics['bi_p5']:.3f} | **{metrics['rr_p5']:.3f}** | {metrics['rr_p5']-metrics['bi_p5']:+.3f} |

| Outcome | Count |
|---------|-------|
| 🟢 Improved | {improved} |
| ⚪ Same | {same} |
| 🔴 Worse | {worse} |

## Configuration

| Parameter | Value |
|-----------|-------|
| **Bi-Encoder** | `{EMBED_MODEL}` (cosine similarity) |
| **Reranker** | `{RERANKER_MODEL}` |
| **Candidates retrieved** | top-{5 * OVERFETCH} (overfetch ×{OVERFETCH}) |
| **Final results** | top-5 / top-10 |
| **Test queries** | {len(GROUND_TRUTH)} (manually curated) |
| **Corpus size** | {col.count():,} chunks |

## Per-Query Results

| Query | NDCG@5 (bi) | NDCG@5 (rr) | Delta | P@5 (bi) | P@5 (rr) |
|-------|:-----------:|:-----------:|:-----:|:--------:|:--------:|
"""
    for q in query_results:
        delta = q["delta_ndcg5"]
        icon = "🟢" if delta > 0.05 else ("🔴" if delta < -0.05 else "⚪")
        report += (
            f"| {q['query'][:45]} | {q['bi_ndcg5']:.2f} | {q['rr_ndcg5']:.2f} | "
            f"{icon} {delta:+.2f} | {q['bi_p5']:.2f} | {q['rr_p5']:.2f} |\n"
        )

    ndcg_delta = metrics['rr_ndcg5'] - metrics['bi_ndcg5']
    verdict = (
        "✅ The cross-encoder reranker improves retrieval quality."
        if ndcg_delta > 0.01
        else "⚠️ The reranker shows marginal or no improvement — review individual queries."
    )

    report += f"""
## Interpretation

> {verdict}

- **NDCG@k** (Normalized Discounted Cumulative Gain): Measures ranking quality,
  giving more weight to relevant results at the top. 1.0 = perfect ranking.
- **Precision@k**: Fraction of top-k results that match expected sources.
- **Ground truth**: Manually curated expected sources per query (fuzzy matched
  against result source names, categories, and text content).

## Methodology

1. **Bi-encoder only**: Query → ChromaDB cosine search → top-k results
2. **+ Reranker**: Query → ChromaDB top-{5*OVERFETCH} → cross-encoder rerank → top-k
3. Each result scored against ground-truth expected sources (binary relevance)
4. NDCG computed per query, then averaged across all queries
5. Side-by-side comparison available with `--side-by-side` flag

---
*Generated by `validate_reranker.py` — Kaizen RAG GPU Optimization Pipeline*
"""

    output = Path("RERANKER_QUALITY.md")
    output.write_text(report, encoding="utf-8")
    print(f"\n  📄 Report: {output.absolute()}")


if __name__ == "__main__":
    main()
