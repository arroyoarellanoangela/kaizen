#!/usr/bin/env python3
"""
Retrieval Quality Validation — FP32 vs FP16
=============================================
Validates that FP16 precision does not degrade retrieval quality compared to FP32.

Methodology:
- Fixed set of test queries
- Run retrieval with both FP32 and FP16 embeddings
- Compare: Recall@k, Ranking similarity (Kendall tau)

Usage:
    python validate_quality.py
"""

import gc
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, KNOWLEDGE_DIR
from rag.chunker import chunk_text
from rag.loader import iter_files, read_file


# ---------------------------------------------------------------------------
# Test queries (diverse topics from the knowledge base)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "What is a star schema?",
    "How does medallion architecture work?",
    "Explain schema evolution handling",
    "What is RAG orchestration?",
    "How to optimize batch sizes for GPU",
    "Explain data pipeline design patterns",
    "What are embedding models?",
    "How does Redshift table design work?",
    "What is CUDA programming?",
    "Explain chunking strategies for documents",
    "How to handle schema changes in data pipelines",
    "What is dimensional modeling?",
    "Explain bronze silver gold data layers",
    "How does vector database indexing work?",
    "What are best practices for data engineering?",
]

TOP_K = 10  # results per query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Simplified Kendall tau: fraction of pairwise orderings that agree."""
    # Get common elements
    common = set(ranking_a) & set(ranking_b)
    if len(common) < 2:
        return 1.0  # trivially same

    # Get positions in each ranking
    pos_a = {item: i for i, item in enumerate(ranking_a) if item in common}
    pos_b = {item: i for i, item in enumerate(ranking_b) if item in common}

    items = list(common)
    concordant = 0
    discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_order = pos_a[items[i]] - pos_a[items[j]]
            b_order = pos_b[items[i]] - pos_b[items[j]]
            if a_order * b_order > 0:
                concordant += 1
            elif a_order * b_order < 0:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def recall_at_k(reference: list[str], candidate: list[str], k: int = None) -> float:
    """Recall@k: what fraction of reference top-k items appear in candidate top-k."""
    if k:
        reference = reference[:k]
        candidate = candidate[:k]
    if not reference:
        return 1.0
    return len(set(reference) & set(candidate)) / len(set(reference))


# ---------------------------------------------------------------------------
# Retrieval engine (direct, no ChromaDB)
# ---------------------------------------------------------------------------

def load_chunks(max_chunks: int = 5000) -> tuple[list[str], list[str]]:
    """Load chunks and their IDs."""
    chunks = []
    chunk_ids = []
    for path in iter_files(KNOWLEDGE_DIR):
        text = read_file(path)
        if not text.strip():
            continue
        file_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        rel = path.relative_to(KNOWLEDGE_DIR)
        for i, c in enumerate(file_chunks):
            chunks.append(c)
            chunk_ids.append(f"{rel}#{i}")
        if len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks], chunk_ids[:max_chunks]


def embed_all(model: SentenceTransformer, texts: list[str], batch_size: int = 256):
    """Embed all texts and return as numpy array."""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def search_topk(query_emb, corpus_embs, chunk_ids: list[str], k: int):
    """Find top-k most similar chunks by cosine similarity."""
    import numpy as np
    # Normalize
    query_norm = query_emb / np.linalg.norm(query_emb)
    corpus_norm = corpus_embs / np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    similarities = corpus_norm @ query_norm
    top_indices = np.argsort(similarities)[::-1][:k]
    return [chunk_ids[i] for i in top_indices], [float(similarities[i]) for i in top_indices]


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RETRIEVAL QUALITY VALIDATION — FP32 vs FP16")
    print("=" * 60)

    # Load chunks
    print("\n  Loading chunks...")
    chunks, chunk_ids = load_chunks(5000)
    print(f"  Loaded {len(chunks)} chunks")

    results = {}

    for precision in ["fp32", "fp16"]:
        print(f"\n{'─'*60}")
        print(f"  Encoding corpus in {precision.upper()}...")
        print(f"{'─'*60}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = SentenceTransformer(EMBED_MODEL)
        if torch.cuda.is_available():
            model = model.to("cuda")
            if precision == "fp16":
                model = model.half()

        # Embed corpus
        t0 = time.perf_counter()
        corpus_embs = embed_all(model, chunks)
        corpus_time = time.perf_counter() - t0
        print(f"  Corpus embedded in {corpus_time:.1f}s")

        # Embed queries
        query_embs = model.encode(TEST_QUERIES, batch_size=32, show_progress_bar=False)

        # Search
        precision_results = {}
        for i, query in enumerate(TEST_QUERIES):
            top_ids, top_scores = search_topk(query_embs[i], corpus_embs, chunk_ids, TOP_K)
            precision_results[query] = {"ids": top_ids, "scores": top_scores}

        results[precision] = precision_results

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Compare -----
    print(f"\n\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")

    recalls = []
    taus = []
    query_details = []

    for query in TEST_QUERIES:
        fp32_ids = results["fp32"][query]["ids"]
        fp16_ids = results["fp16"][query]["ids"]

        r_at_k = recall_at_k(fp32_ids, fp16_ids, TOP_K)
        tau = kendall_tau(fp32_ids, fp16_ids)

        recalls.append(r_at_k)
        taus.append(tau)
        query_details.append({
            "query": query,
            "recall_at_k": r_at_k,
            "kendall_tau": tau,
        })

        status = "✅" if r_at_k >= 0.8 else "⚠️"
        print(f"  {status} Recall@{TOP_K}: {r_at_k:.0%} | Tau: {tau:.2f} | {query[:50]}")

    mean_recall = sum(recalls) / len(recalls)
    mean_tau = sum(taus) / len(taus)

    print(f"\n{'─'*60}")
    print(f"  Mean Recall@{TOP_K}:  {mean_recall:.1%}")
    print(f"  Mean Kendall Tau:  {mean_tau:.2f}")
    print(f"{'─'*60}")

    # Generate report
    report = f"""# Quality Validation — FP32 vs FP16 Retrieval Fidelity

## Summary

| Metric | Value |
|--------|-------|
| **Mean Recall@{TOP_K}** | **{mean_recall:.1%}** |
| **Mean Kendall Tau** | **{mean_tau:.2f}** |
| **Test Queries** | {len(TEST_QUERIES)} |
| **Corpus Size** | {len(chunks):,} chunks |
| **Model** | `{EMBED_MODEL}` |

## Interpretation

- **Recall@{TOP_K}**: Fraction of FP32 top-{TOP_K} results also in FP16 top-{TOP_K} (1.0 = identical)
- **Kendall Tau**: Ranking agreement (-1 to 1, where 1.0 = identical ordering)

{'> **✅ FP16 maintains retrieval quality.** No significant degradation observed.' if mean_recall >= 0.9 else '> **⚠️ Some retrieval differences detected.** Review individual queries below.'}

## Per-Query Results

| Query | Recall@{TOP_K} | Kendall Tau | Status |
|-------|-----------|-------------|--------|
"""
    for d in query_details:
        status = "✅" if d["recall_at_k"] >= 0.8 else "⚠️"
        report += f"| {d['query'][:45]} | {d['recall_at_k']:.0%} | {d['kendall_tau']:.2f} | {status} |\n"

    report += f"""
## Methodology

- Both FP32 and FP16 use identical model (`{EMBED_MODEL}`) and corpus
- Corpus embeddings generated fresh for each precision
- Cosine similarity used for retrieval ranking
- No ChromaDB involved — pure embedding comparison
- Recall@k measures result overlap; Kendall tau measures ordering fidelity

---
*Generated by `validate_quality.py` — Kaizen RAG GPU Optimization Pipeline*
"""

    output = Path("QUALITY_VALIDATION.md")
    output.write_text(report, encoding="utf-8")
    print(f"\n  📄 Report saved to: {output.absolute()}")


if __name__ == "__main__":
    main()
