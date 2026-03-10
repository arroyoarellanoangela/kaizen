"""Search the knowledge base with 2-stage retrieval.

Stage 1: Bi-encoder embedding search (fast, top-N * OVERFETCH)
Stage 2: Cross-encoder reranker (precise, top-N final)

The reranker scores each (query, document) pair directly, producing
much more accurate relevance than cosine similarity alone.
"""

from typing import Any

import torch
from sentence_transformers import CrossEncoder

from .config import OVERFETCH_FACTOR, RERANKER_BATCH_SIZE, RERANKER_MODEL, TOP_K
from .store import get_collection

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the reranker model (FP16 on GPU when available)."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
        if torch.cuda.is_available():
            _reranker.model = _reranker.model.to("cuda").half()
    return _reranker


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    n: int = TOP_K,
    category: str | None = None,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """
    2-stage retrieval:
      1) Bi-encoder: fetch n * OVERFETCH candidates from ChromaDB
      2) Cross-encoder: rerank candidates, keep top n

    Returns list of dicts:
      {text, category, subcategory, source, file_type, score}
    """
    col = get_collection()
    if col.count() == 0:
        return []

    # Stage 1 — fast bi-encoder retrieval (over-fetch)
    fetch_n = min(n * OVERFETCH_FACTOR, col.count()) if rerank else min(n, col.count())
    where = {"category": category} if category else None

    results = col.query(
        query_texts=[query],
        n_results=fetch_n,
        where=where,
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
            "subcategory": meta.get("subcategory", ""),
            "source": meta.get("source", ""),
            "file_type": meta.get("file_type", ""),
            "bi_score": round(1.0 - dist, 4),
            "score": round(1.0 - dist, 4),  # default, overwritten by reranker
        })

    # Stage 2 — cross-encoder reranking
    if rerank and len(candidates) > 0:
        try:
            reranker = get_reranker()
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)

            for c, score in zip(candidates, scores):
                c["score"] = round(float(score), 4)

            # Sort by reranker score (higher = more relevant)
            candidates.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            # Fallback: keep bi-encoder ordering if reranker fails
            pass

    return candidates[:n]


def format_context(results: list[dict[str, Any]]) -> str:
    """Format results into a context block ready for prompt injection."""
    if not results:
        return "[no relevant context found]"

    parts = []
    for r in results:
        path = f"{r['category']}/{r['source']}" if r["subcategory"] == "" else f"{r['category']}/{r['subcategory']}/{r['source']}"
        parts.append(
            f"[{path}] (relevance: {r['score']:.0%})\n{r['text']}"
        )

    return "\n\n---\n\n".join(parts)

