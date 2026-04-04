"""Benchmark metrics — pure functions, no side effects.

All IR metrics operate on a binary relevance vector: [1, 0, 1, 0, 0]
where 1 = relevant document at that rank position.
"""

import math

# ---------------------------------------------------------------------------
# Relevance helpers
# ---------------------------------------------------------------------------


def binary_relevance(retrieved_sources: list[str], relevant_sources: list[str]) -> list[int]:
    """Map retrieved sources to 1 (relevant) or 0 (not relevant)."""
    relevant_set = {s.lower() for s in relevant_sources}
    return [1 if src.lower() in relevant_set else 0 for src in retrieved_sources]


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    DCG  = sum(rel_i / log2(i + 2)) for i in 0..k-1
    IDCG = DCG of perfect ranking (all 1s first)
    NDCG = DCG / IDCG
    """
    rel = relevance[:k]
    if not rel:
        return 0.0

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))

    # Ideal: all relevant docs at top
    ideal = sorted(rel, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr_at_k(relevance: list[int], k: int) -> float:
    """Mean Reciprocal Rank at k. Returns 1/rank of first relevant result."""
    for i, r in enumerate(relevance[:k]):
        if r == 1:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevance: list[int], k: int, total_relevant: int) -> float:
    """Recall at k: fraction of all relevant docs that were retrieved."""
    if total_relevant == 0:
        return 0.0
    found = sum(relevance[:k])
    return min(found / total_relevant, 1.0)


def precision_at_k(relevance: list[int], k: int) -> float:
    """Precision at k: fraction of retrieved docs that are relevant."""
    rel = relevance[:k]
    if not rel:
        return 0.0
    return sum(rel) / len(rel)


# ---------------------------------------------------------------------------
# Generation metrics (no LLM calls)
# ---------------------------------------------------------------------------


def faithfulness_embedding(
    answer: str,
    context: str,
    embed_fn: callable,
) -> float:
    """Cosine similarity between answer and context embeddings.

    embed_fn: callable that takes a list[str] and returns a 2D array of embeddings.
    Uses the same bi-encoder already loaded for retrieval — zero extra model loading.
    """
    if not answer.strip() or not context.strip():
        return 0.0

    embeddings = embed_fn([answer, context])
    a, b = embeddings[0], embeddings[1]

    # Cosine similarity
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def contamination_check(
    retrieved_sources: list[str],
    irrelevant_sources: list[str],
) -> bool:
    """True if any irrelevant source appeared in retrieved results."""
    if not irrelevant_sources:
        return False
    irrelevant_set = {s.lower() for s in irrelevant_sources}
    return any(src.lower() in irrelevant_set for src in retrieved_sources)
