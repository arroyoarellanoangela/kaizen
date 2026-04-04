"""Orchestrator — decides HOW to handle each query.

Receives a query and returns a RoutePlan: which index, models, and mode to use.
V2.2: hybrid search (BM25 + dense + reranker), multi-tool retrieval.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import LLM_MODEL, RERANKER_BATCH_SIZE, TOP_K
from .index_registry import get_index, route_to_index
from .llm import quick_complete
from .model_registry import get_reranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Code detection keywords
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = re.compile(
    r"\b(def|class|import|function|async|await|return|const|let|var|"
    r"SELECT|INSERT|CREATE|ALTER|DROP|JOIN|WHERE|"
    r"docker|kubernetes|kubectl|terraform|"
    r"git\s+(push|pull|merge|rebase|cherry-pick)|"
    r"pip\s+install|npm\s+install|cargo\s+build)\b",
    re.IGNORECASE,
)

_SUMMARY_KEYWORDS = re.compile(
    r"\b(compara|compare|sintetiza|synthesize|resume|summarize|"
    r"diferencia|difference|pros\s+y\s+cons|pros\s+and\s+cons|"
    r"ventajas|advantages|overview)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Route plan
# ---------------------------------------------------------------------------


@dataclass
class RoutePlan:
    indexes: list[str]  # which collections to query
    embed_model: str  # which embed model to use
    use_reranker: bool  # whether to apply cross-encoder
    reranker_model: str  # which reranker to use
    llm_model: str  # which LLM for generation
    mode: str  # "answer" | "summary" | "code"
    top_k: int  # how many results to return
    reason: str  # why this route was chosen (traceability)
    use_bm25: bool = True  # whether to use BM25 hybrid search


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan(
    query: str,
    category: str | None = None,
    top_k: int = TOP_K,
) -> RoutePlan:
    """Build a RoutePlan for the given query. Deterministic rules, no AI."""

    # Mode detection
    if _CODE_KEYWORDS.search(query):
        mode = "code"
        reason = "code keywords detected"
    elif len(query) > 150 or _SUMMARY_KEYWORDS.search(query):
        mode = "summary"
        reason = "summary/comparison request or long query"
    else:
        mode = "answer"
        reason = "default path"

    # Index routing
    index_name = route_to_index(query, hint=category)

    route = RoutePlan(
        indexes=[index_name],
        embed_model="default_embed",
        use_reranker=True,
        reranker_model="default_reranker",
        llm_model=LLM_MODEL,
        mode=mode,
        top_k=top_k,
        reason=reason,
        use_bm25=True,
    )

    logger.info(
        '[orchestrator] query="%s" -> indexes=%s embed=%s reranker=%s llm=%s mode=%s reason="%s"',
        query[:80],
        route.indexes,
        route.embed_model,
        route.use_reranker,
        route.llm_model,
        route.mode,
        route.reason,
    )

    return route


# ---------------------------------------------------------------------------
# Execution (uses the plan to run the full pipeline)
# ---------------------------------------------------------------------------


def _bm25_search(
    query: str,
    col,
    fetch_n: int,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """BM25 keyword search over ChromaDB documents.

    Fetches a broad set of documents and scores them with BM25.
    Catches terms that dense search misses (exact names, acronyms, etc.).
    """
    from rank_bm25 import BM25Okapi

    # Fetch a broad pool for BM25 scoring
    pool_size = min(fetch_n * 5, col.count())
    result = col.get(
        limit=pool_size,
        where=where,
        include=["documents", "metadatas"],
    )
    if not result or not result["documents"]:
        return []

    docs = result["documents"]
    metas = result["metadatas"]

    # Tokenize for BM25
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Rank and return top candidates
    top_indices = np.argsort(scores)[::-1][:fetch_n]
    candidates = []
    for idx in top_indices:
        idx = int(idx)
        if scores[idx] <= 0:
            continue
        meta = metas[idx]
        candidates.append(
            {
                "text": docs[idx],
                "category": meta.get("category", ""),
                "subcategory": meta.get("subcategory", ""),
                "source": meta.get("source", ""),
                "file_type": meta.get("file_type", ""),
                "bm25_score": round(float(scores[idx]), 4),
                "bi_score": 0.0,
                "score": 0.0,
            }
        )

    return candidates


def _merge_hybrid(
    dense: list[dict],
    sparse: list[dict],
    top_n: int,
    dense_weight: float = 0.5,
) -> list[dict]:
    """Reciprocal Rank Fusion (RRF) merge of dense + BM25 results.

    RRF is the industry standard for hybrid search (used by Elasticsearch,
    Weaviate, etc.). It's rank-based so it's robust to score scale differences.
    """
    k = 60  # RRF constant

    seen = {}  # text_hash -> candidate dict

    for rank, c in enumerate(dense):
        h = hash(c["text"][:200])
        if h not in seen:
            seen[h] = {**c, "rrf_score": 0.0, "retrieval_methods": ["dense"]}
        seen[h]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, c in enumerate(sparse):
        h = hash(c["text"][:200])
        if h not in seen:
            seen[h] = {**c, "rrf_score": 0.0, "retrieval_methods": ["bm25"]}
        else:
            seen[h]["retrieval_methods"].append("bm25")
            seen[h]["bm25_score"] = c.get("bm25_score", 0.0)
        seen[h]["rrf_score"] += 1.0 / (k + rank + 1)

    merged = sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_n]


def _fetch_adjacent_chunks(
    col,
    results: list[dict],
    window: int = 1,
) -> list[dict]:
    """Chunk read: fetch adjacent chunks for top results to provide fuller context.

    Inspired by A-RAG's hierarchical retrieval: after finding the best chunk,
    read its neighbors to give the LLM complete context.
    """
    enriched = []
    for r in results:
        source = r.get("source", "")
        try:
            chunk_idx = int(r.get("chunk_index", -1))
        except (ValueError, TypeError):
            enriched.append(r)
            continue

        if chunk_idx < 0 or not source:
            enriched.append(r)
            continue

        # Fetch adjacent chunks from same source
        adjacent_texts = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            adj_idx = chunk_idx + offset
            if adj_idx < 0:
                continue
            try:
                adj = col.get(
                    where={"source": source},
                    include=["documents", "metadatas"],
                    limit=1,
                    offset=adj_idx,
                )
                if adj and adj["documents"]:
                    adjacent_texts.append(adj["documents"][0])
            except Exception:
                pass

        if adjacent_texts:
            # Prepend/append adjacent context
            full_text = (
                "\n".join(adjacent_texts[:window])
                + "\n"
                + r["text"]
                + "\n"
                + "\n".join(adjacent_texts[window:])
            )
            r = {**r, "text": full_text.strip(), "has_adjacent": True}

        enriched.append(r)

    return enriched


_EXPAND_PROMPT = (
    "Rewrite this search query in 2 different ways to improve retrieval. "
    "Each rewrite should capture the same intent but use different terms or phrasing. "
    "Return ONLY the 2 rewrites, one per line, no numbering, no explanation.\n\n"
    "Query: {query}"
)


def expand_query(query: str) -> list[str]:
    """Generate 2 query reformulations via LLM for broader recall.

    Returns list of alternative queries (may be empty on failure).
    Fast path: ~200ms on Groq.
    """
    try:
        raw = quick_complete(_EXPAND_PROMPT.format(query=query), max_tokens=150, timeout=10)
        lines = [
            line.strip()
            for line in raw.strip().split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        # Filter out lines that are just the original query
        expansions = [line for line in lines if line.lower() != query.lower()]
        return expansions[:2]
    except Exception as e:
        logger.warning("[orchestrator] query expansion failed: %s", e)
        return []


def execute_search(
    query: str,
    route: RoutePlan,
    category: str | None = None,
    use_expansion: bool = False,
) -> list[dict[str, Any]]:
    """Execute hybrid retrieval (dense + BM25) + reranking.

    Four-stage pipeline:
    1. Dense search (semantic embeddings)
    1b. Query expansion search (LLM-generated reformulations, merged via RRF)
    2. BM25 search (keyword matching)
    3. RRF merge + cross-encoder reranking
    """
    from .config import OVERFETCH_FACTOR

    col = get_index(route.indexes[0])
    if col.count() == 0:
        return []

    # Per-mode overfetch: code queries need more candidates for exact name matching
    overfetch = OVERFETCH_FACTOR
    if route.mode == "code":
        overfetch = max(overfetch, 8)

    where = {"category": category} if category else None
    fetch_n = (
        min(route.top_k * overfetch, col.count())
        if route.use_reranker
        else min(route.top_k, col.count())
    )

    # Stage 1 — Dense (bi-encoder) search with original query
    results = col.query(query_texts=[query], n_results=fetch_n, where=where)
    dense_candidates = []
    if results and results["documents"]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        ):
            dense_candidates.append(
                {
                    "text": doc,
                    "category": meta.get("category", ""),
                    "subcategory": meta.get("subcategory", ""),
                    "source": meta.get("source", ""),
                    "file_type": meta.get("file_type", ""),
                    "chunk_index": meta.get("chunk_index", "-1"),
                    "bi_score": round(1.0 - dist, 4),
                    "bm25_score": 0.0,
                    "score": round(1.0 - dist, 4),
                }
            )

    # Stage 1b is handled post-reranking (see late fusion below)

    # Stage 2 — BM25 (keyword) search
    bm25_candidates = []
    if route.use_bm25:
        try:
            bm25_candidates = _bm25_search(query, col, fetch_n, where)
            logger.info(
                "[orchestrator] BM25 found %d candidates (dense: %d)",
                len(bm25_candidates),
                len(dense_candidates),
            )
        except Exception as e:
            logger.warning("[orchestrator] BM25 search failed: %s", e)

    # Stage 3 — Hybrid merge via RRF
    if bm25_candidates:
        candidates = _merge_hybrid(dense_candidates, bm25_candidates, fetch_n)
    else:
        candidates = dense_candidates

    # Stage 4 — Cross-encoder reranking
    if route.use_reranker and candidates:
        try:
            reranker = get_reranker(route.reranker_model)
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.predict(
                pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False
            )
            for c, score in zip(candidates, scores, strict=False):
                c["score"] = round(float(score), 4)
            candidates.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            pass  # fallback: keep RRF ordering

    # Stage 4b — Source diversity: max 2 chunks per source
    # Prevents a single dominant source from consuming all top-k slots
    MAX_PER_SOURCE = 3
    source_counts: dict[str, int] = {}
    diverse_candidates = []
    for c in candidates:
        src = c.get("source", "")
        count = source_counts.get(src, 0)
        if count < MAX_PER_SOURCE:
            diverse_candidates.append(c)
            source_counts[src] = count + 1
    primary_results = diverse_candidates[: route.top_k]

    # Stage 5 — Late fusion with query expansion
    # Run full independent searches for each expansion query,
    # then merge via RRF (rank-based, score-scale invariant)
    if use_expansion:
        expansions = expand_query(query)
        if expansions:
            expansion_results = []
            for exp_q in expansions:
                try:
                    # Recursive call WITHOUT expansion to avoid infinite loop
                    exp_candidates = execute_search(
                        exp_q, route, category=category, use_expansion=False
                    )
                    expansion_results.extend(exp_candidates)
                except Exception as e:
                    logger.warning("[orchestrator] expansion search failed: %s", e)
            if expansion_results:
                # RRF merge: primary results + expansion results
                primary_results = _merge_hybrid(
                    primary_results,
                    expansion_results,
                    route.top_k,
                )
                # Re-score merged results with reranker against ORIGINAL query
                if route.use_reranker and primary_results:
                    try:
                        reranker = get_reranker(route.reranker_model)
                        pairs = [[query, c["text"]] for c in primary_results]
                        scores = reranker.predict(
                            pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False
                        )
                        for c, score in zip(primary_results, scores, strict=False):
                            c["score"] = round(float(score), 4)
                        primary_results.sort(key=lambda x: x["score"], reverse=True)
                    except Exception:
                        pass
                logger.info(
                    "[orchestrator] late fusion: merged %d expansion candidates",
                    len(expansion_results),
                )

    return primary_results[: route.top_k]


def format_context(results: list[dict[str, Any]]) -> str:
    """Format results into a context block for the LLM prompt."""
    if not results:
        return "[no relevant context found]"

    parts = []
    for r in results:
        path = (
            f"{r['category']}/{r['source']}"
            if r["subcategory"] == ""
            else f"{r['category']}/{r['subcategory']}/{r['source']}"
        )
        parts.append(f"[{path}] (relevance: {r['score']:.0%})\n{r['text']}")

    return "\n\n---\n\n".join(parts)
