"""Orchestrator — decides HOW to handle each query.

Receives a query and returns a RoutePlan: which index, models, and mode to use.
V2.1: deterministic rules, no AI in the router.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from .config import LLM_MODEL, RERANKER_BATCH_SIZE, TOP_K
from .index_registry import get_index, route_to_index
from .llm import stream_chat
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
    indexes: list[str]           # which collections to query
    embed_model: str             # which embed model to use
    use_reranker: bool           # whether to apply cross-encoder
    reranker_model: str          # which reranker to use
    llm_model: str               # which LLM for generation
    mode: str                    # "answer" | "summary" | "code"
    top_k: int                   # how many results to return
    reason: str                  # why this route was chosen (traceability)


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
    )

    logger.info(
        '[orchestrator] query="%s" → indexes=%s embed=%s reranker=%s llm=%s mode=%s reason="%s"',
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


def execute_search(
    query: str,
    route: RoutePlan,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Execute retrieval + reranking according to the RoutePlan."""
    from .config import OVERFETCH_FACTOR

    col = get_index(route.indexes[0])
    if col.count() == 0:
        return []

    # Stage 1 — bi-encoder
    fetch_n = min(route.top_k * OVERFETCH_FACTOR, col.count()) if route.use_reranker else min(route.top_k, col.count())
    where = {"category": category} if category else None

    results = col.query(query_texts=[query], n_results=fetch_n, where=where)
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
            "score": round(1.0 - dist, 4),
        })

    # Stage 2 — cross-encoder reranking
    if route.use_reranker and candidates:
        try:
            reranker = get_reranker(route.reranker_model)
            pairs = [[query, c["text"]] for c in candidates]
            scores = reranker.predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)
            for c, score in zip(candidates, scores):
                c["score"] = round(float(score), 4)
            candidates.sort(key=lambda x: x["score"], reverse=True)
        except Exception:
            pass  # fallback: keep bi-encoder ordering

    return candidates[:route.top_k]


def format_context(results: list[dict[str, Any]]) -> str:
    """Format results into a context block for the LLM prompt."""
    if not results:
        return "[no relevant context found]"

    parts = []
    for r in results:
        path = f"{r['category']}/{r['source']}" if r["subcategory"] == "" else f"{r['category']}/{r['subcategory']}/{r['source']}"
        parts.append(f"[{path}] (relevance: {r['score']:.0%})\n{r['text']}")

    return "\n\n---\n\n".join(parts)
