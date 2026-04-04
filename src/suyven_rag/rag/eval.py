"""Auto-evaluation — captures quality signals per query, flags failures.

Zero latency impact: all signals are already computed during the normal
query flow. This module observes and records, never adds work to the
user-facing path.

Writes JSON lines to data/eval/query_log.jsonl (fail-silent).
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (internal tuning knobs, not user config)
# ---------------------------------------------------------------------------

RERANKER_FLOOR = -2.0  # logit below which ALL results = retrieval failure
RERANKER_WEAK_MEAN = -0.5  # logit below which mean = weak retrieval
LATENCY_SPIKE_S = 10.0  # total seconds before flagging
CONTAMINATION_CATS = 3  # distinct categories = suspect on non-summary
MAX_ANSWER_LENGTH = 2000  # chars before suspecting wrong routing

# ---------------------------------------------------------------------------
# Eval record
# ---------------------------------------------------------------------------


@dataclass
class QueryEvalRecord:
    # Identity
    timestamp: str
    query_id: str

    # Input
    query: str
    category_filter: str | None
    top_k: int

    # Route
    route_mode: str
    route_reason: str
    route_indexes: list[str]

    # Retrieval signals
    num_results: int
    reranker_scores: list[float]
    bi_encoder_scores: list[float]
    max_reranker_score: float | None
    min_reranker_score: float | None
    mean_reranker_score: float | None
    source_categories: list[str]

    # Generation signals
    llm_said_insufficient: bool
    response_length: int
    token_count_approx: int

    # Performance
    latency_total_s: float
    latency_retrieval_s: float
    latency_llm_s: float

    # Flags
    flags: list[str] = field(default_factory=list)


def new_query_id() -> str:
    return uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Insufficient-context detection
# ---------------------------------------------------------------------------

_INSUFFICIENT_RE = re.compile(
    r"("
    r"context\s+(is\s+)?insufficient"
    r"|not\s+enough\s+information"
    r"|no\s+(relevant\s+)?context\s+found"
    r"|cannot\s+(answer|compare|determine)\s+from\s+the\s+provided\s+context"
    r"|no\s+information\s+(is\s+)?provided"
    r"|no\s+tengo\s+suficiente\s+contexto"
    r"|informaci[oó]n\s+insuficiente"
    r")",
    re.IGNORECASE,
)


def detect_insufficient(response_text: str) -> bool:
    """Check if the LLM response indicates context was insufficient."""
    return bool(_INSUFFICIENT_RE.search(response_text))


# ---------------------------------------------------------------------------
# Heuristic flagging
# ---------------------------------------------------------------------------


def compute_flags(record: QueryEvalRecord) -> list[str]:
    """Apply heuristic rules to flag likely failures. Pure function."""
    flags: list[str] = []

    # Empty retrieval
    if record.num_results == 0:
        flags.append("empty_retrieval")
        return flags  # nothing else to check

    # Retrieval failure — all scores below floor
    if record.reranker_scores and all(s < RERANKER_FLOOR for s in record.reranker_scores):
        flags.append("retrieval_failure")

    # Weak retrieval — mean below threshold
    if record.mean_reranker_score is not None and record.mean_reranker_score < RERANKER_WEAK_MEAN:
        flags.append("weak_retrieval")

    # Corpus gap — LLM explicitly says insufficient
    if record.llm_said_insufficient:
        flags.append("corpus_gap")

    # Category contamination — too many distinct categories (non-summary)
    if record.route_mode != "summary":
        distinct = len(set(record.source_categories))
        if distinct >= CONTAMINATION_CATS:
            flags.append("category_contamination")

    # Latency spike
    if record.latency_total_s > LATENCY_SPIKE_S:
        flags.append("latency_spike")

    return flags


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parents[1] / "data" / "eval"
_LOG_FILE = _LOG_DIR / "query_log.jsonl"


def log_eval(record: QueryEvalRecord) -> None:
    """Append one JSON line to the eval log. Non-blocking, fail-silent."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("eval log write failed", exc_info=True)
