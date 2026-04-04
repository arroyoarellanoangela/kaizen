"""Multi-agent RAG pipeline — 4 coordinating agents with retry loop.

Agents:
  RouterAgent         — classifies query complexity, picks retrieval strategy
  RetrieverAgent      — executes search with strategy, reports quality
  ReACTRetrieverAgent — multi-tool retriever (A-RAG pattern): semantic + keyword
                        + entity extraction + query decomposition + chunk read
  GeneratorAgent      — generates response, adapts prompt to retrieval quality
  EvaluatorAgent      — flags issues, decides retry, logs eval record

Coordination:
  Router -> Retriever -> Generator -> Evaluator
  If Evaluator says should_retry and attempt < max_attempts: loop back to Router

ReACT retriever tools (heuristic, no LLM):
  1. semantic_search — dense embedding search
  2. keyword_search  — BM25 keyword search
  3. entity_search   — extract entities from query, search each
  4. sub_query       — decompose complex queries, search sub-queries
  5. chunk_read      — fetch adjacent chunks for top results
"""

import hashlib
import logging
import re
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import UTC
from statistics import mean as _mean

from .config import (
    FALLBACK_API_KEY,
    FALLBACK_API_URL,
    FALLBACK_MODEL,
    FALLBACK_PROMPT,
    FALLBACK_PROVIDER,
    SYSTEM_PROMPT,
    TOP_K,
)
from .eval import (
    RERANKER_FLOOR,
    RERANKER_WEAK_MEAN,
    QueryEvalRecord,
    compute_flags,
    detect_insufficient,
    log_eval,
    new_query_id,
)
from .orchestrator import execute_search, format_context, plan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AgentContext — shared state between agents
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    # Input
    query: str
    category: str | None = None
    top_k: int = TOP_K
    query_id: str = ""

    # Router output
    route: object | None = None  # RoutePlan
    strategy: str = "dense"  # "dense" | "hybrid" | "category_filtered"
    complexity: str = "simple"  # "simple" | "moderate" | "complex"

    # Retriever output
    results: list[dict] = field(default_factory=list)
    context_text: str = ""
    retrieval_quality: str = "unknown"  # "good" | "weak" | "failed"
    reranker_scores: list[float] = field(default_factory=list)
    bi_encoder_scores: list[float] = field(default_factory=list)

    # Generator output
    response_tokens: list[str] = field(default_factory=list)
    full_response: str = ""
    llm_said_insufficient: bool = False

    # Evaluator output
    eval_flags: list[str] = field(default_factory=list)
    should_retry: bool = False
    retry_reason: str = ""
    retry_strategy: str = ""

    # Coordination state
    attempt: int = 1
    max_attempts: int = 3
    agent_trace: list[dict] = field(default_factory=list)

    # Timing
    t_start: float = 0.0
    t_retrieval: float = 0.0
    t_llm: float = 0.0
    t_total: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def classify_complexity(query: str) -> str:
    """Heuristic query complexity classification. No LLM call."""
    words = query.split()
    n_words = len(words)
    n_questions = query.count("?")
    has_conjunction = bool(
        re.search(r"\b(and|or|but|also|ademas|tambien|y|o)\b", query, re.IGNORECASE)
    )
    has_semicolon = ";" in query
    has_comparison = bool(
        re.search(r"\b(compare|vs|versus|diferencia|difference)\b", query, re.IGNORECASE)
    )

    score = 0
    if n_words > 20:
        score += 1
    if n_words > 40:
        score += 1
    if n_questions > 1:
        score += 1
    if has_conjunction:
        score += 1
    if has_semicolon:
        score += 1
    if has_comparison:
        score += 1

    if score >= 3:
        return "complex"
    elif score >= 1:
        return "moderate"
    return "simple"


def assess_quality(reranker_scores: list[float]) -> str:
    """Assess retrieval quality from reranker scores."""
    if not reranker_scores:
        return "failed"
    if all(s < RERANKER_FLOOR for s in reranker_scores):
        return "failed"
    if _mean(reranker_scores) < RERANKER_WEAK_MEAN:
        return "weak"
    return "good"


def pick_next_strategy(current: str, has_category: bool) -> str:
    """Escalation: dense -> hybrid -> no-category dense."""
    if current == "dense":
        return "hybrid"
    if current == "hybrid" and has_category:
        return "dense"  # will clear category
    if current == "category_filtered":
        return "hybrid"
    return "dense"


def _merge_and_dedup(primary: list[dict], secondary: list[dict], top_k: int) -> list[dict]:
    """Merge two result lists, deduplicate by text hash, sort by score."""
    seen = set()
    merged = []
    for r in primary + secondary:
        h = hashlib.md5(r["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            merged.append(r)
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


# ---------------------------------------------------------------------------
# RouterAgent
# ---------------------------------------------------------------------------


class RouterAgent:
    name = "router"
    role = "router"

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()

        ctx.complexity = classify_complexity(ctx.query)

        # On retry, use Evaluator's recommendation
        if ctx.attempt > 1 and ctx.retry_strategy:
            ctx.strategy = ctx.retry_strategy
            # If evaluator recommended clearing category
            if (
                ctx.strategy == "dense"
                and ctx.attempt > 1
                and ctx.retry_reason
                and "no_category" in ctx.retry_reason
            ):
                ctx.category = None
        elif ctx.category:
            ctx.strategy = "category_filtered"
        elif ctx.complexity == "complex":
            ctx.strategy = "hybrid"
        else:
            ctx.strategy = "dense"

        # Build route using existing orchestrator
        tk = ctx.top_k * 2 if ctx.strategy == "hybrid" else ctx.top_k
        ctx.route = plan(ctx.query, category=ctx.category, top_k=tk)

        ctx.agent_trace.append(
            {
                "agent": self.name,
                "action": "route",
                "attempt": ctx.attempt,
                "strategy": ctx.strategy,
                "complexity": ctx.complexity,
                "mode": ctx.route.mode,
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )

        logger.info(
            "[%s] attempt=%d strategy=%s complexity=%s mode=%s",
            self.name,
            ctx.attempt,
            ctx.strategy,
            ctx.complexity,
            ctx.route.mode,
        )
        return ctx


# ---------------------------------------------------------------------------
# RetrieverAgent
# ---------------------------------------------------------------------------


class RetrieverAgent:
    name = "retriever"
    role = "retriever"

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()

        results = execute_search(ctx.query, ctx.route, category=ctx.category)

        # Hybrid: if category-filtered returned few results, try without
        if ctx.strategy == "hybrid" and ctx.category and len(results) < ctx.top_k:
            more = execute_search(ctx.query, ctx.route, category=None)
            results = _merge_and_dedup(results, more, ctx.top_k)
        elif ctx.strategy == "hybrid" and len(results) > ctx.top_k:
            results = results[: ctx.top_k]

        ctx.results = results
        ctx.context_text = format_context(results)
        ctx.reranker_scores = [r["score"] for r in results]
        ctx.bi_encoder_scores = [r.get("bi_score", 0.0) for r in results]
        ctx.retrieval_quality = assess_quality(ctx.reranker_scores)
        ctx.t_retrieval = time.time() - ctx.t_start

        ctx.agent_trace.append(
            {
                "agent": self.name,
                "action": "retrieve",
                "attempt": ctx.attempt,
                "strategy": ctx.strategy,
                "num_results": len(results),
                "quality": ctx.retrieval_quality,
                "mean_score": round(_mean(ctx.reranker_scores), 4) if ctx.reranker_scores else None,
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )

        logger.info(
            "[%s] attempt=%d results=%d quality=%s",
            self.name,
            ctx.attempt,
            len(results),
            ctx.retrieval_quality,
        )
        return ctx


# ---------------------------------------------------------------------------
# ReACT Retriever — multi-tool retrieval inspired by A-RAG
# ---------------------------------------------------------------------------

# Entity patterns for query entity extraction (subset of finetune/entity_pairs.py)
_ENTITY_PATTERNS = [
    re.compile(
        r"\b(Amazon\s+\w+|AWS\s+\w+|S3|EC2|Lambda|DynamoDB|SageMaker|"
        r"CloudFormation|ECS|EKS|RDS|SNS|SQS|IAM|VPC|CloudWatch|"
        r"Kinesis|Redshift|Glue|Athena|EMR|Step\s+Functions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(transformer|attention\s+mechanism|BERT|GPT|LLM|embedding|"
        r"fine-tuning|LoRA|RAG|retrieval|reranker|cross-encoder|"
        r"bi-encoder|tokenizer|neural\s+network|CNN|RNN|LSTM|GAN)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(Docker|Kubernetes|Terraform|Ansible|Jenkins|GitHub\s+Actions|"
        r"CI/CD|microservices|load\s+balancer|auto\s*scaling|serverless|"
        r"container|service\s+mesh)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Apache\s+Kafka|"
        r"Apache\s+Spark|Hadoop|ChromaDB|Pinecone|Weaviate|FAISS|"
        r"vector\s+database|data\s+lake|data\s+warehouse|ETL)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(FastAPI|Flask|Django|React|Vue\.js|PyTorch|TensorFlow|"
        r"pandas|numpy|scikit-learn|sentence-transformers|HuggingFace)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(CAP\s+theorem|ACID|BASE|MapReduce|sharding|partitioning|"
        r"replication|consensus|Raft|Paxos|eventual\s+consistency|"
        r"idempotency|circuit\s+breaker|rate\s+limiting|caching|CDN)\b",
        re.IGNORECASE,
    ),
]

# Decomposition patterns for complex queries
_CONJUNCTION_SPLIT = re.compile(
    r"\s*(?:\band\b|\by\b|\balso\b|;|\?)\s*",
    re.IGNORECASE,
)


def extract_query_entities(query: str) -> list[str]:
    """Extract technical entities from a query string."""
    entities = set()
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(query):
            e = match.group(0).strip()
            if len(e) >= 2:
                entities.add(e)
    return list(entities)


def decompose_query(query: str) -> list[str]:
    """Split a complex query into sub-queries. Returns [] if not decomposable."""
    parts = _CONJUNCTION_SPLIT.split(query)
    # Filter out fragments that are too short to be meaningful
    subs = [p.strip().rstrip("?").strip() for p in parts if len(p.strip()) > 10]
    # Only decompose if we got 2+ meaningful sub-queries
    if len(subs) >= 2 and len(subs) <= 5:
        return subs
    return []


class ReACTRetrieverAgent:
    """Multi-tool retriever inspired by A-RAG's hierarchical retrieval.

    Heuristic reasoning loop (no LLM calls):
    1. semantic_search -> assess quality
    2. If weak: keyword_search (entity-focused) -> merge
    3. If complex: decompose query -> search sub-queries -> merge
    4. chunk_read on top results for context expansion

    Each tool call is logged in agent_trace for full observability.
    """

    name = "react_retriever"
    role = "retriever"

    def _tool_semantic(self, query: str, route, category: str | None) -> list[dict]:
        """Tool 1: Dense semantic search."""
        return execute_search(query, route, category=category)

    def _tool_entity_search(
        self,
        entities: list[str],
        route,
        category: str | None,
        top_k: int,
    ) -> list[dict]:
        """Tool 2: Entity-focused search — search for each entity independently."""
        all_results = []
        per_entity_k = max(2, top_k // len(entities)) if entities else 0
        for entity in entities[:4]:  # cap at 4 entities
            entity_query = f"What is {entity}?"
            try:
                results = execute_search(entity_query, route, category=category)
                all_results.extend(results[:per_entity_k])
            except Exception:
                pass
        return all_results

    def _tool_sub_query(
        self,
        sub_queries: list[str],
        route,
        category: str | None,
        top_k: int,
    ) -> list[dict]:
        """Tool 3: Sub-query search — search each decomposed query independently."""
        all_results = []
        per_sub_k = max(2, top_k // len(sub_queries)) if sub_queries else 0
        for sub_q in sub_queries[:4]:  # cap at 4 sub-queries
            try:
                results = execute_search(sub_q, route, category=category)
                all_results.extend(results[:per_sub_k])
            except Exception:
                pass
        return all_results

    def _tool_chunk_read(self, results: list[dict], route) -> list[dict]:
        """Tool 4: Adjacent chunk read — expand top results with neighboring chunks."""
        from .index_registry import get_index
        from .orchestrator import _fetch_adjacent_chunks

        col = get_index(route.indexes[0])
        return _fetch_adjacent_chunks(col, results[:3], window=1)

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()
        tools_used = []
        all_candidates = []

        # ---- Step 1: Semantic search (always) ----
        semantic_results = self._tool_semantic(ctx.query, ctx.route, ctx.category)
        all_candidates.extend(semantic_results)
        tools_used.append({"tool": "semantic_search", "results": len(semantic_results)})

        # Assess intermediate quality
        semantic_scores = [r["score"] for r in semantic_results]
        intermediate_quality = assess_quality(semantic_scores)

        # ---- Step 2: Entity search (if entities found and quality is not great) ----
        entities = extract_query_entities(ctx.query)
        if entities and intermediate_quality in ("weak", "failed"):
            entity_results = self._tool_entity_search(
                entities,
                ctx.route,
                ctx.category,
                ctx.top_k,
            )
            all_candidates.extend(entity_results)
            tools_used.append(
                {
                    "tool": "entity_search",
                    "entities": entities,
                    "results": len(entity_results),
                }
            )

        # ---- Step 3: Query decomposition (if complex) ----
        if ctx.complexity == "complex":
            sub_queries = decompose_query(ctx.query)
            if sub_queries:
                sub_results = self._tool_sub_query(
                    sub_queries,
                    ctx.route,
                    ctx.category,
                    ctx.top_k,
                )
                all_candidates.extend(sub_results)
                tools_used.append(
                    {
                        "tool": "sub_query",
                        "sub_queries": sub_queries,
                        "results": len(sub_results),
                    }
                )

        # ---- Merge & deduplicate all candidates ----
        if len(tools_used) > 1:
            # Multiple tools used — merge via score-based dedup
            merged = _merge_and_dedup(all_candidates, [], ctx.top_k)
        else:
            merged = all_candidates[: ctx.top_k]

        # ---- Step 4: Chunk read for context expansion (if quality is good enough) ----
        final_scores = [r["score"] for r in merged]
        final_quality = assess_quality(final_scores)
        if final_quality == "good" and merged:
            try:
                expanded = self._tool_chunk_read(merged, ctx.route)
                # Replace top results with expanded versions
                expanded_map = {r["text"][:200]: r for r in expanded}
                for i, r in enumerate(merged):
                    key = r["text"][:200]
                    if key in expanded_map and expanded_map[key].get("has_adjacent"):
                        merged[i] = expanded_map[key]
                tools_used.append({"tool": "chunk_read", "expanded": len(expanded)})
            except Exception:
                pass  # chunk read is optional enrichment

        ctx.results = merged
        ctx.context_text = format_context(merged)
        ctx.reranker_scores = [r["score"] for r in merged]
        ctx.bi_encoder_scores = [r.get("bi_score", 0.0) for r in merged]
        ctx.retrieval_quality = assess_quality(ctx.reranker_scores)
        ctx.t_retrieval = time.time() - ctx.t_start

        ctx.agent_trace.append(
            {
                "agent": self.name,
                "action": "react_retrieve",
                "attempt": ctx.attempt,
                "strategy": ctx.strategy,
                "tools_used": tools_used,
                "num_tools": len(tools_used),
                "num_results": len(merged),
                "quality": ctx.retrieval_quality,
                "mean_score": round(_mean(ctx.reranker_scores), 4) if ctx.reranker_scores else None,
                "entities_found": entities,
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )

        logger.info(
            "[%s] attempt=%d tools=%s results=%d quality=%s",
            self.name,
            ctx.attempt,
            [t["tool"] for t in tools_used],
            len(merged),
            ctx.retrieval_quality,
        )
        return ctx


# ---------------------------------------------------------------------------
# GeneratorAgent
# ---------------------------------------------------------------------------


_WEAK_CAVEAT = (
    "\n\nIMPORTANT: The retrieved context may be incomplete or only partially relevant. "
    "If the context does not contain enough information to answer accurately, "
    "explicitly state that the context is insufficient."
)

_FAILED_PROMPT = (
    "You are a technical knowledge assistant. No relevant context was found for this query. "
    "Respond honestly that you could not find relevant information in the knowledge base."
)


class GeneratorAgent:
    name = "generator"
    role = "generator"

    def _use_fallback(self, quality: str) -> bool:
        """Should we use the fallback LLM (e.g. Gemini) instead of primary?"""
        return quality == "failed" and bool(FALLBACK_PROVIDER and FALLBACK_MODEL)

    def _build_prompt(self, quality: str) -> str:
        if self._use_fallback(quality):
            return FALLBACK_PROMPT
        if quality == "failed":
            return _FAILED_PROMPT
        if quality == "weak":
            return SYSTEM_PROMPT + _WEAK_CAVEAT
        return SYSTEM_PROMPT

    def _llm_kwargs(self, quality: str) -> dict:
        """Extra kwargs for stream_chat when using fallback provider."""
        if self._use_fallback(quality):
            return {
                "provider": FALLBACK_PROVIDER,
                "model": FALLBACK_MODEL,
                "api_url": FALLBACK_API_URL,
                "api_key": FALLBACK_API_KEY,
            }
        return {}

    def execute(self, ctx: AgentContext) -> AgentContext:
        """Batch mode: collect full response. Used by bench.py."""
        from .llm import stream_chat

        t0 = time.time()
        quality = ctx.retrieval_quality
        prompt = self._build_prompt(quality)
        kwargs = self._llm_kwargs(quality)
        is_fallback = self._use_fallback(quality)

        # Fallback: no RAG context, just the query
        context = "" if is_fallback else ctx.context_text
        tokens = list(stream_chat(ctx.query, context, system_prompt=prompt, **kwargs))
        ctx.response_tokens = tokens
        ctx.full_response = "".join(tokens)
        ctx.llm_said_insufficient = detect_insufficient(ctx.full_response)
        ctx.t_llm = time.time() - t0

        ctx.agent_trace.append(
            {
                "agent": self.name,
                "action": "generate_fallback" if is_fallback else "generate",
                "attempt": ctx.attempt,
                "quality_prompt": quality,
                "fallback": is_fallback,
                "fallback_model": FALLBACK_MODEL if is_fallback else None,
                "response_length": len(ctx.full_response),
                "insufficient": ctx.llm_said_insufficient,
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )

        logger.info(
            "[%s] attempt=%d chars=%d insufficient=%s fallback=%s",
            self.name,
            ctx.attempt,
            len(ctx.full_response),
            ctx.llm_said_insufficient,
            is_fallback,
        )
        return ctx

    def stream(self, ctx: AgentContext) -> Generator[str, None, None]:
        """SSE mode: yield tokens one by one. Fills ctx when done."""
        from .llm import stream_chat

        t0 = time.time()
        quality = ctx.retrieval_quality
        prompt = self._build_prompt(quality)
        kwargs = self._llm_kwargs(quality)
        is_fallback = self._use_fallback(quality)

        context = "" if is_fallback else ctx.context_text
        tokens = []
        for token in stream_chat(ctx.query, context, system_prompt=prompt, **kwargs):
            tokens.append(token)
            yield token
        ctx.response_tokens = tokens
        ctx.full_response = "".join(tokens)
        ctx.llm_said_insufficient = detect_insufficient(ctx.full_response)
        ctx.t_llm = time.time() - t0

        ctx.agent_trace.append(
            {
                "agent": self.name,
                "action": "generate_fallback" if is_fallback else "generate",
                "attempt": ctx.attempt,
                "quality_prompt": quality,
                "fallback": is_fallback,
                "fallback_model": FALLBACK_MODEL if is_fallback else None,
                "response_length": len(ctx.full_response),
                "insufficient": ctx.llm_said_insufficient,
                "duration_ms": round(ctx.t_llm * 1000, 1),
            }
        )


# ---------------------------------------------------------------------------
# EvaluatorAgent
# ---------------------------------------------------------------------------


_RETRYABLE_FLAGS = {"empty_retrieval", "retrieval_failure", "weak_retrieval"}


class EvaluatorAgent:
    name = "evaluator"
    role = "evaluator"

    def _build_eval_record(self, ctx: AgentContext) -> QueryEvalRecord:
        from datetime import datetime

        scores = ctx.reranker_scores
        return QueryEvalRecord(
            timestamp=datetime.now(UTC).isoformat(),
            query_id=ctx.query_id,
            query=ctx.query,
            category_filter=ctx.category,
            top_k=ctx.top_k,
            route_mode=ctx.route.mode if ctx.route else "unknown",
            route_reason=ctx.route.reason if ctx.route else "",
            route_indexes=ctx.route.indexes if ctx.route else [],
            num_results=len(ctx.results),
            reranker_scores=scores,
            bi_encoder_scores=ctx.bi_encoder_scores,
            max_reranker_score=max(scores) if scores else None,
            min_reranker_score=min(scores) if scores else None,
            mean_reranker_score=_mean(scores) if scores else None,
            source_categories=list({r.get("category", "") for r in ctx.results}),
            llm_said_insufficient=ctx.llm_said_insufficient,
            response_length=len(ctx.full_response),
            token_count_approx=len(ctx.full_response) // 4,
            latency_total_s=round(time.time() - ctx.t_start, 4),
            latency_retrieval_s=round(ctx.t_retrieval, 4),
            latency_llm_s=round(ctx.t_llm, 4),
        )

    def execute(self, ctx: AgentContext) -> AgentContext:
        t0 = time.time()

        record = self._build_eval_record(ctx)
        flags = compute_flags(record)

        # Tag with attempt and strategy info
        if ctx.attempt > 1:
            flags.append(f"retry_{ctx.attempt}")
        flags.append(f"strategy_{ctx.strategy}")

        record.flags = flags
        ctx.eval_flags = flags

        # Retry decision
        retryable = _RETRYABLE_FLAGS & set(flags)
        if retryable and ctx.attempt < ctx.max_attempts:
            next_strategy = pick_next_strategy(ctx.strategy, ctx.category is not None)
            # Don't retry with the same strategy
            if next_strategy != ctx.strategy:
                ctx.should_retry = True
                ctx.retry_strategy = next_strategy
                ctx.retry_reason = f"flags={sorted(retryable)}, switch to {next_strategy}"
                if next_strategy == "dense" and ctx.category:
                    ctx.retry_reason += " (no_category)"
            else:
                ctx.should_retry = False
        else:
            ctx.should_retry = False

        # Always log
        log_eval(record)

        ctx.agent_trace.append(
            {
                "agent": self.name,
                "action": "evaluate",
                "attempt": ctx.attempt,
                "flags": flags,
                "should_retry": ctx.should_retry,
                "retry_reason": ctx.retry_reason if ctx.should_retry else "",
                "duration_ms": round((time.time() - t0) * 1000, 1),
            }
        )

        logger.info(
            "[%s] attempt=%d flags=%s retry=%s",
            self.name,
            ctx.attempt,
            flags,
            ctx.should_retry,
        )
        return ctx


# ---------------------------------------------------------------------------
# Coordination loops
# ---------------------------------------------------------------------------


def run_agent_pipeline(
    query: str,
    category: str | None = None,
    top_k: int = TOP_K,
    query_id: str | None = None,
    skip_generation: bool = False,
    use_react: bool = False,
) -> AgentContext:
    """Run the full multi-agent pipeline (batch mode). Returns completed AgentContext.

    Args:
        skip_generation: If True, skip GeneratorAgent (retrieval-only mode for benchmarks).
        use_react: If True, use ReACTRetrieverAgent (multi-tool) instead of simple RetrieverAgent.
    """
    ctx = AgentContext(
        query=query,
        category=category,
        top_k=top_k,
        query_id=query_id or new_query_id(),
        t_start=time.time(),
    )

    router = RouterAgent()
    retriever = ReACTRetrieverAgent() if use_react else RetrieverAgent()
    generator = GeneratorAgent()
    evaluator = EvaluatorAgent()

    while ctx.attempt <= ctx.max_attempts:
        router.execute(ctx)
        retriever.execute(ctx)
        if not skip_generation:
            generator.execute(ctx)
        evaluator.execute(ctx)

        if not ctx.should_retry:
            break

        # Prepare for retry — keep trace, reset outputs
        ctx.attempt += 1
        ctx.results = []
        ctx.context_text = ""
        ctx.response_tokens = []
        ctx.full_response = ""
        ctx.retrieval_quality = "unknown"
        ctx.should_retry = False

    ctx.t_total = time.time() - ctx.t_start
    return ctx


def prepare_agent_context(
    query: str,
    category: str | None = None,
    top_k: int = TOP_K,
    query_id: str | None = None,
    use_react: bool = False,
) -> tuple[
    AgentContext,
    RouterAgent,
    "RetrieverAgent | ReACTRetrieverAgent",
    GeneratorAgent,
    EvaluatorAgent,
]:
    """Create context and agents for SSE streaming (split flow).

    Usage in api.py:
        ctx, router, retriever, generator, evaluator = prepare_agent_context(...)
        router.execute(ctx)
        retriever.execute(ctx)
        # pre-flight retry if needed
        if ctx.retrieval_quality in ("weak", "failed") and ctx.attempt < ctx.max_attempts:
            evaluator.execute(ctx)
            if ctx.should_retry:
                ctx.attempt += 1; ctx.results = []; ...
                router.execute(ctx)
                retriever.execute(ctx)
        # then stream: for token in generator.stream(ctx): yield token
        # then: evaluator.execute(ctx)
    """
    ctx = AgentContext(
        query=query,
        category=category,
        top_k=top_k,
        query_id=query_id or new_query_id(),
        t_start=time.time(),
    )
    retriever = ReACTRetrieverAgent() if use_react else RetrieverAgent()
    return ctx, RouterAgent(), retriever, GeneratorAgent(), EvaluatorAgent()
