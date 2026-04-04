"""Suyven v1 — FastAPI backend for the React frontend."""

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path as _Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from suyven_rag.rag.agents import (
    prepare_agent_context,
)
from suyven_rag.rag.config import (
    FALLBACK_MODEL,
    FALLBACK_PROVIDER,
    KNOWLEDGE_DIR,
    LLM_API_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
    WORKERS,
)
from suyven_rag.rag.domain_registry import (
    create_domain,
    delete_domain,
    get_domain,
    get_domain_prompt,
    list_domains,
    update_domain,
)
from suyven_rag.rag.gap_tracker import analyze_gaps, load_query_log
from suyven_rag.rag.index_registry import get_index, register_index, reset_index
from suyven_rag.rag.loader import iter_files
from suyven_rag.rag.model_registry import get_embed_model, list_models
from suyven_rag.rag.monitoring import gpu_metrics
from suyven_rag.rag.observability import (
    RequestIdFilter,
    configure_logging,
    create_request_middleware,
    metrics,
)
from suyven_rag.rag.pipeline import read_and_chunk
from suyven_rag.rag.security import (
    CORS_ORIGINS,
    rate_limiter,
    require_api_key,
    sanitize_text,
    validate_directory_path,
    validate_domain_name,
    validate_query,
    validate_slug,
    validate_top_k,
)
from suyven_rag.rag.store import add_chunks, ensure_ollama

# Structured JSON logging in production (LOG_FORMAT=json), plain text in dev
_log_format = os.getenv("LOG_FORMAT", "text").strip().lower()
configure_logging(json_logs=(_log_format == "json"))
# Inject request_id into all log records
logging.getLogger().addFilter(RequestIdFilter())

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_ollama()
    get_index()  # warm ChromaDB via index_registry
    yield


app = FastAPI(title="Suyven RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request tracing middleware — adds X-Request-ID, duration logging, metrics
app.add_middleware(BaseHTTPMiddleware, dispatch=create_request_middleware(metrics))

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    category: str | None = None
    use_react: bool = False
    domain: str | None = None  # domain slug — routes to domain-specific index


class IngestRequest(BaseModel):
    force: bool = False


class DomainCreateRequest(BaseModel):
    name: str
    description: str = ""
    language: str = "auto"
    system_prompt: str = ""
    categories: list[str] = []


class DomainUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    language: str | None = None
    system_prompt: str | None = None
    categories: list[str] | None = None


class DomainIngestRequest(BaseModel):
    directory: str  # path to files to ingest
    force: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health(api_key: str = Depends(require_api_key)):
    """Health heartbeat — quick liveness check for monitoring.

    Returns component status (ok/degraded/down) and key metrics.
    Designed for uptime monitors and alerting.
    """
    checks = {}

    # ChromaDB
    try:
        col = get_index()
        chunk_count = col.count()
        checks["chromadb"] = {"status": "ok", "chunks": chunk_count}
        if chunk_count == 0:
            checks["chromadb"]["status"] = "degraded"
    except Exception as e:
        checks["chromadb"] = {"status": "down", "error": str(e)[:100]}

    # GPU
    try:
        gm = gpu_metrics()
        if gm and gm.get("gpu_name"):
            checks["gpu"] = {
                "status": "ok",
                "name": gm["gpu_name"],
                "vram_used_gb": gm.get("vram_used_gb"),
                "vram_total_gb": gm.get("vram_total_gb"),
                "utilization_pct": gm.get("utilization_pct"),
            }
        else:
            checks["gpu"] = {"status": "degraded", "note": "no GPU detected"}
    except Exception as e:
        checks["gpu"] = {"status": "down", "error": str(e)[:100]}

    # Embed model
    try:
        models = list_models()
        embed_loaded = any(m.get("type") == "embed" and m.get("loaded") for m in models)
        checks["embed_model"] = {"status": "ok" if embed_loaded else "degraded"}
    except Exception:
        checks["embed_model"] = {"status": "unknown"}

    # LLM provider
    checks["llm"] = {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
    }
    if FALLBACK_MODEL:
        checks["llm"]["fallback"] = FALLBACK_MODEL

    # Overall
    statuses = [c.get("status", "unknown") for c in checks.values()]
    if all(s == "ok" for s in statuses):
        overall = "healthy"
    elif any(s == "down" for s in statuses):
        overall = "unhealthy"
    else:
        overall = "degraded"

    return {"status": overall, "checks": checks}


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    return PlainTextResponse(metrics.export_prometheus(), media_type="text/plain")


@app.get("/api/gaps")
def gaps(since_days: int | None = None, top: int = 20, api_key: str = Depends(require_api_key)):
    """Knowledge gap analysis — recurring retrieval failures."""
    import dataclasses

    entries = load_query_log(since_days=since_days)
    if not entries:
        return {"total_queries": 0, "gaps": [], "message": "No query log data yet"}
    report = analyze_gaps(entries, top_n=top)
    return dataclasses.asdict(report)


@app.get("/api/status")
def status(api_key: str = Depends(require_api_key)):
    """System status: chunk count, GPU metrics, models, system info."""
    col = get_index()
    # Derive a human-friendly provider label
    if LLM_PROVIDER == "ollama":
        provider_label = "Ollama (local)"
    elif LLM_API_URL:
        try:
            from urllib.parse import urlparse

            host = urlparse(LLM_API_URL).hostname or ""
            # groq.com → Groq, deepseek.com → DeepSeek, etc.
            domain = host.replace("api.", "").split(".")[0].capitalize()
            provider_label = domain
        except Exception:
            provider_label = "Cloud API"
    else:
        provider_label = LLM_PROVIDER

    return {
        "chunks": col.count(),
        "gpu": gpu_metrics(),
        "llm_model": LLM_MODEL,
        "llm_provider": LLM_PROVIDER,
        "provider_label": provider_label,
        "models": list_models(),
        "ollama_url": OLLAMA_URL,
        "fallback_model": FALLBACK_MODEL or None,
        "fallback_provider": FALLBACK_PROVIDER or None,
    }


@app.post("/api/query")
def query(body: QueryRequest, api_key: str = Depends(require_api_key)):
    """
    Query the knowledge base via multi-agent pipeline. Returns SSE stream:
      - data: {"type":"sources","sources":[...]}   (first)
      - data: {"type":"token","content":"..."}      (streaming tokens)
      - data: {"type":"done","agent_trace":[...]}   (end + agent trace)

    If body.domain is set, routes to that domain's isolated index.
    """
    # Auth + rate limit + validate
    rate_limiter.check(api_key)
    body.query = validate_query(body.query)
    body.top_k = validate_top_k(body.top_k)

    metrics.inc("suyven_queries_total")

    # If domain specified, delegate to domain-specific endpoint
    if body.domain:
        return query_domain(body.domain, body, api_key=api_key)

    ctx, router, retriever, generator, evaluator = prepare_agent_context(
        query=body.query,
        category=body.category,
        top_k=body.top_k,
        use_react=body.use_react,
    )

    # Phase 1: Router + Retriever (with pre-flight retry)
    router.execute(ctx)
    retriever.execute(ctx)

    # Pre-flight: if retrieval is bad, retry before streaming
    if ctx.retrieval_quality in ("weak", "failed") and ctx.attempt < ctx.max_attempts:
        evaluator.execute(ctx)
        if ctx.should_retry:
            metrics.inc("suyven_retries_total")
            ctx.attempt += 1
            ctx.results = []
            ctx.context_text = ""
            ctx.should_retry = False
            router.execute(ctx)
            retriever.execute(ctx)

    # No results at all — fallback to smart LLM if configured, else static message
    if not ctx.results:
        has_fallback = bool(FALLBACK_PROVIDER and FALLBACK_MODEL)

        def no_results():
            mode = ctx.route.mode if ctx.route else "answer"
            reason = ctx.route.reason if ctx.route else ""
            yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'route': {'mode': mode, 'reason': reason}, 'fallback': has_fallback})}\n\n"

            if has_fallback:
                # Use fallback LLM (e.g. Gemini) to answer from parametric knowledge
                ctx.retrieval_quality = "failed"
                try:
                    for token in generator.stream(ctx):
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'token', 'content': f'[Fallback LLM error: {e}]'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'content': 'No results found in the knowledge base.'})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'agent_trace': ctx.agent_trace})}\n\n"
            evaluator.execute(ctx)

        return StreamingResponse(no_results(), media_type="text/event-stream")

    # Build sources payload
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    sources = [
        {
            "category": r["category"],
            "source": r["source"],
            "score": round(_sigmoid(r["score"]), 3),
            "text": r["text"][:300],
        }
        for r in ctx.results
    ]

    def stream():
        mode = ctx.route.mode if ctx.route else "answer"
        reason = ctx.route.reason if ctx.route else ""
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'route': {'mode': mode, 'reason': reason}})}\n\n"

        # Stream tokens via GeneratorAgent
        try:
            for token in generator.stream(ctx):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            ctx.full_response += f"[LLM error: {e}]"
            yield f"data: {json.dumps({'type': 'token', 'content': f'[LLM error: {e}]'})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'agent_trace': ctx.agent_trace})}\n\n"

        # Post-stream: Evaluator logs eval record
        evaluator.execute(ctx)

        if ctx.eval_flags:
            for flag in ctx.eval_flags:
                metrics.inc("suyven_eval_flags_total", labels={"flag": flag})
            logging.getLogger(__name__).info(
                "[eval] query_id=%s flags=%s query=%s",
                ctx.query_id,
                ctx.eval_flags,
                body.query[:60],
            )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/ingest")
def ingest(body: IngestRequest, api_key: str = Depends(require_api_key)):
    """Ingest knowledge base. Blocking call — returns when done."""
    files = list(iter_files(KNOWLEDGE_DIR))
    if not files:
        return {"error": f"No files found in {KNOWLEDGE_DIR}"}

    target = reset_index() if body.force else get_index()

    # Phase 1: parallel read + chunk
    file_chunks = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(read_and_chunk, files):
            file_chunks.append(result)

    # Phase 2: ensure model loaded
    get_embed_model()

    # Phase 3: embed + add
    total_added = 0
    total_skipped = 0
    for path, chunks in file_chunks:
        if not chunks:
            continue
        added, skipped = add_chunks(target, path, chunks, KNOWLEDGE_DIR)
        total_added += added
        total_skipped += skipped

    return {
        "files": len(files),
        "added": total_added,
        "skipped": total_skipped,
        "total": target.count(),
    }


# ---------------------------------------------------------------------------
# Domain endpoints — multi-domain model factory
# ---------------------------------------------------------------------------


@app.post("/api/domains")
def create_domain_endpoint(body: DomainCreateRequest, api_key: str = Depends(require_api_key)):
    """Create a new knowledge domain with isolated vector space."""
    import dataclasses

    rate_limiter.check(api_key)
    body.name = validate_domain_name(body.name)
    body.description = sanitize_text(body.description, max_length=1000)
    body.system_prompt = sanitize_text(body.system_prompt, max_length=5000)
    try:
        config = create_domain(
            name=body.name,
            description=body.description,
            language=body.language,
            system_prompt=body.system_prompt,
            categories=body.categories,
        )
        # Register the domain's index in the index registry
        register_index(
            name=f"domain_{config.slug}",
            collection_name=config.collection_name,
            description=f"Domain: {config.name}",
        )
        return {"status": "created", "domain": dataclasses.asdict(config)}
    except ValueError as e:
        return {"error": str(e)}


@app.get("/api/domains")
def list_domains_endpoint(api_key: str = Depends(require_api_key)):
    """List all registered domains."""
    import dataclasses

    domains = list_domains()
    result = []
    for d in domains:
        info = dataclasses.asdict(d)
        # Add live chunk count from ChromaDB
        try:
            col = get_index(f"domain_{d.slug}")
            info["chunk_count"] = col.count()
        except Exception:
            info["chunk_count"] = d.chunk_count
        result.append(info)
    return {"domains": result}


@app.get("/api/domains/{slug}")
def get_domain_endpoint(slug: str, api_key: str = Depends(require_api_key)):
    """Get domain details including chunk count."""
    import dataclasses

    try:
        config = get_domain(slug)
        info = dataclasses.asdict(config)
        try:
            col = get_index(f"domain_{slug}")
            info["chunk_count"] = col.count()
        except Exception:
            pass
        return info
    except KeyError:
        return {"error": f"Domain '{slug}' not found"}


@app.put("/api/domains/{slug}")
def update_domain_endpoint(
    slug: str, body: DomainUpdateRequest, api_key: str = Depends(require_api_key)
):
    """Update domain configuration."""
    import dataclasses

    try:
        kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
        config = update_domain(slug, **kwargs)
        return {"status": "updated", "domain": dataclasses.asdict(config)}
    except KeyError:
        return {"error": f"Domain '{slug}' not found"}


@app.delete("/api/domains/{slug}")
def delete_domain_endpoint(slug: str, api_key: str = Depends(require_api_key)):
    """Delete a domain (config only — ChromaDB data preserved)."""
    try:
        delete_domain(slug)
        return {"status": "deleted", "slug": slug}
    except KeyError:
        return {"error": f"Domain '{slug}' not found"}


@app.post("/api/domains/{slug}/ingest")
def ingest_domain(slug: str, body: DomainIngestRequest, api_key: str = Depends(require_api_key)):
    """Ingest files into a domain's isolated vector space."""
    from pathlib import Path

    rate_limiter.check(api_key)
    slug = validate_slug(slug)
    body.directory = validate_directory_path(body.directory)

    try:
        config = get_domain(slug)
    except KeyError:
        return {"error": f"Domain '{slug}' not found"}

    data_dir = Path(body.directory)
    if not data_dir.exists():
        return {"error": f"Directory not found: {body.directory}"}

    # Register index if not already
    index_name = f"domain_{slug}"
    register_index(
        name=index_name,
        collection_name=config.collection_name,
        description=f"Domain: {config.name}",
    )
    target = reset_index(index_name) if body.force else get_index(index_name)

    # Find files
    files = list(iter_files(data_dir))
    if not files:
        return {"error": f"No files found in {data_dir}"}

    # Parallel read + chunk
    file_chunks = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(read_and_chunk, files):
            file_chunks.append(result)

    # Ensure embed model loaded
    get_embed_model()

    # Embed + add to domain's collection
    total_added = 0
    total_skipped = 0
    for path, chunks in file_chunks:
        if not chunks:
            continue
        added, skipped = add_chunks(target, path, chunks, data_dir)
        total_added += added
        total_skipped += skipped

    # Update cached chunk count
    update_domain(slug, chunk_count=target.count())

    return {
        "domain": slug,
        "files": len(files),
        "added": total_added,
        "skipped": total_skipped,
        "total": target.count(),
    }


@app.post("/api/domains/{slug}/query")
def query_domain(slug: str, body: QueryRequest, api_key: str = Depends(require_api_key)):
    """Query a domain's knowledge base. Same SSE stream as /api/query."""
    try:
        config = get_domain(slug)
    except KeyError:
        return {"error": f"Domain '{slug}' not found"}

    # Register index
    index_name = f"domain_{slug}"
    register_index(
        name=index_name,
        collection_name=config.collection_name,
        description=f"Domain: {config.name}",
    )

    # Use domain-specific system prompt
    domain_prompt = get_domain_prompt(slug)

    ctx, router, retriever, generator, evaluator = prepare_agent_context(
        query=body.query,
        category=body.category,
        top_k=body.top_k,
        use_react=body.use_react,
    )

    # Override route to use domain index
    router.execute(ctx)
    if ctx.route:
        ctx.route.indexes = [index_name]
    retriever.execute(ctx)

    # Pre-flight retry
    if ctx.retrieval_quality in ("weak", "failed") and ctx.attempt < ctx.max_attempts:
        evaluator.execute(ctx)
        if ctx.should_retry:
            ctx.attempt += 1
            ctx.results = []
            ctx.context_text = ""
            ctx.should_retry = False
            router.execute(ctx)
            if ctx.route:
                ctx.route.indexes = [index_name]
            retriever.execute(ctx)

    sources = [
        {"source": r["source"], "category": r["category"], "score": r["score"]} for r in ctx.results
    ]

    def stream():
        mode = ctx.route.mode if ctx.route else "answer"
        reason = ctx.route.reason if ctx.route else ""
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'route': {'mode': mode, 'reason': reason}, 'domain': slug})}\n\n"

        # Override system prompt for this domain
        import time

        from suyven_rag.rag.llm import stream_chat

        t0 = time.time()
        context = ctx.context_text
        tokens = []
        try:
            for token in stream_chat(body.query, context, system_prompt=domain_prompt):
                tokens.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            tokens.append(f"[LLM error: {e}]")
            yield f"data: {json.dumps({'type': 'token', 'content': f'[LLM error: {e}]'})}\n\n"

        ctx.response_tokens = tokens
        ctx.full_response = "".join(tokens)
        ctx.t_llm = time.time() - t0

        yield f"data: {json.dumps({'type': 'done', 'agent_trace': ctx.agent_trace, 'domain': slug})}\n\n"

        evaluator.execute(ctx)

    return StreamingResponse(stream(), media_type="text/event-stream")


class DomainFinetuneRequest(BaseModel):
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 32
    lora_rank: int = 8
    target_pairs: int = 2000
    min_pairs: int = 200
    min_reranker_score: float = 0.2


@app.post("/api/domains/{slug}/finetune")
def finetune_domain(
    slug: str, body: DomainFinetuneRequest, api_key: str = Depends(require_api_key)
):
    """Fine-tune the embedding model for a domain.

    Generates training pairs from the domain's corpus, trains LoRA adapters,
    merges into a domain-specific model, and registers it for automatic use.

    This is a long-running blocking call. Returns training summary when done.
    """
    import dataclasses

    from suyven_rag.finetune.domain_finetune import DomainFinetuneConfig, run_domain_finetune

    rate_limiter.check(api_key)
    slug = validate_slug(slug)

    try:
        get_domain(slug)
    except KeyError:
        return {"error": f"Domain '{slug}' not found"}

    config = DomainFinetuneConfig(
        slug=slug,
        epochs=body.epochs,
        learning_rate=body.learning_rate,
        batch_size=body.batch_size,
        lora_rank=body.lora_rank,
        target_pairs=body.target_pairs,
        min_pairs=body.min_pairs,
        min_reranker_score=body.min_reranker_score,
    )

    result = run_domain_finetune(slug, config)
    return dataclasses.asdict(result)


# ---------------------------------------------------------------------------
# Static frontend (Docker production mode)
# ---------------------------------------------------------------------------

_static_dir = _Path(__file__).parent / "static"
if _static_dir.is_dir() and (_static_dir / "index.html").exists():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # Mount /assets for JS/CSS bundles
    _assets_dir = _static_dir / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="static-assets")

    # SPA fallback: serve index.html for non-API, non-asset paths
    @app.get("/")
    def serve_index():
        return FileResponse(_static_dir / "index.html")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # Never intercept API routes
        if full_path.startswith("api/"):
            return {"error": "not found"}
        # Serve static file if it exists
        file_path = _static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # SPA fallback
        return FileResponse(_static_dir / "index.html")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
