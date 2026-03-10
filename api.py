"""Kaizen v1 — FastAPI backend for the React frontend."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import requests as req
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.config import KNOWLEDGE_DIR, LLM_MODEL, OLLAMA_URL, SYSTEM_PROMPT, WORKERS
from rag.loader import iter_files
from rag.monitoring import gpu_metrics
from rag.pipeline import read_and_chunk
from rag.retriever import format_context, search
from rag.store import (
    add_chunks,
    ensure_ollama,
    get_collection,
    get_embed_model,
    reset_collection,
)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_ollama()
    get_collection()  # warm ChromaDB
    yield


app = FastAPI(title="Kaizen RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    category: str | None = None


class IngestRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/status")
def status():
    """System status: chunk count, GPU metrics, system info."""
    col = get_collection()
    return {
        "chunks": col.count(),
        "gpu": gpu_metrics(),
        "llm_model": LLM_MODEL,
        "ollama_url": OLLAMA_URL,
    }


@app.post("/api/query")
def query(body: QueryRequest):
    """
    Query the knowledge base. Returns SSE stream:
      - data: {"type":"sources","sources":[...]}   (first)
      - data: {"type":"token","content":"..."}      (streaming tokens)
      - data: {"type":"done"}                       (end)
    """
    results = search(body.query, n=body.top_k, category=body.category)

    if not results:
        def no_results():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'content': 'No results found in the knowledge base.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(no_results(), media_type="text/event-stream")

    context = format_context(results)

    # Build sources payload (sent first so UI can show them while streaming)
    # Cross-encoder returns logits; sigmoid converts to 0-1 probability
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    sources = [
        {
            "category": r["category"],
            "source": r["source"],
            "score": round(_sigmoid(r["score"]), 3),
            "text": r["text"][:300],
        }
        for r in results
    ]

    def stream():
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream LLM response
        try:
            resp = req.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {body.query}",
                        },
                    ],
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'token', 'content': f'[LLM error: {e}]'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/ingest")
def ingest(body: IngestRequest):
    """Ingest knowledge base. Blocking call — returns when done."""
    files = list(iter_files(KNOWLEDGE_DIR))
    if not files:
        return {"error": f"No files found in {KNOWLEDGE_DIR}"}

    target = reset_collection() if body.force else get_collection()

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
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
