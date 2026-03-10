"""Kaizen v1 — FastAPI backend for the React frontend."""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import requests as req
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.chunker import chunk_text
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR, OLLAMA_URL
from rag.loader import iter_files, read_file
from rag.retriever import format_context, search
from rag.store import (
    add_chunks,
    ensure_ollama,
    get_collection,
    get_embed_model,
    reset_collection,
)

# ---------------------------------------------------------------------------
# GPU monitoring (pynvml)
# ---------------------------------------------------------------------------

_PYNVML_OK = False
try:
    import pynvml

    pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    pass


def _gpu_metrics() -> dict[str, Any] | None:
    if not _PYNVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "name": name,
            "temp_c": temp,
            "gpu_util": util.gpu,
            "vram_used_gb": round(mem.used / (1024**3), 2),
            "vram_total_gb": round(mem.total / (1024**3), 2),
            "vram_pct": round(mem.used / mem.total * 100, 1),
        }
    except Exception:
        return None


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
# Models
# ---------------------------------------------------------------------------

LLM_MODEL = "qwen3:8b"
SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's question using ONLY the provided context from their knowledge base.
- Synthesize information from all relevant sources into a clear, complete answer.
- Use markdown formatting for readability.
- If the context doesn't contain enough information, say so honestly.
- Cite sources using [source_name] when referencing specific documents.
- Answer in the same language as the question."""


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
        "gpu": _gpu_metrics(),
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


def _read_and_chunk(path):
    text = read_file(path)
    if not text.strip():
        return path, []
    return path, chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)


@app.post("/api/ingest")
def ingest(body: IngestRequest):
    """Ingest knowledge base. Blocking call — returns when done."""
    files = list(iter_files(KNOWLEDGE_DIR))
    if not files:
        return {"error": f"No files found in {KNOWLEDGE_DIR}"}

    target = reset_collection() if body.force else get_collection()

    # Phase 1: parallel read + chunk
    file_chunks = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(_read_and_chunk, files):
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
