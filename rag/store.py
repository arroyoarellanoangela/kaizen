"""ChromaDB vector store — embed with sentence-transformers (direct GPU)."""

import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from .config import ADD_BATCH_SIZE, CHROMA_DIR, COLLECTION_NAME, EMBED_BATCH, EMBED_MODEL, OLLAMA_URL


# ---------------------------------------------------------------------------
# Ollama lifecycle  (still used for LLM chat, not for embeddings)
# ---------------------------------------------------------------------------

def ensure_ollama(timeout: int = 30) -> None:
    """Start Ollama if it is not already running."""
    try:
        requests.get(OLLAMA_URL, timeout=2)
        return
    except Exception:
        pass

    logger.info("Ollama not running — starting ollama serve...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(OLLAMA_URL, timeout=2)
            logger.info("Ollama ready.")
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(f"Ollama did not start within {timeout}s. Is it installed?")


# ---------------------------------------------------------------------------
# Embedding — sentence-transformers (direct GPU, no HTTP overhead)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None

def get_embed_model() -> SentenceTransformer:
    """Lazy-load the embedding model.

    Applies FP16 + CUDA when available for ~38x speedup over CPU
    (see BENCHMARK_RESULTS.md).
    """
    global _model
    if _model is None:
        import torch
        _model = SentenceTransformer(EMBED_MODEL)
        if torch.cuda.is_available():
            _model = _model.to("cuda").half()  # FP16: +109% throughput, −48% VRAM
    return _model


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts on GPU (FP16) via sentence-transformers."""
    model = get_embed_model()
    embeddings = model.encode(texts, batch_size=EMBED_BATCH, show_progress_bar=False)
    return embeddings.tolist()


def embed(text: str) -> list[float]:
    """Embed a single text (convenience wrapper)."""
    return embed_batch([text])[0]


class STEmbedFn(chromadb.EmbeddingFunction):
    """ChromaDB embedding function using sentence-transformers."""
    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_batch(input)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=STEmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=STEmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _chunk_id(path: Path, idx: int, content: str) -> str:
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{path.stem}_{idx}_{h}"


def add_chunks(
    collection: chromadb.Collection,
    path: Path,
    chunks: list[str],
    knowledge_dir: Path,
) -> tuple[int, int]:
    """
    Add chunks to ChromaDB. Returns (added, skipped).
    Skips chunks already indexed (by deterministic ID).
    Uses batch operations for speed.
    """
    if not chunks:
        return 0, 0

    # Build metadata from relative path
    try:
        rel = path.relative_to(knowledge_dir)
        parts = rel.parts
        category = parts[0] if len(parts) > 1 else "root"
        subcategory = parts[1] if len(parts) > 2 else ""
    except ValueError:
        category = "unknown"
        subcategory = ""

    # ── Batch existence check ──────────────────────────────────────────
    all_ids = [_chunk_id(path, i, c) for i, c in enumerate(chunks)]
    try:
        existing_ids = set(collection.get(ids=all_ids)["ids"])
    except Exception:
        existing_ids = set()

    # Filter to only new chunks
    ids, docs, metas = [], [], []
    for idx, (cid, chunk) in enumerate(zip(all_ids, chunks)):
        if cid in existing_ids:
            continue
        ids.append(cid)
        docs.append(chunk)
        metas.append({
            "category": category,
            "subcategory": subcategory,
            "source": path.stem,
            "file_type": path.suffix.lstrip("."),
            "chunk_index": str(idx),
        })

    skipped = len(chunks) - len(ids)

    # ── Batch embed + add ─────────────────────────────────────────────
    if ids:
        embeddings = embed_batch(docs)
        for i in range(0, len(ids), ADD_BATCH_SIZE):
            end = i + ADD_BATCH_SIZE
            collection.add(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metas[i:end],
                embeddings=embeddings[i:end],
            )

    return len(ids), skipped
