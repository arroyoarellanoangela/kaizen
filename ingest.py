#!/usr/bin/env python3
"""
Ingest engineer-knowledge into ChromaDB.

Usage:
    python ingest.py           # incremental (skips already-indexed chunks)
    python ingest.py --force   # wipe collection and re-index everything
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from rag.chunker import chunk_text
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR
from rag.loader import iter_files, read_file
from rag.store import add_chunks, ensure_ollama, get_collection, get_embed_model, reset_collection


def _read_and_chunk(path):
    """Read a file and return its chunks (runs in thread)."""
    text = read_file(path)
    if not text.strip():
        return path, []
    return path, chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)


def main(force: bool = False) -> None:
    ensure_ollama()
    print(f"Knowledge dir : {KNOWLEDGE_DIR}")

    if not KNOWLEDGE_DIR.exists():
        print(f"[error] Not found: {KNOWLEDGE_DIR}")
        sys.exit(1)

    col = reset_collection() if force else get_collection()
    if force:
        print("[info] Collection cleared — full re-index.")

    # ── Phase 1: Parallel read + chunk ────────────────────────────────
    files = list(iter_files(KNOWLEDGE_DIR))
    print(f"\n[Phase 1] Reading & chunking {len(files)} files (parallel)...")
    file_chunks = []
    total_chunk_count = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for path, chunks in tqdm(
            pool.map(_read_and_chunk, files),
            total=len(files),
            desc="Reading",
            unit="file",
        ):
            file_chunks.append((path, chunks))
            total_chunk_count += len(chunks)

    print(f"   -> {total_chunk_count} chunks from {len(files)} files\n")

    # ── Phase 2: Pre-load embedding model ─────────────────────────────
    print("[Phase 2] Loading embedding model to GPU...")
    get_embed_model()
    print("   -> Model ready\n")

    # ── Phase 3: Embed + index ────────────────────────────────────────
    print(f"[Phase 3] Embedding & indexing...")
    total_added = total_skipped = total_files = 0

    pbar = tqdm(file_chunks, desc="Indexing", unit="file")
    for path, chunks in pbar:
        if not chunks:
            continue

        added, skipped = add_chunks(col, path, chunks, KNOWLEDGE_DIR)
        total_files += 1
        total_added += added
        total_skipped += skipped

        pbar.set_postfix(added=total_added, skipped=total_skipped, refresh=True)

    print(f"\n{'─'*60}")
    print(f"Files processed : {total_files}")
    print(f"Chunks added    : {total_added}")
    print(f"Chunks skipped  : {total_skipped}")
    print(f"Total in DB     : {col.count()}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
