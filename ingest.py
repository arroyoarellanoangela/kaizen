#!/usr/bin/env python3
"""
Ingest engineer-knowledge into ChromaDB.

Usage:
    python ingest.py           # incremental (skips already-indexed chunks)
    python ingest.py --force   # wipe collection and re-index everything
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from rag.config import KNOWLEDGE_DIR, WORKERS
from rag.index_registry import get_index, reset_index
from rag.loader import iter_files
from rag.model_registry import get_embed_model
from rag.pipeline import read_and_chunk
from rag.store import add_chunks, ensure_ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main(force: bool = False) -> None:
    ensure_ollama()
    logger.info("Knowledge dir: %s", KNOWLEDGE_DIR)

    if not KNOWLEDGE_DIR.exists():
        logger.error("Not found: %s", KNOWLEDGE_DIR)
        sys.exit(1)

    col = reset_index() if force else get_index()
    if force:
        logger.info("Collection cleared — full re-index.")

    # ── Phase 1: Parallel read + chunk ────────────────────────────────
    files = list(iter_files(KNOWLEDGE_DIR))
    logger.info("[Phase 1] Reading & chunking %d files (parallel)...", len(files))
    file_chunks = []
    total_chunk_count = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for path, chunks in tqdm(
            pool.map(read_and_chunk, files),
            total=len(files),
            desc="Reading",
            unit="file",
        ):
            file_chunks.append((path, chunks))
            total_chunk_count += len(chunks)

    logger.info("   -> %d chunks from %d files", total_chunk_count, len(files))

    # ── Phase 2: Pre-load embedding model ─────────────────────────────
    logger.info("[Phase 2] Loading embedding model to GPU...")
    get_embed_model()
    logger.info("   -> Model ready")

    # ── Phase 3: Embed + index ────────────────────────────────────────
    logger.info("[Phase 3] Embedding & indexing...")
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

    logger.info("─" * 60)
    logger.info("Files processed : %d", total_files)
    logger.info("Chunks added    : %d", total_added)
    logger.info("Chunks skipped  : %d", total_skipped)
    logger.info("Total in DB     : %d", col.count())


if __name__ == "__main__":
    main(force="--force" in sys.argv)
