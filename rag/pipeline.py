"""Shared ingestion pipeline helpers."""

from pathlib import Path

from .chunker import chunk_text
from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .loader import read_file


def read_and_chunk(path: Path) -> tuple[Path, list[str]]:
    """Read a file and return (path, chunks). Safe for ThreadPoolExecutor."""
    text = read_file(path)
    if not text.strip():
        return path, []
    return path, chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
