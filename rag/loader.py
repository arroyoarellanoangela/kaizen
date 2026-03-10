"""Read files from the knowledge base directory."""

from pathlib import Path
from typing import Iterator

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# Folders to skip
SKIP_DIRS = {".git", ".claude", "__pycache__", ".venv", "node_modules"}


def iter_files(knowledge_dir: Path) -> Iterator[Path]:
    """Yield all supported files from the knowledge directory, sorted."""
    for f in sorted(knowledge_dir.rglob("*")):
        if not f.is_file():
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        yield f


def read_file(path: Path) -> str:
    """Return plain text content of a file."""
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            print(f"[warn] pymupdf not installed, skipping {path.name}")
            return ""
        except Exception as e:
            print(f"[warn] PDF read error {path.name}: {e}")
            return ""

    return ""
