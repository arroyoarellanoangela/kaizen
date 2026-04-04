"""Knowledge ingestion from GitHub repos into ChromaDB.

Fetches README, docs, and key source files from public GitHub repos,
chunks them, and adds to ChromaDB. Also generates training pairs
for fine-tuning the embedding model.

Usage:
    python -m rag.vector_store --repos https://github.com/hiyouga/LLaMA-Factory https://github.com/unslothai/unsloth
    python -m rag.vector_store --repos-file repos.txt
    python -m rag.vector_store --generate-pairs
"""

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .chunker import chunk_text
from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .index_registry import get_index
from .store import embed_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "github_knowledge"
PAIRS_OUTPUT = BASE_DIR / "data" / "finetune" / "pairs_github.jsonl"

# Files worth fetching from repos (ordered by priority)
INTERESTING_FILES = [
    "README.md",
    "README.rst",
    "docs/getting_started.md",
    "docs/quickstart.md",
    "docs/overview.md",
    "docs/README.md",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "docs/architecture.md",
    "docs/design.md",
]


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner/repo from GitHub URL."""
    url = url.rstrip("/").rstrip(".git")
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    raise ValueError(f"Invalid GitHub URL: {url}")


def fetch_github_file(owner: str, repo: str, path: str) -> str | None:
    """Fetch a single file from GitHub via raw content URL."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.text
        url2 = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{path}"
        resp2 = requests.get(url2, timeout=15)
        if resp2.status_code == 200:
            return resp2.text
    except Exception as e:
        logger.debug("Failed to fetch %s: %s", path, e)
    return None


def fetch_repo_tree(owner: str, repo: str) -> list[dict]:
    """Fetch repo file tree via GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1"
            resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("tree", [])
    except Exception as e:
        logger.warning("Failed to fetch tree for %s/%s: %s", owner, repo, e)
    return []


def find_key_files(tree: list[dict], max_files: int = 15) -> list[str]:
    """Find the most interesting source files in a repo tree."""
    files = []
    tree_paths = {item["path"] for item in tree if item["type"] == "blob"}

    for pattern in INTERESTING_FILES:
        if pattern in tree_paths:
            files.append(pattern)

    py_files = sorted(
        [
            item["path"]
            for item in tree
            if item["type"] == "blob"
            and item["path"].endswith(".py")
            and item.get("size", 0) < 50000
            and not item["path"].startswith("test")
            and "__pycache__" not in item["path"]
        ],
        key=lambda p: (
            0 if any(kw in p.lower() for kw in ["train", "lora", "config", "model", "eval"]) else 1,
            len(p),
        ),
    )

    for f in py_files:
        if f not in files:
            files.append(f)
        if len(files) >= max_files:
            break

    return files[:max_files]


def categorize_file(path: str) -> str:
    """Categorize a file by its content type."""
    lower = path.lower()
    if any(kw in lower for kw in ["readme", "doc", "guide", "tutorial"]):
        return "documentation"
    if any(kw in lower for kw in ["train", "fine", "lora", "rlhf", "dpo"]):
        return "training"
    if any(kw in lower for kw in ["eval", "bench", "metric", "test"]):
        return "evaluation"
    if any(kw in lower for kw in ["config", "setting", "param"]):
        return "configuration"
    if any(kw in lower for kw in ["model", "arch", "layer", "attention"]):
        return "architecture"
    if any(kw in lower for kw in ["data", "dataset", "loader", "collat"]):
        return "data"
    return "source"


def clean_for_embedding(text: str) -> str:
    """Clean text for better embedding quality."""
    lines = text.split("\n")
    cleaned = []
    in_code_block = False
    code_block_lines = 0

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                if code_block_lines <= 30:
                    cleaned.append(line)
                else:
                    cleaned.append("```")
                    cleaned.append(f"[... {code_block_lines} lines of code omitted ...]")
                in_code_block = False
                code_block_lines = 0
            else:
                in_code_block = True
                code_block_lines = 0
                cleaned.append(line)
        elif in_code_block:
            code_block_lines += 1
            if code_block_lines <= 30:
                cleaned.append(line)
        else:
            cleaned.append(line)

    result = "\n".join(cleaned)
    result = re.sub(r"!\[.*?\]\(.*?\)", "", result)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result.strip()


def fetch_repo_knowledge(url: str, delay: float = 1.0) -> list[dict]:
    """Fetch all interesting files from a GitHub repo."""
    owner, repo = parse_github_url(url)
    logger.info("Fetching %s/%s...", owner, repo)

    tree = fetch_repo_tree(owner, repo)
    key_files = find_key_files(tree) if tree else INTERESTING_FILES[:5]

    docs = []
    for path in key_files:
        content = fetch_github_file(owner, repo, path)
        if content and len(content.strip()) > 100:
            docs.append(
                {
                    "path": path,
                    "content": content,
                    "repo": f"{owner}/{repo}",
                    "url": url,
                    "category": categorize_file(path),
                }
            )
            logger.info("  Fetched: %s (%d chars)", path, len(content))
        time.sleep(delay)

    logger.info("Fetched %d files from %s/%s", len(docs), owner, repo)
    return docs


def save_knowledge_local(docs: list[dict]) -> Path:
    """Save fetched docs to local filesystem for offline access."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    for doc in docs:
        safe_repo = doc["repo"].replace("/", "_")
        safe_path = doc["path"].replace("/", "_")
        out = KNOWLEDGE_DIR / f"{safe_repo}__{safe_path}"
        out.write_text(doc["content"], encoding="utf-8")

    manifest = KNOWLEDGE_DIR / "manifest.json"
    existing = []
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))

    seen = {(d["repo"], d["path"]) for d in existing}
    for doc in docs:
        key = (doc["repo"], doc["path"])
        if key not in seen:
            existing.append(
                {
                    "repo": doc["repo"],
                    "path": doc["path"],
                    "url": doc["url"],
                    "category": doc["category"],
                    "chars": len(doc["content"]),
                }
            )
            seen.add(key)

    manifest.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d docs to %s", len(docs), KNOWLEDGE_DIR)
    return KNOWLEDGE_DIR


def ingest_to_chromadb(docs: list[dict]) -> tuple[int, int]:
    """Chunk and add fetched docs to ChromaDB."""
    collection = get_index()
    total_added = 0
    total_skipped = 0

    for doc in docs:
        content = clean_for_embedding(doc["content"])
        if len(content.strip()) < 50:
            continue

        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            cid = f"gh_{hashlib.md5((doc['repo'] + doc['path'] + str(i) + chunk[:200]).encode()).hexdigest()[:12]}"
            ids.append(cid)
            documents.append(chunk)
            metadatas.append(
                {
                    "category": f"github-{doc['category']}",
                    "subcategory": doc["repo"],
                    "source": f"github/{doc['repo']}/{doc['path']}",
                    "file_type": doc["path"].split(".")[-1] if "." in doc["path"] else "md",
                    "chunk_index": str(i),
                }
            )

        try:
            existing = set(collection.get(ids=ids)["ids"])
        except Exception:
            existing = set()

        new_ids = [i for i in ids if i not in existing]
        if not new_ids:
            total_skipped += len(ids)
            continue

        new_docs = [d for i, d in zip(ids, documents, strict=False) if i not in existing]
        new_metas = [m for i, m in zip(ids, metadatas, strict=False) if i not in existing]

        embeddings = embed_batch(new_docs)
        batch_size = 100
        for j in range(0, len(new_ids), batch_size):
            end = j + batch_size
            collection.add(
                ids=new_ids[j:end],
                documents=new_docs[j:end],
                metadatas=new_metas[j:end],
                embeddings=embeddings[j:end],
            )

        total_added += len(new_ids)
        total_skipped += len(existing)
        logger.info(
            "  %s/%s: +%d chunks (skipped %d)",
            doc["repo"],
            doc["path"],
            len(new_ids),
            len(existing),
        )

    logger.info("Ingestion complete: %d added, %d skipped", total_added, total_skipped)
    return total_added, total_skipped


def _generate_questions_gemini(chunk: str, api_key: str, n: int = 2) -> list[str]:
    """Generate questions via Gemini Flash (free tier)."""
    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": "gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Generate {n} specific technical questions this text answers. "
                        f"Return ONLY the questions, one per line, no numbering.\n\n{chunk[:1500]}",
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            questions = [
                q.strip() for q in content.strip().split("\n") if q.strip() and len(q.strip()) > 10
            ]
            return questions[:n]
    except Exception as e:
        logger.debug("Gemini question gen failed: %s", e)
    return []


def generate_training_pairs(docs: list[dict], output: Path = PAIRS_OUTPUT) -> int:
    """Generate (query, passage) training pairs from GitHub knowledge."""
    import os

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    pairs = []

    for doc in docs:
        content = clean_for_embedding(doc["content"])
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk in chunks:
            if len(chunk.strip()) < 100:
                continue

            if gemini_key:
                questions = _generate_questions_gemini(chunk, gemini_key)
                for q in questions:
                    pairs.append(
                        {
                            "query": q,
                            "positive": chunk,
                            "source": f"github/{doc['repo']}/{doc['path']}",
                            "category": doc["category"],
                        }
                    )
            else:
                # Heuristic fallback: first sentence as query
                sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
                for s in sentences[:1]:
                    s = s.strip()
                    if 15 <= len(s) <= 150 and sum(c.isalpha() for c in s) / max(len(s), 1) > 0.5:
                        pairs.append(
                            {
                                "query": s,
                                "positive": chunk,
                                "source": f"github/{doc['repo']}/{doc['path']}",
                                "category": doc["category"],
                            }
                        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info("Generated %d training pairs -> %s", len(pairs), output)
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Ingest GitHub repos into ChromaDB")
    parser.add_argument("--repos", nargs="+", help="GitHub repo URLs")
    parser.add_argument("--repos-file", type=Path, help="File with repo URLs (one per line)")
    parser.add_argument(
        "--generate-pairs", action="store_true", help="Also generate training pairs"
    )
    parser.add_argument("--skip-chromadb", action="store_true", help="Only save locally")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    args = parser.parse_args()

    urls = args.repos or []
    if args.repos_file and args.repos_file.exists():
        urls += [
            line.strip()
            for line in args.repos_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if not urls:
        print("No repos. Use --repos URL1 URL2 or --repos-file file.txt")
        return

    all_docs = []
    for url in urls:
        docs = fetch_repo_knowledge(url, delay=args.delay)
        all_docs.extend(docs)

    if not all_docs:
        print("No documents fetched.")
        return

    save_knowledge_local(all_docs)

    if not args.skip_chromadb:
        added, skipped = ingest_to_chromadb(all_docs)
        print(f"\nChromaDB: +{added} chunks ({skipped} already existed)")

    if args.generate_pairs:
        n_pairs = generate_training_pairs(all_docs)
        print(f"Training pairs: {n_pairs} generated")

    print(f"\nDone! {len(all_docs)} files from {len(urls)} repos")


if __name__ == "__main__":
    main()
