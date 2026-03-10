import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", r"C:\Users\bingu\engineer-knowledge"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(Path(__file__).parents[1] / "data" / "chroma")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "engineer_knowledge")

# Ollama / embeddings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

_raw_embed_model = os.getenv("EMBED_MODEL", "").strip()
if not _raw_embed_model or _raw_embed_model in {
    "sentence-transformers/nomic-embed-text",
    "nomic-embed-text",
    "nomic-ai/nomic-embed-text-v1.5",
}:
    # Default: fast model with standard PyTorch ops (works on all GPUs)
    EMBED_MODEL = "all-MiniLM-L6-v2"
else:
    EMBED_MODEL = _raw_embed_model

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OVERFETCH_FACTOR = int(os.getenv("OVERFETCH_FACTOR", "4"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

# Embedding batches
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "256"))
ADD_BATCH_SIZE = int(os.getenv("ADD_BATCH_SIZE", "500"))

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
SYSTEM_PROMPT = """You are a technical knowledge assistant. Answer ONLY from the provided context.

STRICT RULES:
1. NEVER mix different categories in one list. Always separate:
   - Foundation models (GPT, Claude, Mistral, DeepSeek, etc.) go under "## Models"
   - Tools, IDEs, agents (Cursor, Windsurf, Claude Code, etc.) go under "## Tools"
   - Libraries, frameworks go under "## Libraries"
2. NEVER use marketing words: "best-in-class", "enterprise-grade", "industry-leading", "cutting-edge", "game-changing". Use factual descriptions instead.
3. When comparing, state WHY: benchmarks, architecture, parameter count, use case.
4. Use markdown. Be concise.
5. Cite sources as [source_name].
6. If context is insufficient, say so. Do not invent information.
7. Answer in the same language as the question."""

# Concurrency
WORKERS = int(os.getenv("WORKERS", "8"))
