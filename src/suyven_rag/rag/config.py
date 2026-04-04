import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    """Read a config value from Docker secrets first, then env var.

    Docker secrets are mounted at /run/secrets/<name> (Linux convention).
    Falls back to os.getenv(name, default) when not in Docker or on Windows.
    """
    secret_path = Path("/run/secrets") / name
    try:
        if secret_path.is_file():
            return secret_path.read_text().strip()
    except (OSError, PermissionError):
        pass
    return os.getenv(name, default)


# Paths
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", r"C:\Users\bingu\engineer-knowledge"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(Path(__file__).parents[1] / "data" / "chroma")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "engineer_knowledge")

# Ollama / embeddings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

_raw_embed_model = os.getenv("EMBED_MODEL", "").strip()
EMBED_MODEL = _raw_embed_model if _raw_embed_model else "BAAI/bge-m3"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OVERFETCH_FACTOR = int(os.getenv("OVERFETCH_FACTOR", "6"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))

# Embedding batches
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "256"))
ADD_BATCH_SIZE = int(os.getenv("ADD_BATCH_SIZE", "500"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
LLM_API_URL = os.getenv("LLM_API_URL", "")  # e.g. https://api.deepseek.com/v1
LLM_API_KEY = _secret("LLM_API_KEY")

# Fallback LLM — used when RAG retrieval fails (answers from parametric knowledge)
FALLBACK_PROVIDER = os.getenv("FALLBACK_PROVIDER", "")  # "openai" (Gemini-compatible)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")  # e.g. "gemini-2.5-flash"
FALLBACK_API_URL = os.getenv(
    "FALLBACK_API_URL", ""
)  # e.g. "https://generativelanguage.googleapis.com/v1beta/openai"
FALLBACK_API_KEY = _secret("FALLBACK_API_KEY")

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

FALLBACK_PROMPT = """You are a helpful technical assistant. The user's question could not be answered from the knowledge base.
Answer from your own knowledge. Be honest about uncertainty. Use markdown. Be concise.
Cite no sources (you are answering from memory, not documents).
Answer in the same language as the question."""

# Concurrency
WORKERS = int(os.getenv("WORKERS", "8"))
