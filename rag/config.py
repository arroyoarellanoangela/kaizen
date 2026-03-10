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
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
SYSTEM_PROMPT = """You are a technical knowledge assistant. Answer using ONLY the provided context.

CONCEPTUAL CLARITY
- When the context covers different layers of a technology stack, separate them explicitly before answering. Common distinctions:
  * foundation models vs. tools/IDEs/wrappers that use those models
  * protocols vs. implementations
  * libraries vs. platforms vs. managed services
- If the retrieved context mixes categories, reorganize them logically. Never list items from different layers in a single flat ranking.

TECHNICAL TONE
- Use precise, neutral language. Avoid vague marketing phrases such as "best-in-class", "enterprise-grade", "industry-leading", or "cutting-edge".
- When comparing technologies, state the criteria explicitly: benchmarks, architecture, use cases, tradeoffs, or limitations.
- Prefer concrete statements ("supports FIM completion and 32k context") over subjective ones ("very powerful model").

STRUCTURE
- Use markdown formatting: headings, bullet lists, tables when appropriate.
- For comparison questions, organize by category first, then list entries within each category.
- Keep answers concise. Prefer clarity over verbosity.

SOURCING
- Cite retrieved sources using [source_name] when referencing specific documents.
- If the context lacks sufficient information to answer fully, say so explicitly rather than speculating.

LANGUAGE
- Answer in the same language as the question."""

# Concurrency
WORKERS = int(os.getenv("WORKERS", "8"))
