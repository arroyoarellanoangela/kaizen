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
