"""Index registry — central access point for ChromaDB collections.

V2.1: single "default" index. route_to_index() always returns "default".
V2.2 will add real routing by category/document type.
"""

import logging
from dataclasses import dataclass

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME, EMBED_BATCH
from .model_registry import get_embed_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding function bridge (connects model_registry → ChromaDB)
# ---------------------------------------------------------------------------


class RegistryEmbedFn(chromadb.EmbeddingFunction):
    """ChromaDB embedding function backed by model_registry."""

    def __init__(self, model_name: str = "default_embed"):
        self._model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = get_embed_model(self._model_name)
        return model.encode(input, batch_size=EMBED_BATCH, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# Index descriptors
# ---------------------------------------------------------------------------


@dataclass
class IndexInfo:
    name: str
    collection_name: str
    embed_model: str       # key in model_registry
    description: str = ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_collections: dict[str, chromadb.Collection] = {}

_registry: dict[str, IndexInfo] = {
    "default": IndexInfo(
        name="default",
        collection_name=COLLECTION_NAME,
        embed_model="default_embed",
        description="Main engineer knowledge base",
    ),
}


def get_index(name: str = "default") -> chromadb.Collection:
    """Get (or create) a ChromaDB collection by registry name."""
    if name in _collections:
        return _collections[name]

    info = _registry.get(name)
    if info is None:
        raise KeyError(f"Index '{name}' not found in registry.")

    logger.info("Opening index: %s (collection=%s, embed=%s)",
                name, info.collection_name, info.embed_model)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name=info.collection_name,
        embedding_function=RegistryEmbedFn(info.embed_model),
        metadata={"hnsw:space": "cosine"},
    )
    _collections[name] = col
    return col


def reset_index(name: str = "default") -> chromadb.Collection:
    """Delete and recreate a ChromaDB collection by registry name."""
    info = _registry.get(name)
    if info is None:
        raise KeyError(f"Index '{name}' not found in registry.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(info.collection_name)
    except Exception:
        pass

    _collections.pop(name, None)
    return get_index(name)


def list_indexes() -> list[str]:
    """Return names of all registered indexes."""
    return list(_registry.keys())


def route_to_index(query: str, hint: str | None = None) -> str:
    """Decide which index to query.

    V2.1: always returns "default".
    V2.2 will add routing by category, extension, or semantic signals.
    """
    return "default"
