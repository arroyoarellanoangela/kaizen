"""Index registry — central access point for ChromaDB collections.

Supports both the static "default" index (main Suyven knowledge base)
and dynamic domain-specific indexes created via domain_registry.
"""

import contextlib
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
    embed_model: str  # key in model_registry
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


def register_index(
    name: str, collection_name: str, embed_model: str = "default_embed", description: str = ""
) -> None:
    """Register a new index (used by domain_registry to add domain indexes)."""
    _registry[name] = IndexInfo(
        name=name,
        collection_name=collection_name,
        embed_model=embed_model,
        description=description,
    )
    logger.info("Registered index: %s (collection=%s)", name, collection_name)


def get_index(name: str = "default") -> chromadb.Collection:
    """Get (or create) a ChromaDB collection by registry name.

    For domain indexes: if not in _registry, tries to auto-register
    from domain_registry (lazy loading).
    """
    if name in _collections:
        return _collections[name]

    # If not in static registry, check if it's a domain
    if name not in _registry and name.startswith("domain_"):
        _try_register_domain_index(name)

    info = _registry.get(name)
    if info is None:
        raise KeyError(f"Index '{name}' not found in registry.")

    logger.info(
        "Opening index: %s (collection=%s, embed=%s)", name, info.collection_name, info.embed_model
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(
        name=info.collection_name,
        embedding_function=RegistryEmbedFn(info.embed_model),
        metadata={"hnsw:space": "cosine"},
    )
    _collections[name] = col
    return col


def _try_register_domain_index(name: str) -> None:
    """Try to auto-register a domain index from domain_registry.

    If a fine-tuned embed model exists for this domain, uses it.
    Otherwise falls back to the default embed model.
    """
    try:
        from .domain_registry import get_domain
        from .model_registry import has_embed_model

        # name format: "domain_<slug>" -> slug
        slug = name[len("domain_") :]
        domain = get_domain(slug)

        # Use domain-specific embed model if fine-tuned, else default
        domain_embed = f"domain_{slug}_embed"
        embed_model = domain_embed if has_embed_model(domain_embed) else "default_embed"

        register_index(
            name=name,
            collection_name=domain.collection_name,
            embed_model=embed_model,
            description=f"Domain: {domain.name}",
        )
    except (KeyError, ImportError):
        pass


def reset_index(name: str = "default") -> chromadb.Collection:
    """Delete and recreate a ChromaDB collection by registry name."""
    info = _registry.get(name)
    if info is None:
        raise KeyError(f"Index '{name}' not found in registry.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    with contextlib.suppress(Exception):
        client.delete_collection(info.collection_name)

    _collections.pop(name, None)
    return get_index(name)


def list_indexes() -> list[str]:
    """Return names of all registered indexes."""
    return list(_registry.keys())


def route_to_index(query: str, hint: str | None = None) -> str:
    """Decide which index to query.

    If hint matches a domain slug, routes to that domain's index.
    Otherwise returns "default".
    """
    if hint and f"domain_{hint}" in _registry:
        return f"domain_{hint}"
    # Also check if hint itself is a registered index name
    if hint and hint in _registry:
        return hint
    return "default"
