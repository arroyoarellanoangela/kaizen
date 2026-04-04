"""Model registry — central access point for embedding and reranker models.

V2.1: wraps existing singletons behind a named interface.
Future versions can register multiple models per type.
"""

import logging
from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import EMBED_MODEL, RERANKER_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model descriptors
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    name: str
    model_id: str
    model_type: str  # "embed" | "reranker"
    precision: str = "fp16"  # "fp32" | "fp16"
    device: str = "auto"  # "auto" | "cuda" | "cpu"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_embed_models: dict[str, SentenceTransformer] = {}
_reranker_models: dict[str, CrossEncoder] = {}

_registry: dict[str, ModelInfo] = {
    "default_embed": ModelInfo(
        name="default_embed",
        model_id=EMBED_MODEL,
        model_type="embed",
    ),
    "default_reranker": ModelInfo(
        name="default_reranker",
        model_id=RERANKER_MODEL,
        model_type="reranker",
    ),
}


def get_embed_model(name: str = "default_embed") -> SentenceTransformer:
    """Load (or return cached) embedding model by registry name."""
    if name in _embed_models:
        return _embed_models[name]

    info = _registry.get(name)
    if info is None or info.model_type != "embed":
        raise KeyError(f"Embedding model '{name}' not found in registry.")

    logger.info("Loading embed model: %s (%s)", name, info.model_id)
    model = SentenceTransformer(info.model_id)
    if info.precision == "fp16" and torch.cuda.is_available():
        model = model.to("cuda").half()
    _embed_models[name] = model
    return model


def get_reranker(name: str = "default_reranker") -> CrossEncoder:
    """Load (or return cached) reranker model by registry name."""
    if name in _reranker_models:
        return _reranker_models[name]

    info = _registry.get(name)
    if info is None or info.model_type != "reranker":
        raise KeyError(f"Reranker model '{name}' not found in registry.")

    logger.info("Loading reranker model: %s (%s)", name, info.model_id)
    model = CrossEncoder(info.model_id)
    if info.precision == "fp16" and torch.cuda.is_available():
        model.model = model.model.to("cuda").half()
    _reranker_models[name] = model
    return model


def register_embed_model(name: str, model_id: str, precision: str = "fp16") -> None:
    """Register (or update) a named embedding model in the registry.

    Used by domain fine-tuning to add domain-specific models.
    The model is lazy-loaded on first access via get_embed_model(name).
    """
    _registry[name] = ModelInfo(
        name=name,
        model_id=model_id,
        model_type="embed",
        precision=precision,
    )
    # Clear cached instance so it reloads from new path
    _embed_models.pop(name, None)
    logger.info("Registered embed model: %s -> %s", name, model_id)


def has_embed_model(name: str) -> bool:
    """Check if a named embedding model is registered."""
    info = _registry.get(name)
    return info is not None and info.model_type == "embed"


def list_models() -> dict[str, dict]:
    """Return registry contents as plain dicts for API/logging."""
    result = {}
    for k, v in _registry.items():
        loaded = (k in _embed_models) if v.model_type == "embed" else (k in _reranker_models)
        result[k] = {
            "model_id": v.model_id,
            "type": v.model_type,
            "precision": v.precision,
            "loaded": loaded,
        }
    return result
