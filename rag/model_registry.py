"""Model registry — central access point for embedding and reranker models.

V2.1: wraps existing singletons behind a named interface.
Future versions can register multiple models per type.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

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
    model_type: str          # "embed" | "reranker"
    precision: str = "fp16"  # "fp32" | "fp16"
    device: str = "auto"     # "auto" | "cuda" | "cpu"


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


def list_models() -> dict[str, dict]:
    """Return registry contents as plain dicts for API/logging."""
    return {k: {"model_id": v.model_id, "type": v.model_type, "precision": v.precision}
            for k, v in _registry.items()}
