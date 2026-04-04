"""Domain Registry — manages isolated knowledge domains.

Each domain gets:
  - Its own ChromaDB collection (isolated vector space)
  - Domain-specific system prompt
  - Metadata (name, description, created_at, chunk_count)
  - Independent eval logs

Suyven is the base engine. Domains are specializations:
  suyven.create_domain("oncologia") -> isolated vectorDB + config
  suyven.ingest("oncologia", files)  -> chunks embedded into domain's collection
  suyven.query("oncologia", "...")   -> retrieval scoped to domain

Persistence: domain configs stored in data/domains/<slug>/config.json
"""

import contextlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DOMAINS_DIR = BASE_DIR / "data" / "domains"


# ---------------------------------------------------------------------------
# Domain config
# ---------------------------------------------------------------------------


@dataclass
class DomainConfig:
    """Configuration for a knowledge domain."""

    slug: str  # URL-safe identifier (e.g., "oncologia")
    name: str  # Human name (e.g., "Oncologia")
    description: str = ""  # What this domain covers
    language: str = "auto"  # Default response language ("auto", "es", "en", etc.)
    system_prompt: str = ""  # Domain-specific system prompt (empty = use base)
    categories: list[str] = field(default_factory=list)  # Expected data categories
    created_at: str = ""  # ISO timestamp
    updated_at: str = ""  # ISO timestamp
    chunk_count: int = 0  # Cached count (updated on ingest)
    collection_name: str = ""  # ChromaDB collection name


# ---------------------------------------------------------------------------
# Slugification
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:50]


# ---------------------------------------------------------------------------
# Default system prompt template
# ---------------------------------------------------------------------------

_DOMAIN_PROMPT_TEMPLATE = """You are a specialized knowledge assistant for the domain: {name}.
{description}

Answer ONLY from the provided context. Rules:
1. Be precise and factual. Use domain-specific terminology.
2. Cite sources as [source_name].
3. If context is insufficient, say so explicitly. Do not invent information.
4. Use markdown for formatting. Be concise.
5. Answer in the same language as the question."""


def _build_system_prompt(config: DomainConfig) -> str:
    """Build the effective system prompt for a domain."""
    if config.system_prompt:
        return config.system_prompt
    desc = f"Description: {config.description}" if config.description else ""
    return _DOMAIN_PROMPT_TEMPLATE.format(name=config.name, description=desc)


# ---------------------------------------------------------------------------
# Registry operations
# ---------------------------------------------------------------------------

_domains: dict[str, DomainConfig] = {}


def _config_path(slug: str) -> Path:
    return DOMAINS_DIR / slug / "config.json"


def _save_config(config: DomainConfig) -> None:
    """Persist domain config to disk."""
    path = _config_path(config.slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_config(slug: str) -> DomainConfig | None:
    """Load domain config from disk."""
    path = _config_path(slug)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DomainConfig(**data)
    except Exception as e:
        logger.warning("Failed to load domain config %s: %s", slug, e)
        return None


def _load_all() -> None:
    """Load all domain configs from disk into memory."""
    if not DOMAINS_DIR.exists():
        return
    for d in DOMAINS_DIR.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            config = _load_config(d.name)
            if config:
                _domains[config.slug] = config


# Auto-load on import
_load_all()


def create_domain(
    name: str,
    description: str = "",
    language: str = "auto",
    system_prompt: str = "",
    categories: list[str] | None = None,
) -> DomainConfig:
    """Create a new knowledge domain.

    Returns the created DomainConfig. Raises ValueError if slug already exists.
    """
    slug = slugify(name)
    if not slug:
        raise ValueError(f"Invalid domain name: '{name}'")
    if slug in _domains:
        raise ValueError(f"Domain '{slug}' already exists")

    now = datetime.now(UTC).isoformat()
    collection_name = f"domain_{slug}"

    config = DomainConfig(
        slug=slug,
        name=name,
        description=description,
        language=language,
        system_prompt=system_prompt,
        categories=categories or [],
        created_at=now,
        updated_at=now,
        chunk_count=0,
        collection_name=collection_name,
    )

    _save_config(config)
    _domains[slug] = config

    logger.info("Created domain: %s (collection=%s)", slug, collection_name)
    return config


def get_domain(slug: str) -> DomainConfig:
    """Get domain config by slug. Raises KeyError if not found."""
    if slug not in _domains:
        # Try loading from disk (might have been created by another process)
        config = _load_config(slug)
        if config:
            _domains[slug] = config
        else:
            raise KeyError(f"Domain '{slug}' not found")
    return _domains[slug]


def list_domains() -> list[DomainConfig]:
    """List all registered domains."""
    return list(_domains.values())


def update_domain(slug: str, **kwargs: Any) -> DomainConfig:
    """Update domain config fields. Returns updated config."""
    config = get_domain(slug)
    allowed = {"name", "description", "language", "system_prompt", "categories", "chunk_count"}
    for k, v in kwargs.items():
        if k in allowed:
            setattr(config, k, v)
    config.updated_at = datetime.now(UTC).isoformat()
    _save_config(config)
    _domains[slug] = config
    return config


def delete_domain(slug: str) -> None:
    """Delete a domain and its config. Does NOT delete ChromaDB data."""
    get_domain(slug)  # raises KeyError if not found
    path = _config_path(slug)
    if path.exists():
        path.unlink()
    # Remove empty dir
    domain_dir = DOMAINS_DIR / slug
    if domain_dir.exists():
        with contextlib.suppress(OSError):
            domain_dir.rmdir()
    _domains.pop(slug, None)
    logger.info("Deleted domain: %s", slug)


def get_domain_prompt(slug: str) -> str:
    """Get the effective system prompt for a domain."""
    config = get_domain(slug)
    return _build_system_prompt(config)


def get_domain_collection_name(slug: str) -> str:
    """Get the ChromaDB collection name for a domain."""
    config = get_domain(slug)
    return config.collection_name


# ---------------------------------------------------------------------------
# Auto-domain detection
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "medicina": [
        "cancer",
        "tumor",
        "oncologia",
        "diagnostico",
        "paciente",
        "tratamiento",
        "farmaco",
        "clinico",
        "sintoma",
        "enfermedad",
        "hospital",
        "cirugia",
        "medical",
        "patient",
        "diagnosis",
        "treatment",
        "drug",
        "clinical",
        "disease",
        "symptom",
        "surgery",
        "therapy",
        "pathology",
    ],
    "medioambiente": [
        "clima",
        "contaminacion",
        "emisiones",
        "co2",
        "biodiversidad",
        "ecosistema",
        "sostenible",
        "residuos",
        "reciclaje",
        "deforestacion",
        "climate",
        "pollution",
        "emissions",
        "biodiversity",
        "ecosystem",
        "sustainable",
        "waste",
        "recycling",
        "deforestation",
        "carbon",
    ],
    "finanzas": [
        "inversion",
        "mercado",
        "accion",
        "bono",
        "riesgo",
        "portfolio",
        "rendimiento",
        "inflacion",
        "banco",
        "credito",
        "fintech",
        "investment",
        "market",
        "stock",
        "bond",
        "risk",
        "portfolio",
        "yield",
        "inflation",
        "bank",
        "credit",
    ],
    "derecho": [
        "ley",
        "contrato",
        "sentencia",
        "tribunal",
        "demanda",
        "recurso",
        "jurisprudencia",
        "normativa",
        "regulacion",
        "compliance",
        "law",
        "contract",
        "court",
        "lawsuit",
        "regulation",
        "legal",
    ],
    "ingenieria": [
        "software",
        "api",
        "database",
        "algorithm",
        "deploy",
        "server",
        "framework",
        "architecture",
        "microservice",
        "kubernetes",
        "docker",
        "cloud",
        "aws",
        "devops",
        "ci/cd",
        "pipeline",
        "testing",
    ],
}


def detect_domain(text_sample: str) -> str | None:
    """Detect the most likely domain from a text sample.

    Returns domain slug or None if no clear match.
    Uses keyword frequency — no LLM call.
    """
    text_lower = text_sample.lower()
    scores: dict[str, int] = {}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score >= 3:  # minimum 3 keyword matches
            scores[domain] = score

    if not scores:
        return None

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info("Auto-detected domain: %s (score=%d)", best, scores[best])
    return best
