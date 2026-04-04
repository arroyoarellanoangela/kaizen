"""Security middleware — API key auth, rate limiting, input validation.

Usage in api.py:
    from rag.security import require_api_key, RateLimiter, validate_query

Auth: API keys from env var API_KEYS (comma-separated).
      If API_KEYS is empty/unset, auth is DISABLED (dev mode).
      Keys are passed via X-API-Key header.

Rate limiting: Sliding window per API key, configurable via env.
Input validation: Query length, injection patterns, field constraints.
"""

import html
import logging
import os
import re
import time
from collections import defaultdict
from threading import Lock

from dotenv import load_dotenv
from fastapi import HTTPException, Request

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------


def _read_secret(name: str, default: str = "") -> str:
    """Read from Docker secret (/run/secrets/<name>) or env var."""
    from pathlib import Path

    secret_path = Path("/run/secrets") / name
    try:
        if secret_path.is_file():
            return secret_path.read_text().strip()
    except (OSError, PermissionError):
        pass
    return os.getenv(name, default)


_raw_keys = _read_secret("API_KEYS").strip()
API_KEYS: set[str] = {k.strip() for k in _raw_keys.split(",") if k.strip()} if _raw_keys else set()

RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))  # requests per minute
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))  # max burst in 1 second
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))  # max chars per query
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))  # max retrieval results

# CORS origins — comma-separated, empty = allow all (dev mode)
_raw_origins = os.getenv("CORS_ORIGINS", "").strip()
CORS_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()] if _raw_origins else []
)

AUTH_ENABLED = len(API_KEYS) > 0


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


def require_api_key(request: Request) -> str:
    """FastAPI dependency — validates X-API-Key header.

    Returns the API key if valid. Raises 401 if invalid.
    If AUTH_ENABLED is False (no keys configured), returns "dev".
    """
    if not AUTH_ENABLED:
        return "dev"

    key = request.headers.get("X-API-Key", "").strip()
    if not key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ---------------------------------------------------------------------------
# Rate limiter — sliding window
# ---------------------------------------------------------------------------


class RateLimiter:
    """In-memory sliding window rate limiter.

    Thread-safe. Tracks per-key request timestamps.
    """

    def __init__(self, rpm: int = RATE_LIMIT_RPM, burst: int = RATE_LIMIT_BURST):
        self.rpm = rpm
        self.burst = burst
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str) -> None:
        """Check if request is allowed. Raises 429 if rate exceeded."""
        now = time.time()

        with self._lock:
            timestamps = self._windows[key]

            # Prune old entries (older than 60s)
            cutoff = now - 60.0
            self._windows[key] = [t for t in timestamps if t > cutoff]
            timestamps = self._windows[key]

            # Check per-minute limit
            if len(timestamps) >= self.rpm:
                retry_after = int(timestamps[0] + 60 - now) + 1
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({self.rpm}/min). Retry after {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Check burst limit (requests in last 1 second)
            one_sec_ago = now - 1.0
            recent = sum(1 for t in timestamps if t > one_sec_ago)
            if recent >= self.burst:
                raise HTTPException(
                    status_code=429,
                    detail=f"Burst limit exceeded ({self.burst}/sec). Slow down.",
                    headers={"Retry-After": "1"},
                )

            timestamps.append(now)

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit state. If key is None, reset all."""
        with self._lock:
            if key:
                self._windows.pop(key, None)
            else:
                self._windows.clear()


# Global rate limiter instance
rate_limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

# Patterns that suggest prompt injection or abuse
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
]


def validate_query(query: str) -> str:
    """Validate and sanitize a query string.

    Returns the cleaned query. Raises 400 on validation failure.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long ({len(query)} chars, max {MAX_QUERY_LENGTH})",
        )

    # Check for prompt injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            logger.warning("[security] Potential prompt injection blocked: %s", query[:100])
            raise HTTPException(
                status_code=400,
                detail="Query contains disallowed patterns",
            )

    # Sanitize: strip null bytes, normalize whitespace
    query = query.replace("\x00", "")
    query = re.sub(r"\s+", " ", query)

    return query


def validate_top_k(top_k: int) -> int:
    """Validate top_k parameter."""
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    if top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=400,
            detail=f"top_k too large ({top_k}, max {MAX_TOP_K})",
        )
    return top_k


def sanitize_text(text: str, max_length: int = 5000) -> str:
    """Sanitize arbitrary text input (descriptions, prompts, etc.)."""
    if not text:
        return ""
    text = text.strip()
    text = text.replace("\x00", "")
    if len(text) > max_length:
        text = text[:max_length]
    # Escape HTML entities to prevent XSS if rendered
    text = html.escape(text)
    return text


def validate_slug(slug: str) -> str:
    """Validate a URL slug."""
    if not slug or not slug.strip():
        raise HTTPException(status_code=400, detail="Slug cannot be empty")
    slug = slug.strip().lower()
    if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", slug):
        raise HTTPException(
            status_code=400,
            detail="Invalid slug format (lowercase letters, numbers, hyphens only)",
        )
    if len(slug) > 50:
        raise HTTPException(status_code=400, detail="Slug too long (max 50 chars)")
    return slug


def validate_domain_name(name: str) -> str:
    """Validate domain name."""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Domain name cannot be empty")
    name = name.strip()
    if len(name) > 100:
        raise HTTPException(status_code=400, detail="Domain name too long (max 100 chars)")
    return name


def validate_directory_path(path: str) -> str:
    """Validate a directory path — prevent path traversal."""
    if not path or not path.strip():
        raise HTTPException(status_code=400, detail="Directory path cannot be empty")
    path = path.strip()
    # Block path traversal
    if ".." in path:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return path
