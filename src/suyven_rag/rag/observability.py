"""Observability — structured logging, request tracing, and metrics.

Zero new dependencies: uses stdlib logging with JSON formatter,
FastAPI middleware for request tracing, and simple in-memory counters
exposed at /metrics in Prometheus text format.
"""

import contextvars
import json
import logging
import time
from collections import defaultdict
from threading import Lock
from typing import Any
from uuid import uuid4

# ---------------------------------------------------------------------------
# Structured JSON log formatter
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Includes timestamp, level, logger name, message, and any extra fields
    passed via the `extra` kwarg to logging calls.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Propagate request_id if present on the record
        if hasattr(record, "request_id"):
            entry["request_id"] = record.request_id
        # Propagate any extra fields set via `extra={...}`
        for key in (
            "method",
            "path",
            "status",
            "duration_ms",
            "query_id",
            "api_key_hint",
            "client_ip",
            "error",
        ):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        # Exception info
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(json_logs: bool = True, level: str = "INFO") -> None:
    """Replace the root logger's handlers with structured JSON output.

    Call once at startup (before any log calls). When json_logs=False
    falls back to the plain-text format for local dev.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s  %(message)s"))
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Request-scoped context (trace ID propagation)
# ---------------------------------------------------------------------------

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id_var.get("")


def set_request_id(rid: str) -> None:
    _request_id_var.set(rid)


def new_request_id() -> str:
    return uuid4().hex[:12]


class RequestIdFilter(logging.Filter):
    """Inject current request_id into every log record automatically."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Metrics collector (Prometheus-compatible text exposition)
# ---------------------------------------------------------------------------


class Metrics:
    """Thread-safe in-memory metrics: counters and histograms.

    Exposed via /metrics in Prometheus text format.
    No external dependencies (no prometheus_client needed).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    # -- Counters --

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    # -- Histograms (simple: store raw values, compute quantiles on read) --

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            bucket = self._histograms[key]
            bucket.append(value)
            # Cap at 10k samples to bound memory
            if len(bucket) > 10_000:
                self._histograms[key] = bucket[-5_000:]

    # -- Export --

    def export_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"{key} {val}")
            for key, values in sorted(self._histograms.items()):
                if not values:
                    continue
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                lines.append(f"{key}_count {n}")
                lines.append(f"{key}_sum {sum(sorted_vals):.4f}")
                # Quantiles
                for q in (0.5, 0.9, 0.95, 0.99):
                    idx = min(int(q * n), n - 1)
                    lines.append(f'{key}{{quantile="{q}"}} {sorted_vals[idx]:.4f}')
        return "\n".join(lines) + "\n" if lines else ""

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics instance
metrics = Metrics()


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------


def create_request_middleware(app_metrics: Metrics | None = None):
    """Return an ASGI middleware that traces every request.

    Adds:
      - X-Request-ID header (generated or forwarded)
      - Request duration logging (structured JSON)
      - Counter + histogram metrics per endpoint
    """
    m = app_metrics or metrics

    async def middleware(request, call_next):
        # Extract or generate request ID
        rid = request.headers.get("X-Request-ID", "") or new_request_id()
        set_request_id(rid)

        method = request.method
        path = request.url.path

        t0 = time.time()
        try:
            response = await call_next(request)
            duration_ms = (time.time() - t0) * 1000
            status = response.status_code

            # Metrics
            m.inc(
                "http_requests_total",
                labels={"method": method, "path": path, "status": str(status)},
            )
            m.observe(
                "http_request_duration_ms", duration_ms, labels={"method": method, "path": path}
            )

            # Structured log
            logger = logging.getLogger("suyven.http")
            logger.info(
                "%s %s -> %d (%.0fms)",
                method,
                path,
                status,
                duration_ms,
                extra={
                    "method": method,
                    "path": path,
                    "status": status,
                    "duration_ms": round(duration_ms, 1),
                },
            )

            # Propagate request ID to response
            response.headers["X-Request-ID"] = rid
            return response
        except Exception:
            duration_ms = (time.time() - t0) * 1000
            m.inc("http_requests_total", labels={"method": method, "path": path, "status": "500"})
            m.observe(
                "http_request_duration_ms", duration_ms, labels={"method": method, "path": path}
            )
            raise

    return middleware
