"""Tests for rag.observability — structured logging, metrics, request tracing."""

import json
import logging

from suyven_rag.rag.observability import (
    JSONFormatter,
    Metrics,
    RequestIdFilter,
    get_request_id,
    new_request_id,
    set_request_id,
)

# -----------------------------------------------------------------------
# JSON Formatter
# -----------------------------------------------------------------------


class TestJSONFormatter:
    def test_basic_output_is_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        line = formatter.format(record)
        parsed = json.loads(line)
        assert parsed["msg"] == "hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"

    def test_includes_ts(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warn",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "ts" in parsed

    def test_propagates_request_id(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc123"  # type: ignore[attr-defined]
        parsed = json.loads(formatter.format(record))
        assert parsed["request_id"] == "abc123"

    def test_propagates_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.method = "GET"  # type: ignore[attr-defined]
        record.path = "/api/health"  # type: ignore[attr-defined]
        record.status = 200  # type: ignore[attr-defined]
        record.duration_ms = 42.5  # type: ignore[attr-defined]
        parsed = json.loads(formatter.format(record))
        assert parsed["method"] == "GET"
        assert parsed["path"] == "/api/health"
        assert parsed["status"] == 200
        assert parsed["duration_ms"] == 42.5

    def test_includes_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="x",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="error",
                args=(),
                exc_info=sys.exc_info(),
            )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "boom" in parsed["exception"]

    def test_no_request_id_if_not_set(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert "request_id" not in parsed


# -----------------------------------------------------------------------
# Request ID context
# -----------------------------------------------------------------------


class TestRequestId:
    def test_new_request_id_format(self):
        rid = new_request_id()
        assert len(rid) == 12
        assert rid.isalnum()

    def test_unique(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_set_and_get(self):
        set_request_id("test-123")
        assert get_request_id() == "test-123"
        # Clean up
        set_request_id("")

    def test_default_empty(self):
        set_request_id("")
        assert get_request_id() == ""


# -----------------------------------------------------------------------
# RequestIdFilter
# -----------------------------------------------------------------------


class TestRequestIdFilter:
    def test_injects_request_id(self):
        set_request_id("filter-test-456")
        f = RequestIdFilter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        f.filter(record)
        assert record.request_id == "filter-test-456"  # type: ignore[attr-defined]
        set_request_id("")

    def test_always_returns_true(self):
        f = RequestIdFilter()
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is True


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------


class TestMetrics:
    def test_counter_inc(self):
        m = Metrics()
        m.inc("test_counter")
        assert m.get_counter("test_counter") == 1.0
        m.inc("test_counter", 5)
        assert m.get_counter("test_counter") == 6.0

    def test_counter_with_labels(self):
        m = Metrics()
        m.inc("requests", labels={"method": "GET", "status": "200"})
        m.inc("requests", labels={"method": "POST", "status": "200"})
        assert m.get_counter("requests", labels={"method": "GET", "status": "200"}) == 1.0
        assert m.get_counter("requests", labels={"method": "POST", "status": "200"}) == 1.0

    def test_counter_default_zero(self):
        m = Metrics()
        assert m.get_counter("nonexistent") == 0.0

    def test_histogram_observe(self):
        m = Metrics()
        for v in [10, 20, 30, 40, 50]:
            m.observe("latency", v)
        export = m.export_prometheus()
        assert "latency_count 5" in export
        assert "latency_sum 150.0000" in export

    def test_histogram_quantiles(self):
        m = Metrics()
        for v in range(1, 101):
            m.observe("duration", float(v))
        export = m.export_prometheus()
        assert 'duration{quantile="0.5"}' in export
        assert 'duration{quantile="0.99"}' in export

    def test_export_empty(self):
        m = Metrics()
        assert m.export_prometheus() == ""

    def test_reset(self):
        m = Metrics()
        m.inc("x")
        m.observe("y", 1.0)
        m.reset()
        assert m.get_counter("x") == 0.0
        assert m.export_prometheus() == ""

    def test_histogram_caps_at_10k(self):
        m = Metrics()
        for i in range(10_001):
            m.observe("big", float(i))
        # After cap, should have trimmed to 5000
        export = m.export_prometheus()
        assert "big_count 5000" in export

    def test_prometheus_format_counters(self):
        m = Metrics()
        m.inc("http_total", labels={"method": "GET", "path": "/api/health"})
        export = m.export_prometheus()
        assert 'http_total{method="GET",path="/api/health"}' in export

    def test_labels_sorted(self):
        m = Metrics()
        m.inc("x", labels={"z": "1", "a": "2"})
        export = m.export_prometheus()
        # Labels should be sorted alphabetically
        assert 'x{a="2",z="1"}' in export
