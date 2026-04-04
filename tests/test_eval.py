"""Tests for rag.eval — flagging heuristics, insufficient detection, log writer."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from suyven_rag.rag.eval import (
    LATENCY_SPIKE_S,
    RERANKER_FLOOR,
    RERANKER_WEAK_MEAN,
    QueryEvalRecord,
    compute_flags,
    detect_insufficient,
    log_eval,
    new_query_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**overrides) -> QueryEvalRecord:
    """Build a healthy record with overrides."""
    defaults = dict(
        timestamp="2026-03-11T12:00:00Z",
        query_id="abc123",
        query="test query",
        category_filter=None,
        top_k=5,
        route_mode="answer",
        route_reason="default path",
        route_indexes=["default"],
        num_results=5,
        reranker_scores=[2.0, 1.5, 1.0, 0.5, 0.2],
        bi_encoder_scores=[0.8, 0.75, 0.7, 0.65, 0.6],
        max_reranker_score=2.0,
        min_reranker_score=0.2,
        mean_reranker_score=1.04,
        source_categories=["ai", "ai", "ai", "ai", "ai"],
        llm_said_insufficient=False,
        response_length=200,
        token_count_approx=40,
        latency_total_s=3.0,
        latency_retrieval_s=2.5,
        latency_llm_s=0.5,
        flags=[],
    )
    defaults.update(overrides)
    return QueryEvalRecord(**defaults)


# ---------------------------------------------------------------------------
# compute_flags
# ---------------------------------------------------------------------------


class TestComputeFlags:
    def test_healthy_record_no_flags(self):
        record = _make_record()
        assert compute_flags(record) == []

    def test_empty_retrieval(self):
        record = _make_record(
            num_results=0,
            reranker_scores=[],
            bi_encoder_scores=[],
            max_reranker_score=None,
            min_reranker_score=None,
            mean_reranker_score=None,
            source_categories=[],
        )
        flags = compute_flags(record)
        assert "empty_retrieval" in flags
        # Should short-circuit: no other flags
        assert flags == ["empty_retrieval"]

    def test_retrieval_failure(self):
        bad_scores = [RERANKER_FLOOR - 0.1] * 5
        record = _make_record(
            reranker_scores=bad_scores,
            mean_reranker_score=RERANKER_FLOOR - 0.1,
            min_reranker_score=RERANKER_FLOOR - 0.1,
            max_reranker_score=RERANKER_FLOOR - 0.1,
        )
        flags = compute_flags(record)
        assert "retrieval_failure" in flags

    def test_weak_retrieval(self):
        record = _make_record(
            mean_reranker_score=RERANKER_WEAK_MEAN - 0.1,
            reranker_scores=[RERANKER_WEAK_MEAN - 0.1] * 5,
        )
        flags = compute_flags(record)
        assert "weak_retrieval" in flags

    def test_corpus_gap(self):
        record = _make_record(llm_said_insufficient=True)
        flags = compute_flags(record)
        assert "corpus_gap" in flags

    def test_category_contamination_answer_mode(self):
        cats = ["ai", "data-eng", "backend", "devops", "infra"]
        record = _make_record(source_categories=cats, route_mode="answer")
        flags = compute_flags(record)
        assert "category_contamination" in flags

    def test_category_contamination_not_triggered_summary(self):
        cats = ["ai", "data-eng", "backend", "devops", "infra"]
        record = _make_record(source_categories=cats, route_mode="summary")
        flags = compute_flags(record)
        assert "category_contamination" not in flags

    def test_category_contamination_below_threshold(self):
        cats = ["ai", "ai", "ai", "backend", "backend"]
        record = _make_record(source_categories=cats, route_mode="answer")
        flags = compute_flags(record)
        assert "category_contamination" not in flags  # only 2 distinct

    def test_latency_spike(self):
        record = _make_record(latency_total_s=LATENCY_SPIKE_S + 1)
        flags = compute_flags(record)
        assert "latency_spike" in flags

    def test_latency_normal(self):
        record = _make_record(latency_total_s=3.0)
        flags = compute_flags(record)
        assert "latency_spike" not in flags

    def test_multiple_flags(self):
        record = _make_record(
            llm_said_insufficient=True,
            latency_total_s=LATENCY_SPIKE_S + 1,
        )
        flags = compute_flags(record)
        assert "corpus_gap" in flags
        assert "latency_spike" in flags


# ---------------------------------------------------------------------------
# detect_insufficient
# ---------------------------------------------------------------------------


class TestDetectInsufficient:
    @pytest.mark.parametrize(
        "text",
        [
            "The context is insufficient to answer this question.",
            "There is not enough information in the provided context.",
            "No relevant context found for this query.",
            "I cannot answer from the provided context.",
            "I cannot compare from the provided context.",
            "I cannot determine from the provided context.",
            "No information is provided about this topic.",
            "No tengo suficiente contexto para responder.",
            "La informacion insuficiente no permite responder.",
            "La informaci\u00f3n insuficiente no permite responder.",
            "Context insufficient to determine the answer.",
        ],
    )
    def test_positive(self, text):
        assert detect_insufficient(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Claude is an AI model developed by Anthropic.",
            "The transformer architecture uses self-attention.",
            "Here is a comparison of GPT-4 and Gemini.",
            "RAG works by retrieving relevant context first.",
            "",
        ],
    )
    def test_negative(self, text):
        assert detect_insufficient(text) is False


# ---------------------------------------------------------------------------
# log_eval
# ---------------------------------------------------------------------------


class TestLogEval:
    def test_writes_valid_jsonl(self):
        record = _make_record()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test_log.jsonl"
            with patch("rag.eval._LOG_FILE", log_file), patch("rag.eval._LOG_DIR", Path(tmpdir)):
                log_eval(record)

            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["query"] == "test query"
            assert data["route_mode"] == "answer"
            assert isinstance(data["reranker_scores"], list)

    def test_appends_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test_log.jsonl"
            with patch("rag.eval._LOG_FILE", log_file), patch("rag.eval._LOG_DIR", Path(tmpdir)):
                log_eval(_make_record(query="first"))
                log_eval(_make_record(query="second"))

            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestNewQueryId:
    def test_length(self):
        qid = new_query_id()
        assert len(qid) == 12

    def test_unique(self):
        ids = {new_query_id() for _ in range(100)}
        assert len(ids) == 100
