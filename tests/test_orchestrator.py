"""Tests for rag.orchestrator — routing logic, mode detection, RoutePlan structure."""

import pytest

from suyven_rag.rag.orchestrator import RoutePlan, plan

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


class TestModeDetection:
    """Verify deterministic mode classification."""

    @pytest.mark.parametrize(
        "query",
        [
            "best AI model for coding",
            "what is Claude?",
            "how does RAG work?",
            "GPU recommendations",
        ],
    )
    def test_answer_mode(self, query):
        route = plan(query)
        assert route.mode == "answer"

    @pytest.mark.parametrize(
        "query",
        [
            "summarize the architecture of this system",
            "compare GPT-4 and Gemini in detail",
            "overview of transformer models",
            "pros and cons of Ollama vs vLLM",
        ],
    )
    def test_summary_mode(self, query):
        route = plan(query)
        assert route.mode == "summary"

    def test_summary_long_query(self):
        """Queries > 150 chars should trigger summary mode."""
        long_query = "Explain the differences between " + "a " * 100
        assert len(long_query) > 150
        route = plan(long_query)
        assert route.mode == "summary"

    @pytest.mark.parametrize(
        "query",
        [
            "def fibonacci(n): how to implement?",
            "pip install sentence-transformers failing",
            "SELECT * FROM users WHERE active = 1",
            "docker compose up not working",
            "git push origin main rejected",
            "npm install react-query",
            "import torch; model.to('cuda')",
        ],
    )
    def test_code_mode(self, query):
        route = plan(query)
        assert route.mode == "code"


# ---------------------------------------------------------------------------
# RoutePlan structure
# ---------------------------------------------------------------------------


class TestRoutePlan:
    """Verify RoutePlan fields are populated correctly."""

    def test_plan_returns_routeplan(self):
        route = plan("test query")
        assert isinstance(route, RoutePlan)

    def test_default_index(self):
        route = plan("test query")
        assert route.indexes == ["default"]

    def test_default_models(self):
        route = plan("test query")
        assert route.embed_model == "default_embed"
        assert route.reranker_model == "default_reranker"
        assert route.use_reranker is True

    def test_llm_model_populated(self):
        route = plan("test query")
        assert isinstance(route.llm_model, str) and route.llm_model

    def test_reason_populated(self):
        route = plan("test query")
        assert isinstance(route.reason, str) and route.reason

    def test_top_k_default(self):
        route = plan("test query")
        assert route.top_k == 5  # from config default

    def test_top_k_override(self):
        route = plan("test query", top_k=10)
        assert route.top_k == 10

    def test_category_does_not_affect_mode(self):
        """Category hint doesn't change mode classification."""
        route_no_cat = plan("best AI model")
        route_with_cat = plan("best AI model", category="ai")
        assert route_no_cat.mode == route_with_cat.mode


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_query(self):
        route = plan("")
        assert route.mode == "answer"  # default path

    def test_whitespace_query(self):
        route = plan("   ")
        assert route.mode == "answer"

    def test_code_keyword_priority_over_summary(self):
        """If both code and summary keywords present, code wins (checked first)."""
        route = plan("summarize this: def main(): import os")
        assert route.mode == "code"
