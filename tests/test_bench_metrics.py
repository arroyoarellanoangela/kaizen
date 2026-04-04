"""Tests for rag/bench_metrics.py — all pure functions with known answers."""

import math

import pytest

from suyven_rag.rag.bench_metrics import (
    binary_relevance,
    contamination_check,
    faithfulness_embedding,
    keyword_coverage,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# -----------------------------------------------------------------------
# binary_relevance
# -----------------------------------------------------------------------


class TestBinaryRelevance:
    def test_all_relevant(self):
        assert binary_relevance(["a", "b"], ["a", "b"]) == [1, 1]

    def test_none_relevant(self):
        assert binary_relevance(["a", "b"], ["c"]) == [0, 0]

    def test_partial(self):
        assert binary_relevance(["a", "b", "c"], ["b", "c"]) == [0, 1, 1]

    def test_case_insensitive(self):
        assert binary_relevance(["Doc_A", "DOC_B"], ["doc_a"]) == [1, 0]

    def test_empty_retrieved(self):
        assert binary_relevance([], ["a"]) == []

    def test_empty_relevant(self):
        assert binary_relevance(["a", "b"], []) == [0, 0]


# -----------------------------------------------------------------------
# NDCG
# -----------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self):
        # All relevant at top — NDCG should be 1.0
        assert ndcg_at_k([1, 1, 0, 0, 0], 5) == 1.0

    def test_inverted_ranking(self):
        # Relevant docs at bottom — NDCG < 1.0
        score = ndcg_at_k([0, 0, 0, 1, 1], 5)
        assert 0.0 < score < 1.0

    def test_no_relevant(self):
        assert ndcg_at_k([0, 0, 0, 0], 4) == 0.0

    def test_single_relevant_at_top(self):
        assert ndcg_at_k([1, 0, 0], 3) == 1.0

    def test_single_relevant_at_bottom(self):
        score = ndcg_at_k([0, 0, 1], 3)
        # DCG = 1/log2(4), IDCG = 1/log2(2) = 1.0
        expected = (1 / math.log2(4)) / (1 / math.log2(2))
        assert abs(score - expected) < 1e-9

    def test_empty(self):
        assert ndcg_at_k([], 5) == 0.0

    def test_k_larger_than_list(self):
        # Should handle gracefully
        assert ndcg_at_k([1, 0], 10) == 1.0


# -----------------------------------------------------------------------
# MRR
# -----------------------------------------------------------------------


class TestMRR:
    def test_first_position(self):
        assert mrr_at_k([1, 0, 0], 3) == 1.0

    def test_second_position(self):
        assert mrr_at_k([0, 1, 0], 3) == 0.5

    def test_third_position(self):
        assert abs(mrr_at_k([0, 0, 1], 3) - 1 / 3) < 1e-9

    def test_no_relevant(self):
        assert mrr_at_k([0, 0, 0], 3) == 0.0

    def test_k_truncates(self):
        # Relevant at position 4, but k=3
        assert mrr_at_k([0, 0, 0, 1], 3) == 0.0


# -----------------------------------------------------------------------
# Recall
# -----------------------------------------------------------------------


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k([1, 1], 2, total_relevant=2) == 1.0

    def test_partial_recall(self):
        assert recall_at_k([1, 0, 1, 0], 4, total_relevant=4) == 0.5

    def test_zero_relevant_total(self):
        assert recall_at_k([0, 0], 2, total_relevant=0) == 0.0

    def test_overcounted(self):
        # More found than total_relevant shouldn't exceed 1.0
        assert recall_at_k([1, 1, 1], 3, total_relevant=2) == 1.0


# -----------------------------------------------------------------------
# Precision
# -----------------------------------------------------------------------


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k([1, 1, 1], 3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k([0, 0, 0], 3) == 0.0

    def test_partial(self):
        assert abs(precision_at_k([1, 0, 1, 0, 0], 5) - 0.4) < 1e-9

    def test_empty(self):
        assert precision_at_k([], 5) == 0.0


# -----------------------------------------------------------------------
# Keyword coverage
# -----------------------------------------------------------------------


class TestKeywordCoverage:
    def test_full_coverage(self):
        assert (
            keyword_coverage(
                "RAG uses retrieval and generation", ["rag", "retrieval", "generation"]
            )
            == 1.0
        )

    def test_partial_coverage(self):
        assert keyword_coverage(
            "RAG uses retrieval", ["rag", "retrieval", "generation"]
        ) == pytest.approx(2 / 3)

    def test_no_coverage(self):
        assert keyword_coverage("hello world", ["rag", "retrieval"]) == 0.0

    def test_empty_keywords(self):
        assert keyword_coverage("anything", []) == 1.0

    def test_case_insensitive(self):
        assert keyword_coverage("RAG RETRIEVAL", ["rag", "retrieval"]) == 1.0


# -----------------------------------------------------------------------
# Contamination
# -----------------------------------------------------------------------


class TestContamination:
    def test_detected(self):
        assert contamination_check(["good", "bad", "good2"], ["bad"]) is True

    def test_clean(self):
        assert contamination_check(["good", "good2"], ["bad"]) is False

    def test_empty_irrelevant(self):
        assert contamination_check(["anything"], []) is False

    def test_case_insensitive(self):
        assert contamination_check(["BAD_Source"], ["bad_source"]) is True


# -----------------------------------------------------------------------
# Faithfulness (embedding similarity)
# -----------------------------------------------------------------------


class TestFaithfulness:
    def _mock_embed(self, texts: list[str]) -> list[list[float]]:
        """Deterministic mock: hash-based pseudo-embeddings."""
        result = []
        for t in texts:
            vec = [0.0] * 8
            for i, ch in enumerate(t[:8]):
                vec[i] = ord(ch) / 128.0
            result.append(vec)
        return result

    def test_identical_texts(self):
        score = faithfulness_embedding("hello world", "hello world", self._mock_embed)
        assert abs(score - 1.0) < 1e-6

    def test_different_texts(self):
        score = faithfulness_embedding("hello world", "zzzzz yyyyy", self._mock_embed)
        assert 0.0 <= score < 1.0

    def test_empty_answer(self):
        assert faithfulness_embedding("", "some context", self._mock_embed) == 0.0

    def test_empty_context(self):
        assert faithfulness_embedding("some answer", "", self._mock_embed) == 0.0
