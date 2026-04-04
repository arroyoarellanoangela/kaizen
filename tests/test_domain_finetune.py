"""Tests for finetune/domain_finetune.py — domain-specific embedding fine-tuning."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from suyven_rag.finetune.domain_finetune import (
    DomainFinetuneConfig,
    DomainFinetuneResult,
    _extract_first_sentence,
    _generate_definition_pairs,
    _generate_first_sentence_pairs,
    _generate_question_pairs,
    generate_domain_pairs,
    sample_domain_chunks,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_chunks(n: int = 100) -> list[dict]:
    """Generate realistic test chunks."""
    chunks = []
    for i in range(n):
        chunks.append(
            {
                "text": f"Sentence number {i} explains a concept. "
                f"This chunk covers topic {i} in detail with enough "
                f"content to be useful for training. The algorithm processes "
                f"data through multiple stages of transformation.",
                "source": f"doc_{i % 10}",
                "category": f"cat_{i % 5}",
            }
        )
    # Add some definition-style chunks
    chunks.append(
        {
            "text": "Transformer Architecture is a neural network design that uses "
            "self-attention mechanisms to process sequences in parallel. "
            "Unlike RNNs, transformers can handle long-range dependencies.",
            "source": "ml_guide",
            "category": "ml",
        }
    )
    chunks.append(
        {
            "text": "To configure a VPC with private subnets, you need to create "
            "a NAT gateway in the public subnet and update route tables.",
            "source": "aws_guide",
            "category": "cloud",
        }
    )
    return chunks


# -----------------------------------------------------------------------
# First sentence extraction
# -----------------------------------------------------------------------


class TestExtractFirstSentence:
    def test_extracts_first_meaningful(self):
        text = "This is a good opening sentence about RAG systems. More details follow here."
        result = _extract_first_sentence(text)
        assert result is not None
        assert "RAG systems" in result

    def test_skips_short_sentences(self):
        text = "Hi. This is the actual meaningful sentence about the topic at hand."
        result = _extract_first_sentence(text)
        assert result is not None
        assert len(result) >= 20

    def test_skips_code_lines(self):
        text = "def func(x): return x * 2. This is a normal explanatory sentence about data."
        result = _extract_first_sentence(text)
        # Should skip the code-like first line
        assert result is not None

    def test_returns_none_for_empty(self):
        assert _extract_first_sentence("") is None
        assert _extract_first_sentence("   ") is None

    def test_returns_none_for_very_short(self):
        # All sentences under 20 chars
        text = "x = 1. y = 2. z = 3."
        assert _extract_first_sentence(text) is None


# -----------------------------------------------------------------------
# Pair generation strategies
# -----------------------------------------------------------------------


class TestFirstSentencePairs:
    def test_generates_pairs(self):
        chunks = _make_chunks(50)
        pairs = _generate_first_sentence_pairs(chunks, max_pairs=10)
        assert len(pairs) > 0
        assert len(pairs) <= 10
        for p in pairs:
            assert "query" in p
            assert "positive" in p
            assert p["strategy"] == "first_sentence"

    def test_respects_max_pairs(self):
        chunks = _make_chunks(200)
        pairs = _generate_first_sentence_pairs(chunks, max_pairs=5)
        assert len(pairs) <= 5

    def test_removes_first_sentence_from_positive(self):
        chunks = _make_chunks(50)
        pairs = _generate_first_sentence_pairs(chunks, max_pairs=5)
        for p in pairs:
            # The query (first sentence) should NOT be in the positive
            assert p["query"] not in p["positive"]


class TestDefinitionPairs:
    def test_detects_definitions(self):
        chunks = _make_chunks(10)  # includes the Transformer Architecture chunk
        pairs = _generate_definition_pairs(chunks, max_pairs=100)
        assert len(pairs) > 0
        for p in pairs:
            assert p["strategy"] == "definition"

    def test_respects_max_pairs(self):
        chunks = _make_chunks(100)
        pairs = _generate_definition_pairs(chunks, max_pairs=3)
        assert len(pairs) <= 3


class TestQuestionPairs:
    def test_generates_questions(self):
        chunks = _make_chunks(10)
        pairs = _generate_question_pairs(chunks, max_pairs=100)
        # May or may not find patterns depending on chunk text
        for p in pairs:
            assert p["strategy"] == "question_reformat"
            assert "?" in p["query"]


# -----------------------------------------------------------------------
# Domain pair generation (integration of all strategies)
# -----------------------------------------------------------------------


class TestGenerateDomainPairs:
    @patch("finetune.domain_finetune._filter_with_reranker")
    def test_combines_strategies(self, mock_filter):
        """Test that all strategies are combined and filtered."""
        # Mock reranker to pass everything through
        mock_filter.side_effect = lambda pairs, **kw: pairs

        chunks = _make_chunks(100)
        pairs = generate_domain_pairs(chunks, target=50, min_score=0.0)
        assert len(pairs) > 0
        assert len(pairs) <= 50

        # Should have a mix of strategies
        strategies = {p.get("strategy") for p in pairs}
        assert len(strategies) >= 1  # At least one strategy produced pairs

    @patch("finetune.domain_finetune._filter_with_reranker")
    def test_deduplicates(self, mock_filter):
        mock_filter.side_effect = lambda pairs, **kw: pairs
        chunks = _make_chunks(50)
        pairs = generate_domain_pairs(chunks, target=100, min_score=0.0)
        # Check no duplicates
        seen = set()
        for p in pairs:
            key = (p["query"][:100], p["positive"][:100])
            assert key not in seen, f"Duplicate pair found: {key}"
            seen.add(key)


# -----------------------------------------------------------------------
# Sample domain chunks (mocked)
# -----------------------------------------------------------------------


class TestSampleDomainChunks:
    @patch("rag.index_registry.get_index")
    def test_loads_chunks(self, mock_get_index):
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.get.return_value = {
            "documents": ["chunk 1", "chunk 2", "chunk 3"],
            "metadatas": [
                {"source": "doc1", "category": "cat1"},
                {"source": "doc2", "category": "cat1"},
                {"source": "doc3", "category": "cat2"},
            ],
        }
        mock_get_index.return_value = mock_col

        chunks = sample_domain_chunks("test-domain")
        assert len(chunks) == 3
        assert chunks[0]["text"] == "chunk 1"
        mock_get_index.assert_called_with("domain_test-domain")

    @patch("rag.index_registry.get_index")
    def test_empty_collection(self, mock_get_index):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_get_index.return_value = mock_col

        chunks = sample_domain_chunks("empty-domain")
        assert chunks == []


# -----------------------------------------------------------------------
# Config and result dataclasses
# -----------------------------------------------------------------------


class TestDomainFinetuneConfig:
    def test_defaults(self):
        config = DomainFinetuneConfig(slug="test")
        assert config.slug == "test"
        assert config.min_pairs == 200
        assert config.epochs == 3
        assert config.lora_rank == 8
        assert config.batch_size == 32

    def test_custom(self):
        config = DomainFinetuneConfig(
            slug="oncologia",
            epochs=5,
            lora_rank=16,
            batch_size=16,
        )
        assert config.epochs == 5
        assert config.lora_rank == 16


class TestDomainFinetuneResult:
    def test_defaults(self):
        result = DomainFinetuneResult(slug="test", status="success")
        assert result.slug == "test"
        assert result.status == "success"
        assert result.pairs_generated == 0
        assert result.error == ""


# -----------------------------------------------------------------------
# Model registry integration
# -----------------------------------------------------------------------


class TestRegisterDomainModel:
    @patch("rag.model_registry.register_embed_model")
    def test_registers_model(self, mock_register):
        from suyven_rag.finetune.domain_finetune import DOMAIN_FT_DIR, register_domain_model

        merged = DOMAIN_FT_DIR / "test" / "checkpoints" / "merged_model"
        merged.mkdir(parents=True, exist_ok=True)

        try:
            register_domain_model("test")
            mock_register.assert_called_once_with(
                "domain_test_embed",
                str(merged),
            )
        finally:
            # Cleanup
            import shutil

            shutil.rmtree(DOMAIN_FT_DIR / "test", ignore_errors=True)

    def test_raises_on_missing_checkpoint(self):
        from suyven_rag.finetune.domain_finetune import register_domain_model

        with pytest.raises(FileNotFoundError):
            register_domain_model("nonexistent-domain-xyz")


# -----------------------------------------------------------------------
# Full pipeline (mocked heavy deps)
# -----------------------------------------------------------------------


class TestRunDomainFinetune:
    @patch("finetune.domain_finetune.register_domain_model")
    @patch("finetune.domain_finetune.train_domain_model")
    @patch("finetune.domain_finetune._filter_with_reranker")
    @patch("finetune.domain_finetune.sample_domain_chunks")
    @patch("rag.domain_registry.get_domain")
    @patch("rag.domain_registry.update_domain")
    def test_success_path(
        self, mock_update, mock_get_domain, mock_sample, mock_filter, mock_train, mock_register
    ):
        from suyven_rag.finetune.domain_finetune import DOMAIN_FT_DIR, run_domain_finetune

        mock_get_domain.return_value = MagicMock(name="Test Domain")
        mock_sample.return_value = _make_chunks(300)
        mock_filter.side_effect = lambda pairs, **kw: pairs[:250]
        mock_train.return_value = {
            "train_pairs": 225,
            "eval_pairs": 25,
            "final_train_loss": 0.5,
            "final_eval_loss": 0.6,
        }
        mock_register.return_value = Path("/fake/path")

        result = run_domain_finetune("test-domain")

        assert result.status == "success"
        assert result.train_pairs == 225
        assert result.final_train_loss == 0.5
        mock_train.assert_called_once()
        mock_register.assert_called_once_with("test-domain")

        # Cleanup
        import shutil

        shutil.rmtree(DOMAIN_FT_DIR / "test-domain", ignore_errors=True)

    @patch("finetune.domain_finetune.sample_domain_chunks")
    @patch("rag.domain_registry.get_domain")
    def test_insufficient_data(self, mock_get_domain, mock_sample):
        from suyven_rag.finetune.domain_finetune import run_domain_finetune

        mock_get_domain.return_value = MagicMock(name="Test")
        mock_sample.return_value = _make_chunks(5)  # Too few chunks

        config = DomainFinetuneConfig(slug="tiny", min_pairs=200)
        result = run_domain_finetune("tiny", config)

        assert result.status == "insufficient_data"
        assert "chunks" in result.error
        assert "need at least" in result.error

    @patch("rag.domain_registry.get_domain")
    def test_domain_not_found(self, mock_get_domain):
        from suyven_rag.finetune.domain_finetune import run_domain_finetune

        mock_get_domain.side_effect = KeyError("not found")

        result = run_domain_finetune("nonexistent")
        assert result.status == "error"
        assert "not found" in result.error
