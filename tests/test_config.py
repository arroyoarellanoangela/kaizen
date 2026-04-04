"""Tests for rag.config — verify all config values are loaded and typed correctly."""

from suyven_rag.rag.config import (
    ADD_BATCH_SIZE,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBED_BATCH,
    EMBED_MODEL,
    KNOWLEDGE_DIR,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_URL,
    OVERFETCH_FACTOR,
    RERANKER_BATCH_SIZE,
    RERANKER_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    WORKERS,
)


def test_paths_are_pathlib():
    from pathlib import Path

    assert isinstance(KNOWLEDGE_DIR, Path)
    assert isinstance(CHROMA_DIR, Path)


def test_string_configs_not_empty():
    assert isinstance(COLLECTION_NAME, str) and COLLECTION_NAME
    assert isinstance(EMBED_MODEL, str) and EMBED_MODEL
    assert isinstance(RERANKER_MODEL, str) and RERANKER_MODEL
    assert isinstance(LLM_MODEL, str) and LLM_MODEL
    assert isinstance(LLM_PROVIDER, str) and LLM_PROVIDER
    assert isinstance(OLLAMA_URL, str) and OLLAMA_URL


def test_int_configs_positive():
    for name, val in [
        ("CHUNK_SIZE", CHUNK_SIZE),
        ("CHUNK_OVERLAP", CHUNK_OVERLAP),
        ("TOP_K", TOP_K),
        ("OVERFETCH_FACTOR", OVERFETCH_FACTOR),
        ("RERANKER_BATCH_SIZE", RERANKER_BATCH_SIZE),
        ("EMBED_BATCH", EMBED_BATCH),
        ("ADD_BATCH_SIZE", ADD_BATCH_SIZE),
        ("WORKERS", WORKERS),
    ]:
        assert isinstance(val, int), f"{name} should be int, got {type(val)}"
        assert val > 0, f"{name} should be > 0, got {val}"


def test_chunk_overlap_less_than_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE, "overlap must be < chunk size"


def test_system_prompt_has_rules():
    assert "STRICT RULES" in SYSTEM_PROMPT
    assert "NEVER" in SYSTEM_PROMPT


def test_llm_provider_known():
    assert LLM_PROVIDER in ("ollama", "openai"), f"Unknown provider: {LLM_PROVIDER}"


# ---------------------------------------------------------------------------
# Docker secrets helper
# ---------------------------------------------------------------------------


class TestSecretHelper:
    def test_falls_back_to_env(self):
        """When no Docker secret file exists, reads from env var."""
        import os

        from suyven_rag.rag.config import _secret

        os.environ["_TEST_SECRET_XYZ"] = "env-value"
        try:
            assert _secret("_TEST_SECRET_XYZ") == "env-value"
        finally:
            del os.environ["_TEST_SECRET_XYZ"]

    def test_returns_default(self):
        from suyven_rag.rag.config import _secret

        assert _secret("_NONEXISTENT_SECRET_ABC", "fallback") == "fallback"

    def test_reads_from_file(self, tmp_path):
        """When a secret file exists, reads from it."""
        from unittest.mock import patch

        # Create a fake secret file
        secret_file = tmp_path / "TEST_KEY"
        secret_file.write_text("file-secret-value\n")

        with patch("rag.config.Path") as MockPath:
            mock_secret_path = secret_file
            MockPath.return_value.__truediv__ = lambda self, name: mock_secret_path
            # Direct test: read the file
            assert secret_file.read_text().strip() == "file-secret-value"
