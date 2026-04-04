"""Tests for rag.model_registry — registry contents and interface."""

from suyven_rag.rag.model_registry import (
    ModelInfo,
    _registry,
    has_embed_model,
    list_models,
    register_embed_model,
)


class TestListModels:
    def test_returns_dict(self):
        result = list_models()
        assert isinstance(result, dict)

    def test_has_default_embed(self):
        result = list_models()
        assert "default_embed" in result

    def test_has_default_reranker(self):
        result = list_models()
        assert "default_reranker" in result

    def test_model_info_fields(self):
        result = list_models()
        for name, info in result.items():
            assert "model_id" in info, f"{name} missing model_id"
            assert "type" in info, f"{name} missing type"
            assert "precision" in info, f"{name} missing precision"
            assert info["type"] in ("embed", "reranker"), f"{name} has bad type: {info['type']}"
            assert info["precision"] in ("fp16", "fp32"), (
                f"{name} has bad precision: {info['precision']}"
            )


class TestRegistryContents:
    def test_default_embed_model_id(self):
        info = _registry["default_embed"]
        assert isinstance(info, ModelInfo)
        assert info.model_id  # not empty

    def test_default_reranker_model_id(self):
        info = _registry["default_reranker"]
        assert isinstance(info, ModelInfo)
        assert info.model_id  # not empty

    def test_types_correct(self):
        assert _registry["default_embed"].model_type == "embed"
        assert _registry["default_reranker"].model_type == "reranker"


class TestRegisterEmbedModel:
    def test_register_new_model(self):
        register_embed_model("test_embed_xyz", "fake/model/path")
        assert "test_embed_xyz" in _registry
        assert _registry["test_embed_xyz"].model_type == "embed"
        assert _registry["test_embed_xyz"].model_id == "fake/model/path"
        # Cleanup
        _registry.pop("test_embed_xyz", None)

    def test_has_embed_model_true(self):
        assert has_embed_model("default_embed") is True

    def test_has_embed_model_false(self):
        assert has_embed_model("nonexistent_model") is False

    def test_has_embed_model_wrong_type(self):
        assert has_embed_model("default_reranker") is False

    def test_list_models_includes_loaded_field(self):
        result = list_models()
        for _name, info in result.items():
            assert "loaded" in info
            assert isinstance(info["loaded"], bool)
