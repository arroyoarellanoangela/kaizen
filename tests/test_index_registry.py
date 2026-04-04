"""Tests for rag.index_registry — registry contents, routing, interface."""

from suyven_rag.rag.index_registry import IndexInfo, _registry, list_indexes, route_to_index


class TestListIndexes:
    def test_returns_list(self):
        result = list_indexes()
        assert isinstance(result, list)

    def test_has_default(self):
        assert "default" in list_indexes()


class TestRouteToIndex:
    """V2.1: route_to_index always returns 'default'."""

    def test_default_routing(self):
        assert route_to_index("any query") == "default"

    def test_with_hint(self):
        assert route_to_index("any query", hint="ai") == "default"

    def test_empty_query(self):
        assert route_to_index("") == "default"

    def test_none_hint(self):
        assert route_to_index("test", hint=None) == "default"


class TestRegistryContents:
    def test_default_index_exists(self):
        assert "default" in _registry

    def test_default_index_info(self):
        info = _registry["default"]
        assert isinstance(info, IndexInfo)
        assert info.collection_name  # not empty
        assert info.embed_model == "default_embed"
