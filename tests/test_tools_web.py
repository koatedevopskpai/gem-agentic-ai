"""
TESTS: WEB SEARCH TOOL (MOCKED PATHS)
=====================================
Tests formatting paths and error handling in the web search tool
that require mocking the DuckDuckGo client.
"""

import sys
import types

from gem_agentic_ai.tools.web import web_search


def _patch_duckduckgo(cls):
    """Replace duckduckgo_search.DDGS with a custom class for the duration.

    This is necessary because web_search() does a lazy import:
        from duckduckgo_search import DDGS

    To intercept this, we replace the module in sys.modules with a
    fake module that has DDGS = our custom class.
    """
    original = sys.modules.get("duckduckgo_search")
    fake = types.ModuleType("duckduckgo_search")
    fake.DDGS = cls

    class FakeDDGS(cls):
        """Wrapper so DDGS() returns an instance of the supplied class."""
        def __new__(subcls, *args, **kwargs):
            return cls(*args, **kwargs)

    fake.DDGS = FakeDDGS
    sys.modules["duckduckgo_search"] = fake

    def restore():
        if original:
            sys.modules["duckduckgo_search"] = original
        else:
            sys.modules.pop("duckduckgo_search", None)

    return restore


class MockDDGSBase:
    """Base class for mock DDGS context managers."""
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class TestWebSearchErrorPaths:
    """Error paths that require mocking the DuckDuckGo client."""

    def test_search_exception_returns_message(self) -> None:
        """When DDGS raises an exception, return a descriptive error."""

        class FailingDDGS(MockDDGSBase):
            def text(self, query, max_results=5):
                raise ConnectionError("Simulated network failure")

        restore = _patch_duckduckgo(FailingDDGS)
        try:
            result = web_search.invoke({"query": "test", "max_results": 1})
        finally:
            restore()

        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()


class TestWebSearchFormattingPaths:
    """Tests for the formatting/fallback paths."""

    def test_no_results_message(self) -> None:
        """When DDGS returns an empty list, return 'No results found.'"""

        class EmptyDDGS(MockDDGSBase):
            def text(self, query, max_results=5):
                return []

        restore = _patch_duckduckgo(EmptyDDGS)
        try:
            result = web_search.invoke({"query": "test", "max_results": 5})
        finally:
            restore()

        assert isinstance(result, str)
        assert "no results" in result.lower()

    def test_import_error_path(self) -> None:
        """When duckduckgo_search library cannot be imported, return a helpful error.

        This test forces the import inside web_search() to fail by clearing
        the module from sys.modules and removing the installed library.
        We simulate a missing library by temporarily placing a module without
        the DDGS attribute.
        """

        class NoDDGS_Module(types.ModuleType):
            """Module with no DDGS attribute — causes ImportError on from-import."""

        restore = _patch_duckduckgo(MockDDGSBase)
        # But this time our mock module won't have DDGS...
        # Actually, let's just put a module without DDGS into sys.modules.
        original_ddgs = sys.modules.get("duckduckgo_search")
        fake = types.ModuleType("duckduckgo_search")
        # Deliberately NOT setting DDGS
        sys.modules["duckduckgo_search"] = fake

        try:
            result = web_search.invoke({"query": "test", "max_results": 1})
        finally:
            if original_ddgs:
                sys.modules["duckduckgo_search"] = original_ddgs
            else:
                sys.modules.pop("duckduckgo_search", None)

        assert isinstance(result, str)
        assert "error" in result.lower() or "not available" in result.lower()

    def test_formatted_results(self) -> None:
        """When DDGS returns results, they should be properly formatted."""

        class ResultDDGS(MockDDGSBase):
            def text(self, query, max_results=5):
                return [
                    {
                        "title": "Test Result",
                        "href": "https://example.com",
                        "body": "This is a test snippet.",
                    },
                ]

        restore = _patch_duckduckgo(ResultDDGS)
        try:
            result = web_search.invoke({"query": "test", "max_results": 5})
        finally:
            restore()

        assert isinstance(result, str)
        assert "Test Result" in result
        assert "https://example.com" in result
        assert "test snippet" in result

    def test_missing_fields(self) -> None:
        """DDGS results with missing fields should still be handled gracefully."""

        class PartialDDGS(MockDDGSBase):
            def text(self, query, max_results=5):
                return [
                    {
                        # title is missing
                        "href": "https://example.com",
                        # body is missing
                    },
                ]

        restore = _patch_duckduckgo(PartialDDGS)
        try:
            result = web_search.invoke({"query": "test", "max_results": 5})
        finally:
            restore()

        assert isinstance(result, str)
        # Should use "Untitled" as fallback title.
        assert "Untitled" in result

    def test_multiple_results(self) -> None:
        """Multiple results should be separated by double newlines."""

        class MultiDDGS(MockDDGSBase):
            def text(self, query, max_results=5):
                return [
                    {"title": "First", "href": "https://a.com", "body": "First snippet"},
                    {"title": "Second", "href": "https://b.com", "body": "Second snippet"},
                ]

        restore = _patch_duckduckgo(MultiDDGS)
        try:
            result = web_search.invoke({"query": "test", "max_results": 5})
        finally:
            restore()

        # Two results should be separated by exactly two newlines.
        assert "First snippet" in result
        assert "Second snippet" in result
        assert "\n\n" in result
