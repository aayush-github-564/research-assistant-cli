import pytest
from ddgs.exceptions import DDGSException

from research_assistant_cli.providers import (
    DuckDuckGoSearchProvider,
    TavilySearchProvider,
    SearchProvider,
    search_multiple,
)
from research_assistant_cli.models import SearchResult


class FakeDDGS:
    def __init__(self, timeout=None):
        pass

    def text(self, query, max_results=5):
        return [
            {"title": "Result A", "href": "https://a.com", "body": "Snippet A"},
        ]


class FakeDDGSMissingFields:
    def __init__(self, timeout=None):
        pass

    def text(self, query, max_results=5):
        return [{}]  # no title/href/body at all


class FakeDDGSRaises:
    def __init__(self, timeout=None):
        pass

    def text(self, query, max_results=5):
        raise DDGSException("simulated rate limit")


class FakeTavilyClient:
    def __init__(self, api_key=None):
        pass

    def search(self, query, max_results=5, timeout=10):
        return {
            "results": [
                {"title": "Result B", "url": "https://b.com", "content": "Snippet B"},
            ]
        }


def test_duckduckgo_search_success(monkeypatch, db):
    monkeypatch.setattr("research_assistant_cli.providers.DDGS", FakeDDGS)

    provider = DuckDuckGoSearchProvider(db=db)
    results = provider.search("python decorators")

    assert results == [
        SearchResult(title="Result A", url="https://a.com", snippet="Snippet A")
    ]


def test_duckduckgo_search_missing_fields_uses_defaults(monkeypatch, db):
    monkeypatch.setattr("research_assistant_cli.providers.DDGS", FakeDDGSMissingFields)

    provider = DuckDuckGoSearchProvider(db=db)
    results = provider.search("obscure query")

    assert results == [
        SearchResult(title="No title", url="No URL", snippet="No description")
    ]


def test_duckduckgo_search_raises_after_retries_exhausted(monkeypatch, db):
    monkeypatch.setattr("research_assistant_cli.providers.DDGS", FakeDDGSRaises)
    monkeypatch.setattr(
        "research_assistant_cli.resilience.time.sleep", lambda seconds: None
    )

    provider = DuckDuckGoSearchProvider(db=db)

    with pytest.raises(DDGSException):
        provider.search("python decorators")


def test_tavily_search_success(monkeypatch, db):
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key-for-testing")
    monkeypatch.setattr(
        "research_assistant_cli.providers.TavilyClient", FakeTavilyClient
    )

    provider = TavilySearchProvider(db=db)
    results = provider.search("python decorators")

    assert results == [
        SearchResult(title="Result B", url="https://b.com", snippet="Snippet B")
    ]


def test_tavily_missing_api_key_raises(monkeypatch, db):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError):
        TavilySearchProvider(db=db)


class WorkingProvider(SearchProvider):
    def __init__(self, results):
        self._results = results

    def search(self, query, max_results=5):
        return self._results


class FailingProvider(SearchProvider):
    def search(self, query, max_results=5):
        raise ConnectionError("simulated network failure")


def test_search_multiple_merges_and_dedups_by_url():
    provider_a = WorkingProvider(
        [
            SearchResult(title="A", url="https://shared.com", snippet="from A"),
            SearchResult(title="A2", url="https://a-only.com", snippet="from A"),
        ]
    )
    provider_b = WorkingProvider(
        [
            SearchResult(title="B", url="https://shared.com", snippet="from B"),
            SearchResult(title="B2", url="https://b-only.com", snippet="from B"),
        ]
    )

    results = asyncio_run(search_multiple([provider_a, provider_b], "query"))

    urls = [r.url for r in results]
    assert urls == ["https://shared.com", "https://a-only.com", "https://b-only.com"]
    # first occurrence wins on the duplicate URL
    shared = next(r for r in results if r.url == "https://shared.com")
    assert shared.title == "A"


def test_search_multiple_partial_failure_returns_working_results():
    working = WorkingProvider(
        [SearchResult(title="Good", url="https://good.com", snippet="fine")]
    )
    failing = FailingProvider()

    results = asyncio_run(search_multiple([working, failing], "query"))

    assert results == [
        SearchResult(title="Good", url="https://good.com", snippet="fine")
    ]


def test_search_multiple_total_failure_returns_empty_list():
    results = asyncio_run(
        search_multiple([FailingProvider(), FailingProvider()], "query")
    )

    assert results == []


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
