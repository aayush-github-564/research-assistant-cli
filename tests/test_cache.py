import pytest

from research_assistant_cli.cache import cached_search, make_cache_key
from research_assistant_cli.models import SearchResult


def test_make_cache_key_is_deterministic():
    key1 = make_cache_key("TavilySearchProvider", "python decorators", 5)
    key2 = make_cache_key("TavilySearchProvider", "python decorators", 5)

    assert key1 == key2


def test_make_cache_key_differs_on_max_results():
    key1 = make_cache_key("TavilySearchProvider", "python decorators", 5)
    key2 = make_cache_key("TavilySearchProvider", "python decorators", 10)

    assert key1 != key2


class FakeDB:
    """Controllable stand-in for Database — lets each test dictate
    exactly what get/set_cached_result do, without touching sqlite."""

    def __init__(self, cached_value=None, raise_on_get=False, raise_on_set=False):
        self.cached_value = cached_value
        self.raise_on_get = raise_on_get
        self.raise_on_set = raise_on_set
        self.set_calls = []

    def get_cached_result(self, key, ttl_seconds):
        if self.raise_on_get:
            raise sqlite3_like_error()
        return self.cached_value

    def set_cached_result(self, key, results):
        if self.raise_on_set:
            raise sqlite3_like_error()
        self.set_calls.append((key, results))


def sqlite3_like_error():
    return Exception("simulated corrupt row / db error")


class FakeProvider:
    def __init__(self, db):
        self.db = db
        self.call_count = 0

    @cached_search()
    def search(self, query, max_results=5):
        self.call_count += 1
        return [
            SearchResult(title="Fresh", url="https://fresh.com", snippet="fresh result")
        ]


def test_cache_hit_skips_underlying_function():
    cached_results = [
        SearchResult(title="Cached", url="https://cached.com", snippet="from cache")
    ]
    db = FakeDB(cached_value=cached_results)
    provider = FakeProvider(db=db)

    results = provider.search("python decorators")

    assert results == cached_results
    assert provider.call_count == 0  # never actually called .search's real body


def test_cache_miss_calls_function_and_stores_result():
    db = FakeDB(cached_value=None)
    provider = FakeProvider(db=db)

    results = provider.search("python decorators")

    assert provider.call_count == 1
    assert results[0].title == "Fresh"
    assert len(db.set_calls) == 1
    stored_key, stored_results = db.set_calls[0]
    assert stored_results == results


def test_cache_read_failure_falls_back_to_search():
    db = FakeDB(raise_on_get=True)
    provider = FakeProvider(db=db)

    results = provider.search("python decorators")

    assert provider.call_count == 1
    assert results[0].title == "Fresh"


def test_cache_write_failure_still_returns_results():
    db = FakeDB(cached_value=None, raise_on_set=True)
    provider = FakeProvider(db=db)

    results = provider.search("python decorators")

    assert provider.call_count == 1
    assert results[0].title == "Fresh"
