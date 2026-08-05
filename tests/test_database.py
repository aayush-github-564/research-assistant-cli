import threading
import time

import pytest

from research_assistant_cli.database import Database
from research_assistant_cli.models import SearchResult


@pytest.fixture
def db(tmp_path):
    database = Database(
        shutdown_event=threading.Event(),
        db_path=tmp_path / "test.db",
    )
    yield database
    database.close()


def test_save_search_and_get_recent_searches(db):
    results = [
        SearchResult(title="Result A", url="https://a.com", snippet="Snippet A"),
        SearchResult(title="Result B", url="https://b.com", snippet="Snippet B"),
    ]

    search_id = db.save_search("python decorators", results)
    recent = db.get_recent_searches(limit=5)

    assert len(recent) == 1
    assert recent[0]["id"] == search_id
    assert recent[0]["query"] == "python decorators"


def test_get_results_for_search(db):
    results = [
        SearchResult(title="Result A", url="https://a.com", snippet="Snippet A"),
        SearchResult(title="Result B", url="https://b.com", snippet="Snippet B"),
    ]
    search_id = db.save_search("python decorators", results)

    fetched = db.get_results_for_search(search_id)

    assert len(fetched) == 2
    assert fetched[0].title == "Result A"
    assert fetched[1].url == "https://b.com"


def test_search_history_matches_keyword(db):
    db.save_search("python decorators", [])
    db.save_search("python generators", [])
    db.save_search("javascript closures", [])

    matches = db.search_history("python")

    assert len(matches) == 2
    queries = [m["query"] for m in matches]
    assert "python decorators" in queries
    assert "python generators" in queries


def test_search_history_no_match_returns_empty(db):
    db.save_search("python decorators", [])

    matches = db.search_history("rust")

    assert matches == []


def test_cache_miss_returns_none(db):
    result = db.get_cached_result("nonexistent-key", ttl_seconds=60)

    assert result is None


def test_cache_set_then_get_returns_results(db):
    results = [
        SearchResult(title="Cached", url="https://cached.com", snippet="Cached snippet")
    ]

    db.set_cached_result("some-key", results)
    cached = db.get_cached_result("some-key", ttl_seconds=60)

    assert cached == results


def test_cache_expired_returns_none(db, monkeypatch):
    results = [
        SearchResult(title="Old", url="https://old.com", snippet="Stale snippet")
    ]
    db.set_cached_result("some-key", results)

    # simulate time passing well beyond the TTL
    real_time = time.time
    monkeypatch.setattr(
        "research_assistant_cli.database.time.time", lambda: real_time() + 3600
    )

    cached = db.get_cached_result("some-key", ttl_seconds=60)

    assert cached is None
