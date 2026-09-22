import threading
import time

import pytest
import numpy as np

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

def test_save_and_get_result_ids_for_search(db):
    results = [
        SearchResult(title="A", url="https://a.com", snippet="snippet a"),
        SearchResult(title="B", url="https://b.com", snippet="snippet b"),
    ]
    search_id = db.save_search("test query", results)

    result_ids = db.get_result_ids_for_search(search_id)

    assert len(result_ids) == 2


def test_save_chunks_and_get_all_chunks_roundtrip(db):
    results = [SearchResult(title="A", url="https://a.com", snippet="snippet a")]
    search_id = db.save_search("test query", results)
    result_id = db.get_result_ids_for_search(search_id)[0]

    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    db.save_chunks(result_id, ["chunk one"], [embedding], "fake-model")

    all_chunks = db.get_all_chunks()

    assert len(all_chunks) == 1
    stored = all_chunks[0]
    assert stored["chunk_text"] == "chunk one"
    assert stored["embedding_model"] == "fake-model"
    assert stored["title"] == "A"
    assert stored["url"] == "https://a.com"
    assert stored["query"] == "test query"
    np.testing.assert_array_almost_equal(stored["embedding"], embedding)


def test_save_chunks_rolls_back_on_shutdown(db):
    results = [SearchResult(title="A", url="https://a.com", snippet="snippet a")]
    search_id = db.save_search("test query", results)
    result_id = db.get_result_ids_for_search(search_id)[0]

    db.shutdown_event.set()
    embedding = np.zeros(3, dtype=np.float32)

    with pytest.raises(SystemExit):
        db.save_chunks(result_id, ["c1", "c2"], [embedding, embedding], "fake-model")

    assert db.get_all_chunks() == []