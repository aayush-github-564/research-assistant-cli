import numpy as np

from research_assistant_cli.ingestion import embed_and_store_results
from research_assistant_cli.models import SearchResult


class FakeProvider:
    name = "fake-model"

    def embed(self, texts, input_type):
        assert input_type == "document"
        return [np.zeros(3, dtype=np.float32) for _ in texts]


class FakeDb:
    def __init__(self, result_ids):
        self._result_ids = result_ids
        self.saved = []

    def get_result_ids_for_search(self, search_id):
        return self._result_ids

    def save_chunks(self, result_id, chunks, embeddings, model):
        self.saved.append((result_id, chunks, embeddings, model))


def test_embed_and_store_uses_fetched_page_text(monkeypatch):
    monkeypatch.setattr(
        "research_assistant_cli.ingestion.fetch_page_text", lambda url: "full page text"
    )
    monkeypatch.setattr(
        "research_assistant_cli.ingestion.chunk_text",
        lambda text, **kwargs: ["chunk 1", "chunk 2"],
    )

    db = FakeDb(result_ids=[101])
    results = [SearchResult(title="T", url="https://a.com", snippet="snip")]

    embed_and_store_results(db, FakeProvider(), search_id=1, results=results)

    assert len(db.saved) == 1
    result_id, chunks, embeddings, model = db.saved[0]
    assert result_id == 101
    assert chunks == ["chunk 1", "chunk 2"]
    assert model == "fake-model"


def test_embed_and_store_falls_back_to_snippet_when_fetch_fails(monkeypatch):
    def raise_connection_error(url):
        raise ConnectionError("boom")

    monkeypatch.setattr(
        "research_assistant_cli.ingestion.fetch_page_text", raise_connection_error
    )
    captured = {}

    def fake_chunk_text(text, **kwargs):
        captured["text"] = text
        return ["snippet chunk"]

    monkeypatch.setattr("research_assistant_cli.ingestion.chunk_text", fake_chunk_text)

    db = FakeDb(result_ids=[101])
    results = [SearchResult(title="T", url="https://a.com", snippet="the snippet text")]

    embed_and_store_results(db, FakeProvider(), search_id=1, results=results)

    assert captured["text"] == "the snippet text"
    assert len(db.saved) == 1


def test_embed_and_store_skips_result_with_no_chunks(monkeypatch):
    monkeypatch.setattr("research_assistant_cli.ingestion.fetch_page_text", lambda url: "text")
    monkeypatch.setattr("research_assistant_cli.ingestion.chunk_text", lambda text, **kwargs: [])

    db = FakeDb(result_ids=[101])
    results = [SearchResult(title="T", url="https://a.com", snippet="snip")]

    embed_and_store_results(db, FakeProvider(), search_id=1, results=results)

    assert db.saved == []