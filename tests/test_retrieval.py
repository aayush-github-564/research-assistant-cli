import numpy as np

from research_assistant_cli.retrieval import retrieve_relevant_chunks


class FakeProvider:
    name = "fake-model"

    def __init__(self, query_vector):
        self._query_vector = query_vector

    def embed(self, texts, input_type):
        assert input_type == "query"
        return [self._query_vector]


class FakeDb:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_all_chunks(self):
        return self._chunks


def test_retrieve_ranks_closer_vector_first():
    chunks = [
        {"chunk_text": "close", "embedding": np.array([1.0, 0.0], dtype=np.float32),
         "embedding_model": "fake-model", "title": "A", "url": "https://a.com", "query": "q"},
        {"chunk_text": "far", "embedding": np.array([0.0, 1.0], dtype=np.float32),
         "embedding_model": "fake-model", "title": "B", "url": "https://b.com", "query": "q"},
    ]
    db = FakeDb(chunks)
    provider = FakeProvider(np.array([1.0, 0.0], dtype=np.float32))

    results = retrieve_relevant_chunks(db, provider, "query", top_k=2)

    assert results[0]["chunk_text"] == "close"
    assert results[0]["similarity"] > results[1]["similarity"]


def test_retrieve_filters_out_model_mismatch():
    chunks = [
        {"chunk_text": "wrong model", "embedding": np.array([1.0, 0.0], dtype=np.float32),
         "embedding_model": "other-model", "title": "A", "url": "https://a.com", "query": "q"},
    ]
    db = FakeDb(chunks)
    provider = FakeProvider(np.array([1.0, 0.0], dtype=np.float32))

    assert retrieve_relevant_chunks(db, provider, "query") == []


def test_retrieve_returns_empty_when_no_chunks_stored():
    db = FakeDb([])
    provider = FakeProvider(np.array([1.0, 0.0], dtype=np.float32))

    assert retrieve_relevant_chunks(db, provider, "query") == []