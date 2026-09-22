import numpy as np
import pytest

from research_assistant_cli.embeddings import (
    CohereEmbeddingProvider,
    GeminiEmbeddingProvider,
    get_embedding_provider,
)


class FakeCohereResponse:
    def __init__(self, vectors):
        self.embeddings = type("E", (), {"float": vectors})()


class FakeCohereClient:
    def __init__(self, api_key=None):
        pass

    def embed(self, texts, model, input_type, embedding_types):
        return FakeCohereResponse([[0.1, 0.2, 0.3] for _ in texts])


def test_cohere_embed_returns_numpy_float32_arrays(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    monkeypatch.setattr(
        "research_assistant_cli.embeddings.cohere.ClientV2", FakeCohereClient
    )

    result = CohereEmbeddingProvider().embed(["a", "b"], input_type="document")

    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)
    assert result[0].dtype == np.float32


def test_cohere_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    with pytest.raises(ValueError):
        CohereEmbeddingProvider()


class FakeGeminiModels:
    def embed_content(self, model, contents, config):
        vectors = [[0.1, 0.2] for _ in contents]
        return type("R", (), {"embeddings": [type("V", (), {"values": v})() for v in vectors]})()


class FakeGeminiClient:
    def __init__(self, api_key=None):
        self.models = FakeGeminiModels()


def test_gemini_embed_returns_numpy_arrays(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(
        "research_assistant_cli.embeddings.genai.Client", FakeGeminiClient
    )

    result = GeminiEmbeddingProvider().embed(["a"], input_type="query")

    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)


def test_gemini_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError):
        GeminiEmbeddingProvider()


def test_get_embedding_provider_defaults_to_cohere(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(
        "research_assistant_cli.embeddings.cohere.ClientV2", FakeCohereClient
    )

    assert isinstance(get_embedding_provider(), CohereEmbeddingProvider)


def test_get_embedding_provider_unknown_name_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "not-a-real-provider")

    with pytest.raises(ValueError):
        get_embedding_provider()