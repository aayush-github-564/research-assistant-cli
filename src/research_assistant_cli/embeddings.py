import os
from abc import ABC, abstractmethod

import cohere
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig

import httpx
from cohere.core.api_error import ApiError
from google.genai.errors import APIError as GeminiAPIError

from .resilience import retry_with_backoff


class EmbeddingProvider(ABC):
    """Contract: any embedding provider must implement `.embed(texts, input_type)`
    and return a list of numpy arrays, one per input text."""

    name: str  # stored per-chunk in the DB — must stay stable per provider

    @abstractmethod
    def embed(self, texts: list[str], input_type: str) -> list[np.ndarray]:
        """input_type is 'document' when embedding chunks for storage,
        'query' when embedding a --ask question. Each provider maps this
        to whatever its own API expects."""
        raise NotImplementedError


class CohereEmbeddingProvider(EmbeddingProvider):
    name = "cohere-embed-v4.0"

    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("Missing COHERE_API_KEY. Check your .env file.")
        self.client = cohere.ClientV2(api_key=api_key)

    @retry_with_backoff(max_attempts=3, retry_on=(ApiError, httpx.HTTPError))
    def embed(self, texts: list[str], input_type: str) -> list[np.ndarray]:
        cohere_type = "search_document" if input_type == "document" else "search_query"
        response = self.client.embed(
            texts=texts,
            model="embed-v4.0",
            input_type=cohere_type,
            embedding_types=["float"],
        )
        return [np.array(vec, dtype=np.float32) for vec in response.embeddings.float]


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini-text-embedding-005"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY. Check your .env file.")
        self.client = genai.Client(api_key=api_key)

    @retry_with_backoff(max_attempts=3, retry_on=(GeminiAPIError, httpx.HTTPError))
    def embed(self, texts: list[str], input_type: str) -> list[np.ndarray]:
        task_type = "RETRIEVAL_DOCUMENT" if input_type == "document" else "RETRIEVAL_QUERY"
        response = self.client.models.embed_content(
            model="text-embedding-005",
            contents=texts,
            config=EmbedContentConfig(task_type=task_type),
        )
        return [np.array(e.values, dtype=np.float32) for e in response.embeddings]


def get_embedding_provider() -> EmbeddingProvider:
    """Picks the provider from EMBEDDING_PROVIDER in .env (default: cohere)."""
    provider_name = os.getenv("EMBEDDING_PROVIDER", "cohere").lower()
    if provider_name == "cohere":
        return CohereEmbeddingProvider()
    if provider_name == "gemini":
        return GeminiEmbeddingProvider()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider_name!r}")