import asyncio
import os
from abc import ABC, abstractmethod

import requests
from duckduckgo_search import DDGS
from tavily import TavilyClient

from models import SearchResult
from resilience import retry_with_backoff
from cache import cached_search
from logger import setup_logging


logger = setup_logging()


class SearchProvider(ABC):
    """Contract: any search provider must implement `.search(query)`
    and return a list of SearchResult objects."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


async def search_multiple(
    providers: list[SearchProvider], query: str
) -> list[SearchResult]:
    """Runs .search() for each provider concurrently, merges results,
    and dedups by URL (first occurrence wins)."""
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, p.search, query) for p in providers]
    provider_results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[SearchResult] = []
    seen_urls: set[str] = set()

    for provider, result in zip(providers, provider_results):
        if isinstance(result, Exception):
            logger.warning(f"{type(provider).__name__} failed: {result}")
            continue

        for r in result:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                merged.append(r)

    return merged


class DuckDuckGoSearchProvider(SearchProvider):
    def __init__(self, db):
        self.db = db

    @cached_search()
    @retry_with_backoff(max_attempts=3, retry_on=(Exception,))
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raw_results = DDGS(timeout=10).text(query, max_results=max_results)

        return [
            SearchResult(
                title=r.get("title", "No title"),
                url=r.get("href", "No URL"),
                snippet=r.get("body", "No description"),
            )
            for r in raw_results
        ]


class TavilySearchProvider(SearchProvider):
    def __init__(self, db):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Missing TAVILY_API_KEY. Check your .env file.")
        self.client = TavilyClient(api_key=api_key)
        self.db = db

    @cached_search()
    @retry_with_backoff(
        max_attempts=3, retry_on=(requests.exceptions.RequestException,)
    )
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = self.client.search(query, max_results=max_results, timeout=10)
        raw_results = response.get("results", [])

        return [
            SearchResult(
                title=r.get("title", "No title"),
                url=r.get("url", "No URL"),
                snippet=r.get("content", "No description"),
            )
            for r in raw_results
        ]
