import asyncio
import os
from abc import ABC, abstractmethod

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException
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
    and dedups by URL (first occurrence wins).

    NOTE: each provider's .search() runs in a separate worker thread
    (via run_in_executor). All providers share one Database instance
    for caching — Database now uses check_same_thread=False plus an
    internal lock, specifically to make this safe. If you ever swap
    Database for something else, that thread-safety requirement goes
    with it.
    """
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

    # Narrowed from (Exception,) to the actual transient failure types:
    # DDGSException covers the library's own network/rate-limit errors
    # (duckduckgo_search was renamed to ddgs upstream; DDGSException
    # replaces the old DuckDuckGoSearchException), and requests exceptions
    # cover the underlying HTTP layer. A bug in our own code (e.g.
    # AttributeError from a bad self.db) will now surface immediately
    # instead of being silently retried 3 times.
    @cached_search()
    @retry_with_backoff(
        max_attempts=3,
        retry_on=(DDGSException, requests.exceptions.RequestException),
    )
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
