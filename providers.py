import asyncio
import os
from abc import ABC, abstractmethod

from duckduckgo_search import DDGS
from tavily import TavilyClient

from models import SearchResult


class SearchProvider(ABC):
    """Contract: any search provider must implement `.search(query)`
    and return a list of SearchResult objects."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        raise NotImplementedError


async def search_multiple(providers: list[SearchProvider], query: str) -> dict:
    """Runs .search() for each provider concurrently."""
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, p.search, query) for p in providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip([type(p).__name__ for p in providers], results))


class DuckDuckGoSearchProvider(SearchProvider):
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
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Missing TAVILY_API_KEY. Check your .env file.")
        self.client = TavilyClient(api_key=api_key)

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
