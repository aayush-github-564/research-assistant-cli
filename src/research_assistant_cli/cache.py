import functools
import hashlib
import json

from .logger import setup_logging

logger = setup_logging()

DEFAULT_CACHE_TTL = 5  # 1 hour


def make_cache_key(provider_name: str, query: str, max_results: int) -> str:
    raw = json.dumps(
        {"provider": provider_name, "query": query, "max_results": max_results},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def cached_search(ttl_seconds: int = DEFAULT_CACHE_TTL):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, query, max_results=5, **kwargs):
            key = make_cache_key(type(self).__name__, query, max_results)

            # A cache READ failure (corrupt row, schema drift, bad JSON)
            # should never take down the whole search — a real network
            # call is one line away and can still succeed. Treat any
            # read-side error as a miss, log it, and move on.
            try:
                cached = self.db.get_cached_result(key, ttl_seconds)
            except Exception as e:
                logger.warning(f"Cache read failed for {type(self).__name__}: {e}")
                cached = None

            if cached is not None:
                return cached

            results = func(self, query, max_results, **kwargs)

            # A cache WRITE failure shouldn't discard results we already
            # have in hand — the search succeeded; only the "remember this
            # for next time" step failed. Log it and return results anyway.
            try:
                self.db.set_cached_result(key, results)
            except Exception as e:
                logger.warning(f"Cache write failed for {type(self).__name__}: {e}")

            return results

        return wrapper

    return decorator
