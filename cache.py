import functools
import hashlib
import json

DEFAULT_CACHE_TTL = 3600  # 1 hour


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

            cached = self.db.get_cached_result(key, ttl_seconds)
            if cached is not None:
                return cached

            results = func(self, query, max_results, **kwargs)
            self.db.set_cached_result(key, results)
            return results

        return wrapper

    return decorator
