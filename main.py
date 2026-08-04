import asyncio
import signal
import sys
import threading

import requests

from database import Database
from logger import setup_logging
from providers import DuckDuckGoSearchProvider, TavilySearchProvider, search_multiple

shutdown_event = threading.Event()
logger = setup_logging()


async def run_search(query, db):
    """Constructs both providers and runs them concurrently via search_multiple.

    NOTE: provider construction happens here, synchronously, BEFORE
    search_multiple's internal try/except-per-provider logic ever runs.
    That means a config error (e.g. missing TAVILY_API_KEY -> ValueError)
    raises immediately and is NOT caught by search_multiple's graceful
    per-provider failure handling — it propagates up to main(), which is
    intentional: a broken setup should surface clearly, not be silently
    treated the same as "one provider timed out mid-search."
    """
    providers = [TavilySearchProvider(db), DuckDuckGoSearchProvider(db)]
    return await search_multiple(providers, query)


def handle_sigint(signum, frame):
    print("\nInterrupt received — finishing safely...")
    shutdown_event.set()


async def main():
    signal.signal(signal.SIGINT, handle_sigint)

    db = Database(shutdown_event)

    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "--history":
            recent = db.get_recent_searches(limit=5)
            if not recent:
                print("No past searches yet.")
            for s in recent:
                print(f"#{s['id']}  {s['query']}  ({s['created_at']})")
            return

        if len(sys.argv) >= 3 and sys.argv[1] == "--show":
            try:
                search_id = int(sys.argv[2])
            except ValueError:
                print(f"'{sys.argv[2]}' isn't a valid search ID.")
                return

            results = db.get_results_for_search(search_id)
            if not results:
                print(f"No results found for search #{search_id}.")
            for i, r in enumerate(results, start=1):
                print(f"{i}. {r}")
            return

        if len(sys.argv) >= 3 and sys.argv[1] == "--find":
            keyword = sys.argv[2]
            matches = db.search_history(keyword)
            if not matches:
                print(f"No past searches matching '{keyword}'.")
            for s in matches:
                print(f"#{s['id']}  {s['query']}  ({s['created_at']})")
            return

        if len(sys.argv) < 2:
            print('Usage: uv run python main.py "your search query"')
            sys.exit(1)

        query = sys.argv[1]

        # Config errors (missing API key -> ValueError) surface here,
        # raised during provider construction inside run_search(), before
        # any network activity happens. Network-level failures (timeouts,
        # request errors) are instead handled per-provider INSIDE
        # search_multiple, which logs a warning and continues with
        # whichever provider(s) succeeded — so this except block is only
        # for setup/config problems, not transient search failures.
        try:
            results = await run_search(query, db)
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Search failed: {e}")
            sys.exit(1)

        if not results:
            print(f"No results found for '{query}'.")
            sys.exit(0)

        search_id = db.save_search(query, results)

        print(f"\nResults for: {query}  (saved as search #{search_id})\n{'-' * 40}")
        for i, result in enumerate(results, start=1):
            print(f"{i}. {result}")

    finally:
        # Guarantees the connection is closed on every exit path —
        # normal return, sys.exit(), or an uncaught exception bubbling up.
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
