import asyncio
import signal
import sys
import threading

import requests

from .database import Database
from .logger import setup_logging
from .providers import DuckDuckGoSearchProvider, TavilySearchProvider, search_multiple
from .embeddings import get_embedding_provider
from .ingestion import embed_and_store_results
from .embeddings import get_embedding_provider
from .retrieval import retrieve_relevant_chunks
from .synthesis import synthesize_answer
from dotenv import load_dotenv

load_dotenv()

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

        if len(sys.argv) >= 3 and sys.argv[1] == "--ask":
            question = sys.argv[2]

            try:
                provider = get_embedding_provider()
            except ValueError as e:
                print(f"Can't answer — {e}")
                return

            # Always searches first — same config-error-vs-network-failure
            # split as the main search path above — so --ask can answer
            # questions on topics never searched before, not just what's
            # already sitting in the corpus.
            try:
                results = await run_search(question, db)
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning(f"Fresh search for --ask failed, falling back to existing corpus: {e}")
                results = []

            if results:
                search_id = db.save_search(question, results)
                print(f"Searched for: {question}  (saved as search #{search_id})\n")
                try:
                    embed_and_store_results(db, provider, search_id, results)
                except Exception as e:
                    logger.warning(f"Embedding failed — results saved, but not indexed: {e}")

            chunks = retrieve_relevant_chunks(db, provider, question)
            if not chunks:
                print("Nothing relevant found — even after a fresh search.")
                return

            try:
                answer = synthesize_answer(question, chunks)
            except ValueError as e:
                print(f"Can't answer — {e}")
                return

            print(f"\n{answer}\n\nSources:")
            for c in chunks:
                print(f"- {c['title']} ({c['url']})")
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

        try:
            provider = get_embedding_provider()
            embed_and_store_results(db, provider, search_id, results)
        except Exception as e:
            logger.warning(f"Embedding failed — results saved, but not indexed for --ask: {e}")

    finally:
        # Guarantees the connection is closed on every exit path —
        # normal return, sys.exit(), or an uncaught exception bubbling up.
        db.close()


def cli():
    """Sync entry point for the installed console script.
    console-script entry points call a plain sync callable with no args —
    they don't know how to await a coroutine, so this wraps the real
    async main() in asyncio.run(), exactly like the __main__ block did."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
