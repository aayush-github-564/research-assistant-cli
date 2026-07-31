import sys

from database import Database
from providers import TavilySearchProvider


def main():
    db = Database()

    if len(sys.argv) >= 2 and sys.argv[1] == "--history":
        recent = db.get_recent_searches(limit=5)
        if not recent:
            print("No past searches yet.")
        for s in recent:
            print(f"#{s['id']}  {s['query']}  ({s['created_at']})")
        db.close()
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--show":
        search_id = int(sys.argv[2])
        results = db.get_results_for_search(search_id)
        if not results:
            print(f"No results found for search #{search_id}.")
        for i, r in enumerate(results, start=1):
            print(f"{i}. {r}")
        db.close()
        return

    if len(sys.argv) >= 3 and sys.argv[1] == "--find":
        keyword = sys.argv[2]
        matches = db.search_history(keyword)
        if not matches:
            print(f"No past searches matching '{keyword}'.")
        for s in matches:
            print(f"#{s['id']}  {s['query']}  ({s['created_at']})")
        db.close()
        return

    if len(sys.argv) < 2:
        print('Usage: uv run python main.py "your search query"')
        sys.exit(1)

    query = sys.argv[1]

    provider = TavilySearchProvider()  # swap this to change provider

    try:
        results = provider.search(query, max_results=5)
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

    if not results:
        print(f"No      results found for '{query}'.")
        sys.exit(0)

    search_id = db.save_search(query, results)

    print(f"\nResults for: {query}  (saved as search #{search_id})\n{'-' * 40}")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result}")

    db.close()


if __name__ == "__main__":
    main()
