import sys

from providers import TavilySearchProvider


def main():
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
        print(f"No results found for '{query}'.")
        sys.exit(0)

    print(f"\nResults for: {query}\n{'-' * 40}")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result}")


if __name__ == "__main__":
    main()
