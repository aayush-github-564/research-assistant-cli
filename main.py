import os
import sys
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print('Usage: uv run python main.py "your search query"')
        sys.exit(1)

    query = sys.argv[1]

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("Missing TAVILY_API_KEY. Check your .env file.")
        sys.exit(1)

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(query, max_results=5)
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

    results = response.get("results", [])

    if not results:
        print(f"No results found for '{query}'.")
        sys.exit(0)

    print(f"\nResults for: {query}\n{'-' * 40}")
    for i, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        snippet = result.get("content", "No description")

        print(f"{i}. {title}")
        print(f"   {url}")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()
