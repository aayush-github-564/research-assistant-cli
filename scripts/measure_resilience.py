import asyncio
import threading
from pathlib import Path

from research_assistant_cli.database import Database
from research_assistant_cli.providers import (
    DuckDuckGoSearchProvider,
    TavilySearchProvider,
    search_multiple,
)
from dotenv import load_dotenv

load_dotenv()

QUERIES = [
    "what is retrieval augmented generation",
    "what is python asyncio best practices",
    "what is vector databases",
    "what is cosine similarity",
    "what is a large language model",
    "what is prompt engineering",
    "what is fine-tuning a language model",
    "what is semantic search",
    "what is an embedding model",
    "what is a REST API",
    "what is thread safety in python",
    "what is a rate limiter",
    "what is redis caching",
    "what is postgresql indexing",
    "what is a message queue",
    "what is exponential backoff",
    "what is the model context protocol",
    "what is an AI agent",
    "what is chunking in NLP",
    "what is a knowledge base",
    "what is SQLite",
    "what is dependency injection",
    "what is CI/CD",
    "what is docker containerization",
    "what is JWT authentication",
    "what is multi-tenant architecture",
    "what is a load balancer",
    "what is horizontal scaling",
    "what is a webhook",
    "what is an ORM",
    "what is a microservice architecture",
    "what is a message broker",
    "what is a content delivery network",
    "what is a search engine optimization",
    "what is a web application firewall",
    "what is a load test",
    "what is a performance test",
    "what is a stress test",
] 


def always_fails(self, query, max_results=5):
    raise ConnectionError("simulated total provider outage")


async def main():
    db = Database(threading.Event(), db_path=Path("measure_temp.db"))
    ddg = DuckDuckGoSearchProvider(db)
    tavily = TavilySearchProvider(db)
    ddg.search = always_fails.__get__(ddg)  # DuckDuckGo is "down" for every query

    successes = 0
    for q in QUERIES:
        if await search_multiple([ddg, tavily], q):
            successes += 1

    print(f"{successes}/{len(QUERIES)} searches still returned results "
          f"({successes / len(QUERIES) * 100:.0f}%) despite one provider fully down")
    db.close()


asyncio.run(main())