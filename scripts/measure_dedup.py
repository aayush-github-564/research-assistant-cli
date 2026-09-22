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
    "retrieval augmented generation",
    "python asyncio best practices",
    # ... fill to 20-30 real queries
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


async def main():
    db = Database(threading.Event(), db_path=Path("measure_temp.db"))
    ddg = DuckDuckGoSearchProvider(db)
    tavily = TavilySearchProvider(db)

    raw_total = 0
    deduped_total = 0

    for query in QUERIES:
        raw_total += len(ddg.search(query, max_results=5)) + len(tavily.search(query, max_results=5))
        merged = await search_multiple([ddg, tavily], query)  # the real, deployed function
        deduped_total += len(merged)

    print(f"Raw: {raw_total}  Deduped: {deduped_total}")
    print(f"Dedup rate: {(raw_total - deduped_total) / raw_total * 100:.1f}%")
    db.close()


asyncio.run(main())