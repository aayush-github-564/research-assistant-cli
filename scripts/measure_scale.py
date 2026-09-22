import asyncio
import threading
import time

from research_assistant_cli.database import Database
from research_assistant_cli.providers import DuckDuckGoSearchProvider, TavilySearchProvider, search_multiple
from research_assistant_cli.embeddings import get_embedding_provider
from research_assistant_cli.ingestion import embed_and_store_results
from research_assistant_cli.retrieval import retrieve_relevant_chunks

from dotenv import load_dotenv

load_dotenv()

TOPICS = [
    # RAG / LLM / embeddings — the actual subject of this project
    "what is retrieval augmented generation",
    "what is a vector database",
    "what is cosine similarity",
    "what is an embedding model",
    "what is semantic search",
    "what is chunking strategy in RAG",
    "what is a reranking model",
    "what is prompt engineering",
    "what is fine-tuning a language model",
    "what is few-shot prompting",
    "what is the model context protocol",
    "what is an AI agent",
    "what is agentic RAG",
    "what is LangChain",
    "what is hallucination in large language models",
    "what is context window in LLMs",
    "what is token limit in LLM APIs",
    "what is a system prompt",
    "what is retrieval evaluation recall at k",
    "what is HNSW vector indexing",

    # Backend / API design
    "what is REST API design",
    "what is idempotency in APIs",
    "what is API rate limiting",
    "what is JWT authentication",
    "what is OAuth 2.0",
    "what is API versioning",
    "what is a webhook",
    "what is GraphQL",
    "what is gRPC",
    "what is API gateway pattern",
    "what is multi-tenant architecture",
    "what is dependency injection",

    # Databases
    "what is database indexing",
    "what is database normalization",
    "what is a database transaction",
    "what is ACID compliance",
    "what is database sharding",
    "what is a database connection pool",
    "what is redis caching",
    "what is SQLite",
    "what is PostgreSQL",
    "what is an ORM",
    "what is a database migration",
    "what is optimistic locking",

    # Systems / concurrency / networking
    "what is thread safety in python",
    "what is asyncio in python",
    "what is a race condition",
    "what is exponential backoff",
    "what is a circuit breaker pattern",
    "what is TCP vs UDP",
    "what is DNS resolution",
    "what is a load balancer",
    "what is horizontal scaling",
    "what is a message queue",
    "what is event-driven architecture",
    "what is a distributed system",

    # DevOps / testing / tooling
    "what is CI/CD",
    "what is docker containerization",
    "what is kubernetes",
    "what is infrastructure as code",
    "what is unit testing vs integration testing",
    "what is test coverage",
    "what is mocking in unit tests",
    "what is git rebase vs merge",
    "what is a monorepo",

    # CS fundamentals / interview prep
    "what is big o notation",
    "what is a hash table",
    "what is a binary search tree",
    "what is dynamic programming",
    "what is the CAP theorem",
    "what is a design pattern",
    "what is SOLID principles",
    "what is garbage collection",
    "what is a memory leak",
]


async def main():
    db = Database(threading.Event())
    provider = get_embedding_provider()
    ddg = DuckDuckGoSearchProvider(db)
    tavily = TavilySearchProvider(db)

    for i, topic in enumerate(TOPICS, start=1):
        results = await search_multiple([ddg, tavily], topic)
        if results:
            search_id = db.save_search(topic, results)
            embed_and_store_results(db, provider, search_id, results)
        print(f"[{i}/{len(TOPICS)}] {topic}")

    chunk_count = len(db.get_all_chunks())
    print(f"\nTotal chunks indexed: {chunk_count}")

    timings = []
    for q in ["what is retrieval augmented generation", "how do vector databases work"]:
        start = time.perf_counter()
        retrieve_relevant_chunks(db, provider, q)
        timings.append((time.perf_counter() - start) * 1000)

    print(f"Average retrieval latency: {sum(timings) / len(timings):.1f}ms across {chunk_count} chunks")
    db.close()


asyncio.run(main())