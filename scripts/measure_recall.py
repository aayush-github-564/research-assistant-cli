import threading
from dotenv import load_dotenv

from research_assistant_cli.database import Database
from research_assistant_cli.embeddings import get_embedding_provider
from research_assistant_cli.retrieval import retrieve_relevant_chunks

load_dotenv()


# Question -> one designated relevant source URL.
# Build this from sources that actually exist in your indexed corpus.
EVAL_SET = [
    (
        "How does retrieval-augmented generation combine retrieved external information with LLM generation?",
        "https://www.ibm.com/think/topics/retrieval-augmented-generation",
    ),
    (
        "Why are vector databases useful for storing and searching high-dimensional embeddings?",
        "https://www.pinecone.io/learn/vector-database",
    ),
    (
        "How do embedding models convert text into representations suitable for semantic search?",
        "https://aws.amazon.com/what-is/embeddings-in-machine-learning",
    ),
    (
        "What is the purpose of a reranking model in a retrieval pipeline?",
        "https://www.pinecone.io/learn/series/rag/rerankers",
    ),
    (
        "How does the Model Context Protocol standardize connections between AI applications and external tools or data?",
        "https://www.anthropic.com/news/model-context-protocol",
    ),
    (
        "What distinguishes an AI agent from a conventional LLM application?",
        "https://www.ibm.com/think/topics/ai-agents",
    ),
    (
        "How does an API gateway act as an intermediary between clients and backend services?",
        "https://learn.microsoft.com/en-us/dotnet/architecture/microservices/architect-microservice-container-applications/direct-client-to-microservice-communication-versus-the-api-gateway-pattern",
    ),
    (
        "Why do APIs use rate limiting, and what problems does it help prevent?",
        "https://blog.postman.com/what-is-api-rate-limiting",
    ),
    (
        "How does JWT-based authentication allow an API to verify a user's identity?",
        "https://jwt.io/introduction",
    ),
    (
        "What are the main stages and practices involved in a CI/CD pipeline?",
        "https://about.gitlab.com/topics/ci-cd",
    ),
    (
        "How does DNS resolution translate a domain name into the information needed to connect to a server?",
        "https://www.cloudflare.com/learning/dns/what-is-dns",
    ),
    (
        "How does GraphQL allow clients to request the specific data they need from an API?",
        "https://graphql.org",
    ),
    (
        "How does HNSW enable approximate nearest-neighbor search over vectors?",
        "https://www.pinecone.io/learn/series/faiss/hnsw",
    ),
    (
        "What properties does a database transaction provide and why are transactions important?",
        "https://www.postgresql.org/docs/current/tutorial-transactions.html",
    ),
    (
        "What makes SQLite different from a traditional client-server database?",
        "https://www.sqlite.org/about.html",
    ),
    (
        "Why are locks needed to make shared state thread-safe in Python?",
        "https://realpython.com/python-thread-lock",
    ),
    (
        "How does Python's asyncio model handle asynchronous I/O without blocking on each operation?",
        "https://realpython.com/async-io-python",
    ),
    (
        "What does the CAP theorem say about consistency, availability, and partition tolerance in distributed systems?",
        "https://www.ibm.com/think/topics/cap-theorem",
    ),
    (
        "How do unit tests differ from integration tests in terms of what they validate?",
        "https://www.browserstack.com/guide/unit-testing-vs-integration-testing",
    ),
    (
        "What characteristics distinguish a distributed system from a single-machine application?",
        "https://www.ibm.com/think/topics/distributed-systems",
    ),
]


def normalize_url(url: str) -> str:
    """Normalize URLs so trailing slashes don't create false mismatches."""
    return url.strip().rstrip("/")


def unique_sources(results: list[dict]) -> list[dict]:
    """
    Collapse multiple retrieved chunks from the same source URL.

    Retrieval still happens at the chunk level, but evaluation is performed
    at the source level so multiple chunks from one page don't consume all
    top-5 positions.
    """
    seen = set()
    unique = []

    for result in results:
        url = normalize_url(result["url"])

        if url not in seen:
            seen.add(url)
            unique.append(result)

    return unique


def main():
    db = Database(threading.Event())
    provider = get_embedding_provider()

    hits = 0

    try:
        for question, expected_url in EVAL_SET:
            # Retrieve more chunks first so we have enough candidates after
            # collapsing duplicate sources.
            results = retrieve_relevant_chunks(
                db,
                provider,
                question,
                top_k=20,
            )

            # Convert chunk-level results into unique source-level results.
            sources = unique_sources(results)

            # Evaluate only the top 5 unique sources.
            top_sources = sources[:5]

            expected = normalize_url(expected_url)
            retrieved_urls = [
                normalize_url(result["url"])
                for result in top_sources
            ]

            if expected in retrieved_urls:
                rank = retrieved_urls.index(expected) + 1
                hits += 1

                print(f"HIT  @ {rank}  {question}")

            else:
                print(f"MISS     {question}")
                print("         Retrieved:")

                for i, result in enumerate(top_sources, start=1):
                    print(f"         {i}. {normalize_url(result['url'])}")

        recall = hits / len(EVAL_SET) * 100

        print(
            f"\nSource Recall@5: "
            f"{hits}/{len(EVAL_SET)} ({recall:.1f}%)"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()