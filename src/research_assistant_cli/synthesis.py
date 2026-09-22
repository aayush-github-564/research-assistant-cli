import os

import anthropic

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY "
    "the provided context chunks. Cite sources by their URL. If the "
    "context doesn't contain enough information to answer, say so plainly "
    "rather than guessing."
)


def synthesize_answer(query: str, chunks: list[dict]) -> str:
    """Builds a context block from retrieved chunks and asks Claude to
    answer, citing sources."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Missing ANTHROPIC_API_KEY. Check your .env file.")

    client = anthropic.Anthropic(api_key=api_key)
    context = "\n\n".join(
        f"[Source: {c['title']} — {c['url']}]\n{c['chunk_text']}" for c in chunks
    )

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return "".join(block.text for block in response.content if block.type == "text")