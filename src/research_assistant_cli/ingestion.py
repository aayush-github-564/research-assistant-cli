import logging

from .content import chunk_text, fetch_page_text
from .database import Database
from .embeddings import EmbeddingProvider
from .models import SearchResult

logger = logging.getLogger(__name__)


def embed_and_store_results(
    db: Database,
    provider: EmbeddingProvider,
    search_id: int,
    results: list[SearchResult],
) -> None:
    """Fetches full-page text per result (falling back to the search
    snippet if fetching fails or the page yields nothing extractable),
    chunks it, embeds the chunks, and persists them. One bad result
    doesn't abort the rest — mirrors search_multiple's per-provider
    failure isolation, at the per-URL level instead."""
    result_ids = db.get_result_ids_for_search(search_id)

    for result_id, result in zip(result_ids, results):
        try:
            text = fetch_page_text(result.url)
        except Exception as e:
            logger.warning(f"Fetch failed permanently for {result.url}: {e}")
            text = None

        if text is None:
            text = result.snippet  # still embed *something* rather than skip

        chunks = chunk_text(text)
        if not chunks:
            logger.warning(f"No chunks produced for {result.url} — skipping.")
            continue

        try:
            embeddings = provider.embed(chunks, input_type="document")
        except Exception as e:
            logger.warning(f"Embedding failed for {result.url}: {e}")
            continue

        db.save_chunks(result_id, chunks, embeddings, provider.name)