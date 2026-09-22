import logging

import numpy as np

from .database import Database
from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    db: Database,
    provider: EmbeddingProvider,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Embeds the query and returns the top_k most similar stored chunks,
    ranked by cosine similarity. Only chunks embedded by the CURRENT
    provider are considered — comparing vectors from different embedding
    models is meaningless, so a mismatch is filtered out, not scored."""
    all_chunks = db.get_all_chunks()

    matching = [c for c in all_chunks if c["embedding_model"] == provider.name]
    skipped = len(all_chunks) - len(matching)
    if skipped:
        logger.warning(
            f"Skipped {skipped} chunk(s) from a different embedding model "
            f"than the current provider ({provider.name})."
        )

    if not matching:
        return []

    query_embedding = provider.embed([query], input_type="query")[0]

    matrix = np.stack([c["embedding"] for c in matching])
    row_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    similarities = (matrix @ query_embedding) / (row_norms * query_norm + 1e-10)

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [{**matching[i], "similarity": float(similarities[i])} for i in top_indices]