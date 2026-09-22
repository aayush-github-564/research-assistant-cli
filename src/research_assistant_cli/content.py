import logging

import requests
import trafilatura

from .resilience import retry_with_backoff

logger = logging.getLogger(__name__)


@retry_with_backoff(
    max_attempts=2,
    base_delay=1,
    retry_on=(requests.exceptions.RequestException,),
)
def fetch_page_text(url: str, timeout: int = 10) -> str | None:
    """Fetches a URL and extracts its main readable text.

    Returns None instead of raising when the page can't be fetched or
    has nothing extractable — a single bad URL (paywall, 404, JS-only
    page) shouldn't abort embedding the rest of a search's results.
    Network errors ARE retried (transient); "fetched fine but nothing
    extractable" is not (retrying won't change what's on the page).
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (research-assistant-cli)"},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Fetch failed for {url}: {e}")
        raise  # let retry_with_backoff handle transient retries

    text = trafilatura.extract(response.text)
    if not text:
        logger.warning(f"No extractable content at {url}")
        return None

    return text


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """Splits text into overlapping word-count chunks.

    Sizes are in words, not characters — words are the more meaningful
    unit for an embedding model. `overlap` re-includes the tail of one
    chunk at the start of the next, so context doesn't get severed
    right at a chunk boundary (e.g. a sentence cut exactly at word 200).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break

    return chunks