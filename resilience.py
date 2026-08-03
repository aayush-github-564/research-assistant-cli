import functools
import logging
import time

logger = logging.getLogger(__name__)


def retry_with_backoff(max_attempts=3, base_delay=1, backoff_factor=2, retry_on=(Exception,)):
    """Retries a function on failure, waiting longer between each attempt.

    retry_on: tuple of exception types worth retrying (e.g. network errors).
              Anything else (like a bad API key) fails immediately — retrying
              a config error just wastes time.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {attempt} attempts: {e}")
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed ({e}), "
                        f"retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator