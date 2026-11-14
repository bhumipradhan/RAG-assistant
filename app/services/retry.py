"""Tiny retry helper with exponential backoff. Free API tiers rate-limit (HTTP 429) often."""

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def retry_call[T](
    fn: Callable[[], T],
    *,
    should_retry: Callable[[Exception], bool],
    attempts: int = 4,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Call `fn`; on a retryable error wait base_delay * 2**n seconds and try again."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not should_retry(exc):
                raise
            delay = base_delay * 2**attempt
            logger.warning("Retryable error (%s); retrying in %.0fs", exc, delay)
            (sleep or time.sleep)(delay)  # resolved at call time so tests can patch it
    raise AssertionError("unreachable")  # pragma: no cover
