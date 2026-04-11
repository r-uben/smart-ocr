"""Retry with exponential backoff for transient HTTP errors.

Used by all HTTP-based engines (Gemini API, DeepSeek vLLM, etc.) to handle
transient failures: 429 rate limits, 500/502/503 server errors, network blips.
"""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that are safe to retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Exception types that indicate transient network issues
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: float = 0.5  # random jitter factor (0-1)
    retry_on_status: frozenset[int] = frozenset(RETRYABLE_STATUS_CODES)


def retry_on_transient(
    fn: Callable[[], T],
    config: RetryConfig | None = None,
    label: str = "",
) -> T:
    """Execute fn with retry on transient failures.

    Args:
        fn: Callable that returns a result. Should raise or return an
            httpx.Response for status-code checking.
        config: Retry configuration. Uses defaults if None.
        label: Label for log messages (e.g. "gemini-api page 3").

    Returns:
        The result of fn() on success.

    Raises:
        The last exception if all retries exhausted.
    """
    cfg = config or RetryConfig()
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            result = fn()

            # If result is an httpx.Response, check status
            if isinstance(result, httpx.Response) and result.status_code in cfg.retry_on_status:
                if attempt < cfg.max_retries:
                    delay = _backoff_delay(attempt, cfg)
                    # Respect Retry-After header if present
                    retry_after = result.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        f"[{label}] HTTP {result.status_code}, "
                        f"retry {attempt + 1}/{cfg.max_retries} in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

            return result

        except RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            if attempt < cfg.max_retries:
                delay = _backoff_delay(attempt, cfg)
                logger.warning(
                    f"[{label}] {type(exc).__name__}: {exc}, "
                    f"retry {attempt + 1}/{cfg.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
            else:
                raise

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exhausted without result")


def _backoff_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = config.base_delay * (2 ** attempt)
    delay = min(delay, config.max_delay)
    # Add random jitter to avoid thundering herd
    jitter = random.uniform(0, config.jitter * delay)
    return delay + jitter
