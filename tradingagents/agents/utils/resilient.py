"""Resilient tool call wrapper with retry and fault tolerance.

Wraps any data-fetching function with:
- Automatic retry (3 attempts with exponential backoff)
- Exception → error string conversion (never crashes)
- Logging for debugging
"""

import time
import logging

logger = logging.getLogger(__name__)

# Exceptions that are worth retrying (transient network/API issues)
_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,            # covers socket errors
)

# Try to include requests exceptions if available
try:
    import requests.exceptions
    _RETRYABLE = _RETRYABLE + (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
except ImportError:
    pass


def resilient_call(func, *args, tool_name: str = "", max_retries: int = 3, **kwargs) -> str:
    """Call `func(*args, **kwargs)` with retry and error handling.

    Args:
        func: The function to call (e.g., route_to_vendor)
        *args: Positional arguments to pass through
        tool_name: Name for logging/error messages
        max_retries: Number of attempts (default 3)
        **kwargs: Keyword arguments to pass through

    Returns:
        str: Either the successful result or an "ERROR: ..." message.
              Never raises an exception.
    """
    label = tool_name or getattr(func, "__name__", "unknown")
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            result = func(*args, **kwargs)

            # Guard against None or empty results
            if result is None:
                return f"ERROR [{label}]: No data returned. The data may not be available for the given parameters."
            if isinstance(result, str) and not result.strip():
                return f"ERROR [{label}]: Empty response received. Try different parameters or skip this data source."

            return result

        except _RETRYABLE as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(
                    f"[{label}] Attempt {attempt}/{max_retries} failed (retryable): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"[{label}] All {max_retries} attempts failed: {e}")

        except Exception as e:
            # Non-retryable error — return immediately, don't retry
            logger.error(f"[{label}] Non-retryable error: {type(e).__name__}: {e}")
            return (
                f"ERROR [{label}]: {type(e).__name__}: {e}. "
                f"This data source is unavailable. Continue analysis without it."
            )

    # All retries exhausted for a retryable error
    return (
        f"ERROR [{label}]: Failed after {max_retries} attempts. "
        f"Last error: {last_error}. "
        f"This data source is temporarily unavailable. Continue analysis without it."
    )
