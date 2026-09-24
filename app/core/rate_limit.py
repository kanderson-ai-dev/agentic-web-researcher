"""Simple in-process sliding-window rate limiter.

Single-instance by design (the service ships as one container). Limits are
enforced per client key (IP) per scope (route family).
"""

import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window counter: ``check`` returns False when the limit is hit."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, scope: str, key: str, *, limit: int, window_seconds: float) -> bool:
        now = time.monotonic()
        bucket = self._hits[f"{scope}:{key}"]
        while bucket and bucket[0] <= now - window_seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()


rate_limiter = RateLimiter()
