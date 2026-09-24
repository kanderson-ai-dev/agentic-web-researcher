"""Per-job LLM token usage tracking and cost estimation.

The active tracker lives in a :class:`contextvars.ContextVar` so each
``asyncio.Task`` running a research job gets isolated accounting — the LLM
client records usage without needing the tracker threaded through the graph
state (which must stay checkpoint-serializable).
"""

import contextvars
from dataclasses import dataclass, field

from app.core.metrics import LLM_TOKENS

# USD per 1M tokens: (input, output). Keep in sync with provider pricing.
_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
_DEFAULT_PRICE = (0.15, 0.60)


@dataclass
class UsageRecord:
    """Token usage of a single LLM call, tagged by agent role."""

    role: str
    input_tokens: int
    output_tokens: int


@dataclass
class UsageTracker:
    """Accumulates usage records for one job run."""

    records: list[UsageRecord] = field(default_factory=list)

    def add(self, role: str, input_tokens: int, output_tokens: int) -> None:
        self.records.append(
            UsageRecord(role=role, input_tokens=input_tokens, output_tokens=output_tokens)
        )

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def cost_usd(self, model: str) -> float:
        """Estimated USD cost for the accumulated usage on ``model``."""
        input_price, output_price = _PRICING_USD_PER_1M.get(model, _DEFAULT_PRICE)
        return round(
            self.input_tokens * input_price / 1_000_000
            + self.output_tokens * output_price / 1_000_000,
            6,
        )


_current_tracker: contextvars.ContextVar[UsageTracker | None] = contextvars.ContextVar(
    "usage_tracker", default=None
)


def set_tracker(tracker: UsageTracker) -> contextvars.Token[UsageTracker | None]:
    """Activate ``tracker`` for the current context; returns the reset token."""
    return _current_tracker.set(tracker)


def reset_tracker(token: contextvars.Token[UsageTracker | None]) -> None:
    """Restore the previous tracker (call in ``finally``)."""
    _current_tracker.reset(token)


def record_usage(role: str, input_tokens: int, output_tokens: int) -> None:
    """Record usage on the active tracker; no-op when none is set."""
    tracker = _current_tracker.get()
    if tracker is None:
        return
    tracker.add(role, input_tokens, output_tokens)
    LLM_TOKENS.labels(role=role, kind="input").inc(input_tokens)
    LLM_TOKENS.labels(role=role, kind="output").inc(output_tokens)
