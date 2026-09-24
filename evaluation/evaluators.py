"""Deterministic evaluators for the research pipeline (EDD).

All metrics are pure functions over the final graph output — they run offline,
are unit-tested independently, and gate CI the same way unit tests do.

- ``citation_support_rate`` — verified citations / total produced.
- ``topic_coverage`` — expected aspects present in the final report.
- ``source_precision`` — sources containing at least one relevance term.
"""

from collections.abc import Sequence

from app.core.schemas import Source


def citation_support_rate(kept: int, dropped: int) -> float:
    """Fraction of produced citations that survived output verification."""
    total = kept + dropped
    return kept / total if total else 0.0


def topic_coverage(report: str | None, expected_aspects: Sequence[str]) -> float:
    """Fraction of expected aspects mentioned (case-insensitive) in the report."""
    if not expected_aspects:
        return 0.0
    text = (report or "").lower()
    covered = sum(1 for aspect in expected_aspects if aspect.lower() in text)
    return covered / len(expected_aspects)


def source_precision(
    sources: Sequence[Source], relevance_terms: Sequence[str]
) -> float:
    """Fraction of sources containing at least one relevance term."""
    if not sources:
        return 0.0
    terms = [t.lower() for t in relevance_terms]
    relevant = sum(
        1
        for source in sources
        if any(term in source.extracted_text.lower() for term in terms)
    )
    return relevant / len(sources)
