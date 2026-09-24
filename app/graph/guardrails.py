"""Guardrails for the research pipeline.

Three layers:

1. ``screen_topic`` — input guardrail run before any LLM call or external
   request (guardrail-first: don't spend tokens or bandwidth on malicious
   input).
2. ``sanitize_scraped_content`` — treats fetched web content as *untrusted
   data*: embedded instruction-injection lines are removed and the text is
   truncated. Sanitized text is the only form allowed downstream.
3. ``verify_citations`` — output guardrail requiring every citation's quote to
   appear in the claimed source's extracted text (verbatim modulo
   punctuation/case, or high sliding-window similarity). Unsupported
   citations are dropped, never silently kept.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.core.schemas import Citation, Source

MAX_TOPIC_CHARS = 2_000
MAX_SOURCE_CHARS = 20_000

_INJECTION_PATTERNS = [
    r"ignore\s+(?:(?:all|any|the|previous|prior|above|earlier|your|following)\s+)*instructions",
    r"disregard\s+(?:(?:all|any|the|previous|prior|above|your|following)\s+)*instructions",
    r"forget\s+(?:everything|all|your\s+instructions|previous\s+instructions)",
    r"(?:reveal|show|print|leak|repeat)\s+(?:your\s+)?(?:system\s+prompt|initial\s+prompt|instructions)",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbroken|unrestricted)",
    r"act\s+as\s+(?:a\s+)?(?:dan|jailbroken|unrestricted\s+ai)",
    r"system\s*:\s*you\s+are",
    r"<\s*/?\s*system\s*>",
    r"\[/?INST\]",
    r"<\|im_(?:start|end)\|>",
    r"<\|(?:system|user|assistant)\|>",
    r"override\s+(?:your\s+)?(?:safety|guardrails|restrictions|programming)",
    r"new\s+instructions\s*:",
    r"do\s+not\s+follow\s+(?:your\s+)?(?:previous|prior)\s+(?:instructions|rules)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_HARMFUL_PATTERNS = [
    r"how\s+to\s+(?:make|build|create)\s+(?:a\s+|an\s+)?(?:bomb|explosive|weapon)",
    r"how\s+to\s+hack\s+(?:into\s+)?",
    r"(?:buy|sell|produce)\s+(?:illegal\s+)?drugs",
    r"write\s+(?:a\s+|an\s+)?(?:phishing|malware|ransomware|keylogger)",
]
_HARMFUL_RE = re.compile("|".join(_HARMFUL_PATTERNS), re.IGNORECASE)

_INVISIBLE_RE = re.compile(r"[​‌‍﻿‏‪-‮⁠]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of the input guardrail."""

    allowed: bool
    sanitized_topic: str
    reason: str | None = None


def _clean_text(text: str) -> str:
    """Strip control/invisible characters and collapse whitespace."""
    text = _CONTROL_RE.sub("", _INVISIBLE_RE.sub(" ", text))
    return _WS_RE.sub(" ", text).strip()


def screen_topic(topic: str, *, max_chars: int = MAX_TOPIC_CHARS) -> GuardrailResult:
    """Validate a user-supplied research topic before it reaches the planner.

    Returns a blocked result for prompt-injection payloads and clearly harmful
    requests. The sanitized topic (control chars removed, whitespace collapsed,
    length capped) is what the graph may use — never the raw input.
    """
    sanitized = _clean_text(topic)[:max_chars]
    if not sanitized:
        return GuardrailResult(allowed=False, sanitized_topic="", reason="empty topic")
    if _INJECTION_RE.search(sanitized):
        return GuardrailResult(
            allowed=False, sanitized_topic=sanitized, reason="prompt injection detected"
        )
    if _HARMFUL_RE.search(sanitized):
        return GuardrailResult(
            allowed=False, sanitized_topic=sanitized, reason="harmful request detected"
        )
    return GuardrailResult(allowed=True, sanitized_topic=sanitized)


def sanitize_scraped_content(
    text: str, *, max_chars: int = MAX_SOURCE_CHARS
) -> str:
    """Neutralize instruction-injection attempts inside fetched web content.

    Lines matching known injection patterns are dropped outright; the rest is
    de-invisibilized and truncated. The result is still treated as data — it is
    never interpolated into instruction prompts.
    """
    kept_lines = [
        line for line in text.splitlines() if not _INJECTION_RE.search(line)
    ]
    return _clean_text("\n".join(kept_lines))[:max_chars]


def _normalized(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _squashed(text: str) -> str:
    """Alnum-only lowercase form — ignores punctuation/typography variance."""
    return re.sub(r"[^a-z0-9]+", "", _normalized(text))


_SIMILARITY_THRESHOLD = 0.85


def _best_window_ratio(quote: str, text: str) -> float:
    """Highest similarity of ``quote`` against same-length windows of ``text``."""
    n = len(quote)
    if n == 0 or not text:
        return 0.0
    matcher = SequenceMatcher()
    matcher.set_seq2(quote)
    best = 0.0
    step = max(1, n // 4)
    for i in range(0, max(1, len(text) - n + 1), step):
        matcher.set_seq1(text[i : i + n])
        if matcher.quick_ratio() >= best:
            best = max(best, matcher.ratio())
    return best


def _quote_supported(quote: str, extracted_text: str) -> bool:
    norm_quote, norm_text = _normalized(quote), _normalized(extracted_text)
    if norm_quote and norm_quote in norm_text:
        return True
    squash_quote = _squashed(quote)
    if len(squash_quote) >= 20 and squash_quote in _squashed(extracted_text):
        return True
    return _best_window_ratio(norm_quote, norm_text) >= _SIMILARITY_THRESHOLD


def verify_citations(
    citations: Sequence[Citation], sources: Sequence[Source]
) -> tuple[list[Citation], list[Citation]]:
    """Split citations into (supported, unsupported).

    A citation is supported only if ``source_id`` exists and its ``quote``
    appears in that source's extracted text. Matching is verbatim modulo
    whitespace/case, punctuation-insensitive, and finally tolerant to light
    paraphrase (sliding-window similarity >= 0.85) — the quote must still
    demonstrably come from the cited source.
    """
    by_id = {s.id: s for s in sources}
    kept: list[Citation] = []
    dropped: list[Citation] = []
    for citation in citations:
        source = by_id.get(citation.source_id)
        if source is None or not _quote_supported(
            citation.quote, source.extracted_text
        ):
            dropped.append(citation)
        else:
            kept.append(citation)
    return kept, dropped
