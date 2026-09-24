"""Guardrails for the research pipeline.

Three layers:

1. ``screen_topic`` — input guardrail run before any LLM call or external
   request (guardrail-first: don't spend tokens or bandwidth on malicious
   input).
2. ``sanitize_scraped_content`` — treats fetched web content as *untrusted
   data*: embedded instruction-injection lines are removed and the text is
   truncated. Sanitized text is the only form allowed downstream.
3. ``verify_citations`` — output guardrail requiring every citation's quote to
   appear verbatim in the claimed source's extracted text. Unsupported
   citations are dropped, never silently kept.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

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


def verify_citations(
    citations: Sequence[Citation], sources: Sequence[Source]
) -> tuple[list[Citation], list[Citation]]:
    """Split citations into (supported, unsupported).

    A citation is supported only if ``source_id`` exists and its ``quote``
    appears verbatim (modulo whitespace/case) in that source's extracted text.
    """
    by_id = {s.id: s for s in sources}
    kept: list[Citation] = []
    dropped: list[Citation] = []
    for citation in citations:
        source = by_id.get(citation.source_id)
        if source is None:
            dropped.append(citation)
            continue
        if _normalized(citation.quote) and _normalized(citation.quote) in _normalized(
            source.extracted_text
        ):
            kept.append(citation)
        else:
            dropped.append(citation)
    return kept, dropped
