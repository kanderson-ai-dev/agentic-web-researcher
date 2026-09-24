"""Prompt templates for the LLM-driven agent roles: planner, critic, writer.

Prompts live here as a first-class, reviewable artifact rather than inline in
the LLM client — a prompt change is a behavioral change to an agent role, so
it should be visible in diffs and measurable through the EDD harness
(``evaluation/run_eval.py``).

Design decisions baked into these prompts:

- **Structured outputs** (``with_structured_output``) carry the contract;
  prompts shape behavior, schemas enforce shape — no free-text parsing.
- **Bounded evidence**: each source is truncated before interpolation so
  context size, latency, and cost stay predictable.
- **Citations must be verbatim**: the writer is instructed to copy short
  quotes directly from the cited source's extracted text and to reference
  the provided ``[source_id]`` labels. The output guardrail then verifies
  every quote mechanically — fabricated citations are dropped, never kept.
- **No untrusted instructions**: scraped text is presented strictly as
  ``Evidence`` data inside the human turn; the sanitization guardrail has
  already stripped injection-style lines from it (OWASP LLM01).
"""

from collections.abc import Sequence

from app.core.schemas import Source, SubQuestion

# ---------------------------------------------------------------------------
# System prompts (one per agent role)
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a research planner. Decompose the topic into focused, "
    "non-overlapping sub-questions that together cover it. Return at "
    "most {max_questions} questions, in {language}."
)

_CRITIC_SYSTEM = (
    "You are a research critic. Given the planned sub-questions and "
    "the evidence collected so far, grade coverage and whether the "
    "evidence could support citations. List concrete missing aspects "
    "only when important gaps remain; do not ask for more research "
    "on marginal gaps."
)

_WRITER_SYSTEM = (
    "You are a research writer. Write a markdown report answering "
    "the topic in {language}. Every factual claim must be backed "
    "by a citation: claim text, the source id it came from, and a "
    "short verbatim quote from that source's extracted text. Never "
    "invent facts or citations."
)

# Evidence excerpt caps per role — the critic needs breadth (skim), the
# writer needs enough verbatim material to quote accurately.
CRITIC_EVIDENCE_CHARS = 1500
WRITER_EVIDENCE_CHARS = 2000


def planner_system(*, max_questions: int, language: str) -> str:
    """System prompt for the planner agent."""
    return _PLANNER_SYSTEM.format(max_questions=max_questions, language=language)


def critic_system() -> str:
    """System prompt for the critic agent."""
    return _CRITIC_SYSTEM


def writer_system(*, language: str) -> str:
    """System prompt for the writer agent."""
    return _WRITER_SYSTEM.format(language=language)


# ---------------------------------------------------------------------------
# Human-message builders (shared by critic and writer)
# ---------------------------------------------------------------------------


def format_evidence(sources: Sequence[Source], *, per_source_chars: int) -> str:
    """Render collected sources as a labeled evidence block.

    Each source is labeled ``[source_id]`` so the model can cite by id; text
    is truncated to ``per_source_chars`` to bound context size.
    """
    return "\n\n".join(
        f"[{s.id}] {s.title or s.url} ({s.url})\n{s.extracted_text[:per_source_chars]}"
        for s in sources
    )


def format_sub_questions(sub_questions: Sequence[SubQuestion]) -> str:
    """Render the planned sub-questions as a bullet list."""
    return "\n".join(f"- {sq.question}" for sq in sub_questions)


def research_prompt(
    *,
    topic: str,
    sub_questions: Sequence[SubQuestion],
    evidence: str,
) -> str:
    """Human message shared by the critic and writer turns."""
    return (
        f"Topic: {topic}\n\nSub-questions:\n{format_sub_questions(sub_questions)}"
        f"\n\nEvidence:\n{evidence or '(none)'}"
    )
