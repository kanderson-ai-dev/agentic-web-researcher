"""LLM abstraction for the planner / critic / writer agents.

Two implementations:

- :class:`OpenAILLMClient` — real calls through ``langchain-openai`` with
  structured outputs.
- :class:`StubLLMClient` — deterministic, offline double used by tests and by
  the app when no API key is configured (CI must run with zero secrets).
"""

from collections.abc import Sequence
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.core.config import Settings
from app.core.schemas import Citation, CriticVerdict, Source, SubQuestion
from app.services.usage import record_usage


class LLMClient(Protocol):
    """Interface the graph nodes depend on."""

    async def plan(
        self, *, topic: str, max_questions: int, language: str
    ) -> list[SubQuestion]:
        """Decompose a research topic into sub-questions."""
        ...

    async def critique(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
    ) -> CriticVerdict:
        """Grade coverage and citation support; request more research if needed."""
        ...

    async def write(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
        language: str,
    ) -> tuple[str, list[Citation]]:
        """Synthesize the final report plus its citations."""
        ...


class _PlanOutput(BaseModel):
    """Structured output for the planner agent."""

    questions: list[str] = Field(min_length=1)


class _ReportOutput(BaseModel):
    """Structured output for the writer agent."""

    report: str
    citations: list[Citation] = Field(default_factory=list)


def _usage_from_raw(raw: Any) -> tuple[int, int]:
    """Extract (input, output) token counts from a raw AIMessage."""
    metadata = getattr(raw, "usage_metadata", None) or {}
    return int(metadata.get("input_tokens", 0)), int(metadata.get("output_tokens", 0))


class OpenAILLMClient:
    """Real LLM client backed by OpenAI chat models with structured outputs."""

    def __init__(self, api_key: str, model: str) -> None:
        self._chat = ChatOpenAI(model=model, api_key=SecretStr(api_key), temperature=0)

    async def plan(
        self, *, topic: str, max_questions: int, language: str
    ) -> list[SubQuestion]:
        planner = self._chat.with_structured_output(_PlanOutput, include_raw=True)
        result = await planner.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a research planner. Decompose the topic into focused, "
                        "non-overlapping sub-questions that together cover it. Return at "
                        f"most {max_questions} questions, in {language}."
                    )
                ),
                HumanMessage(content=topic),
            ]
        )
        record_usage("planner", *_usage_from_raw(result["raw"]))
        parsed = cast(_PlanOutput, result["parsed"])
        return [
            SubQuestion(id=f"q{i + 1}", question=q)
            for i, q in enumerate(parsed.questions[:max_questions])
        ]

    async def critique(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
    ) -> CriticVerdict:
        critic = self._chat.with_structured_output(CriticVerdict, include_raw=True)
        evidence = "\n\n".join(
            f"[{s.id}] {s.title or s.url}\n{s.extracted_text[:1500]}" for s in sources
        )
        result = await critic.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a research critic. Given the planned sub-questions and "
                        "the evidence collected so far, grade coverage and whether the "
                        "evidence could support citations. List concrete missing aspects "
                        "only when important gaps remain; do not ask for more research "
                        "on marginal gaps."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\nSub-questions:\n"
                        + "\n".join(f"- {sq.question}" for sq in sub_questions)
                        + f"\n\nEvidence:\n{evidence or '(none)'}"
                    )
                ),
            ]
        )
        record_usage("critic", *_usage_from_raw(result["raw"]))
        return cast(CriticVerdict, result["parsed"])

    async def write(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
        language: str,
    ) -> tuple[str, list[Citation]]:
        writer = self._chat.with_structured_output(_ReportOutput, include_raw=True)
        evidence = "\n\n".join(
            f"[{s.id}] {s.title or s.url} ({s.url})\n{s.extracted_text[:2000]}"
            for s in sources
        )
        result = await writer.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You are a research writer. Write a markdown report answering "
                        f"the topic in {language}. Every factual claim must be backed "
                        "by a citation: claim text, the source id it came from, and a "
                        "short verbatim quote from that source's extracted text. Never "
                        "invent facts or citations."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Topic: {topic}\n\nSub-questions:\n"
                        + "\n".join(f"- {sq.question}" for sq in sub_questions)
                        + f"\n\nEvidence:\n{evidence or '(none)'}"
                    )
                ),
            ]
        )
        record_usage("writer", *_usage_from_raw(result["raw"]))
        parsed = cast(_ReportOutput, result["parsed"])
        return parsed.report, parsed.citations


class StubLLMClient:
    """Deterministic offline LLM double.

    ``force_needs_more`` makes the critic always request another research round,
    which lets tests prove the critic/re-plan loop is bounded.
    """

    def __init__(self, *, force_needs_more: bool = False) -> None:
        self.force_needs_more = force_needs_more

    async def plan(
        self, *, topic: str, max_questions: int, language: str
    ) -> list[SubQuestion]:
        del language
        record_usage("planner", 400, 150)
        count = min(3, max_questions)
        return [
            SubQuestion(id=f"q{i + 1}", question=f"{topic} — aspect {i + 1}")
            for i in range(count)
        ]

    async def critique(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
    ) -> CriticVerdict:
        del topic, sub_questions
        record_usage("critic", 800, 200)
        if self.force_needs_more:
            return CriticVerdict(
                coverage_score=0.4,
                citation_support_score=0.4,
                missing_aspects=["additional aspect"],
                needs_more_research=True,
                feedback="stub: always needs more research",
            )
        has_sources = bool(sources)
        return CriticVerdict(
            coverage_score=1.0 if has_sources else 0.0,
            citation_support_score=1.0 if has_sources else 0.0,
            missing_aspects=[],
            needs_more_research=False,
            feedback="stub: satisfied",
        )

    async def write(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
        language: str,
    ) -> tuple[str, list[Citation]]:
        del language
        record_usage("writer", 1200, 400)
        lines = [f"# Research report: {topic}", ""]
        citations: list[Citation] = []
        if not sub_questions:
            lines.append("No sub-questions were planned.")
        for sq in sub_questions:
            lines.append(f"## {sq.question}")
            related = [s for s in sources if s.sub_question_id == sq.id]
            if not related:
                lines.append("No evidence was collected for this aspect.")
            for source in related:
                quote = source.extracted_text[:120].strip() or source.url
                lines.append(f"- Finding backed by {source.url}: {quote}")
                citations.append(
                    Citation(
                        claim=f"Finding for '{sq.question}' from {source.url}",
                        source_id=source.id,
                        quote=quote,
                    )
                )
        return "\n".join(lines), citations


def get_llm_client(settings: Settings) -> LLMClient:
    """Factory: real client when configured, deterministic stub otherwise."""
    if settings.has_llm_credentials():
        assert settings.openai_api_key is not None
        return OpenAILLMClient(
            api_key=settings.openai_api_key.get_secret_value(), model=settings.llm_model
        )
    return StubLLMClient()
