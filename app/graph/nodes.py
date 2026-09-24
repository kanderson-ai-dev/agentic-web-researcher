"""Node factories for the research graph.

Each factory closes over :class:`GraphDeps` so the compiled graph carries its
dependencies without globals. Nodes return *partial* state updates; reducer
channels merge parallel worker output automatically.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from app.core.schemas import SubQuestion, SubQuestionStatus
from app.graph.deps import GraphDeps
from app.graph.guardrails import (
    sanitize_scraped_content,
    screen_topic,
    verify_citations,
)
from app.graph.state import ParseTask, ResearchState, ScrapeTask, SearchTask

_DEPTH_TARGETS = {"quick": 3, "standard": 5, "deep": 8}

StateNode = Callable[[ResearchState], Awaitable[dict[str, Any]]]
SearchNode = Callable[[SearchTask], Awaitable[dict[str, Any]]]
ScrapeNode = Callable[[ScrapeTask], Awaitable[dict[str, Any]]]
ParseNode = Callable[[ParseTask], Awaitable[dict[str, Any]]]


def make_input_guardrail(_deps: GraphDeps) -> StateNode:
    """Screen the topic before any LLM call or external request."""

    async def input_guardrail(state: ResearchState) -> dict[str, Any]:
        result = screen_topic(state["topic"])
        if not result.allowed:
            return {"blocked": True, "rejection_reason": result.reason}
        return {"blocked": False, "topic": result.sanitized_topic}

    return input_guardrail


def make_rejection_output(_deps: GraphDeps) -> StateNode:
    """Terminal node for blocked requests: generic refusal, no detail leakage."""

    async def rejection_output(state: ResearchState) -> dict[str, Any]:
        del state
        return {
            "report": "This research request could not be processed.",
            "citations": [],
        }

    return rejection_output


def make_planner(deps: GraphDeps) -> StateNode:
    """Decompose the topic into sub-questions via the LLM."""

    async def planner(state: ResearchState) -> dict[str, Any]:
        request = state["request"]
        target = min(_DEPTH_TARGETS[request.depth], deps.max_sub_questions)
        questions = await deps.llm.plan(
            topic=state["topic"], max_questions=target, language=request.language
        )
        return {"sub_questions": questions, "pending_questions": questions}

    return planner


def make_search_worker(deps: GraphDeps) -> SearchNode:
    """Run a web search for one sub-question (fan-out branch)."""

    async def search_worker(task: SearchTask) -> dict[str, Any]:
        sub_question = task["sub_question"]
        try:
            results = await deps.search(sub_question)
        except Exception as exc:  # noqa: BLE001 — record and continue
            return {"errors": [f"search failed for {sub_question.id}: {exc}"]}
        capped = results[: deps.max_search_results]
        for result in capped:
            result.sub_question_id = sub_question.id
        return {"search_results": capped}

    return search_worker


def make_scrape_worker(deps: GraphDeps) -> ScrapeNode:
    """Fetch raw content for one search result (fan-out branch)."""

    async def scrape_worker(task: ScrapeTask) -> dict[str, Any]:
        result = task["search_result"]
        try:
            document = await deps.scrape(result)
        except Exception as exc:  # noqa: BLE001 — record and continue
            return {"errors": [f"scrape failed for {result.url}: {exc}"]}
        if document is None:
            return {"errors": [f"scrape skipped for {result.url}"]}
        return {"raw_documents": [document]}

    return scrape_worker


def make_document_worker(deps: GraphDeps) -> ParseNode:
    """Parse one raw document into a sanitized Source (fan-out branch)."""

    async def document_worker(task: ParseTask) -> dict[str, Any]:
        document = task["raw_document"]
        try:
            source = await deps.parse(document)
        except Exception as exc:  # noqa: BLE001 — record and continue
            return {"errors": [f"parse failed for {document.url}: {exc}"]}
        if source is None:
            return {"errors": [f"parse produced no source for {document.url}"]}
        sanitized = source.model_copy(
            update={
                "extracted_text": sanitize_scraped_content(source.extracted_text)
            }
        )
        return {"sources": [sanitized]}

    return document_worker


def make_critic(deps: GraphDeps) -> StateNode:
    """Grade coverage/citation support and decide whether to re-plan."""

    async def critic(state: ResearchState) -> dict[str, Any]:
        verdict = await deps.llm.critique(
            topic=state["topic"],
            sub_questions=state.get("sub_questions", []),
            sources=state.get("sources", []),
        )
        return {
            "critic_verdict": verdict,
            "critic_rounds": state.get("critic_rounds", 0) + 1,
        }

    return critic


def make_replan(_deps: GraphDeps) -> StateNode:
    """Turn the critic's missing aspects into new pending sub-questions."""

    async def replan(state: ResearchState) -> dict[str, Any]:
        verdict = state.get("critic_verdict")
        if verdict is None or not verdict.missing_aspects:
            return {"pending_questions": []}
        existing = {sq.question for sq in state.get("sub_questions", [])}
        round_no = state.get("critic_rounds", 1)
        new_questions: list[SubQuestion] = []
        for aspect in verdict.missing_aspects:
            if aspect in existing:
                continue
            existing.add(aspect)
            new_questions.append(
                SubQuestion(id=f"r{round_no}q{len(new_questions) + 1}", question=aspect)
            )
        return {
            "pending_questions": new_questions,
            "sub_questions": [*state.get("sub_questions", []), *new_questions],
        }

    return replan


def make_writer(deps: GraphDeps) -> StateNode:
    """Synthesize the final report and citations."""

    async def writer(state: ResearchState) -> dict[str, Any]:
        report, citations = await deps.llm.write(
            topic=state["topic"],
            sub_questions=state.get("sub_questions", []),
            sources=state.get("sources", []),
            language=state["request"].language,
        )
        return {"report": report, "citations": citations}

    return writer


def make_output_guardrail(_deps: GraphDeps) -> StateNode:
    """Drop citations whose quote cannot be verified against its source."""

    async def output_guardrail(state: ResearchState) -> dict[str, Any]:
        kept, dropped = verify_citations(
            state.get("citations", []), state.get("sources", [])
        )
        update: dict[str, Any] = {
            "citations": kept,
            "dropped_citations": len(dropped),
        }
        if dropped:
            update["errors"] = [
                f"output guardrail dropped {len(dropped)} unsupported citations"
            ]
        return update

    return output_guardrail


def make_report_assembler(_deps: GraphDeps) -> StateNode:
    """Finalize sub-question statuses and a deterministic source hash index."""

    async def report_assembler(state: ResearchState) -> dict[str, Any]:
        sources = state.get("sources", [])
        answered_ids = {s.sub_question_id for s in sources}
        updated = [
            sq.model_copy(
                update={
                    "status": (
                        SubQuestionStatus.ANSWERED
                        if sq.id in answered_ids
                        else SubQuestionStatus.UNRESOLVED
                    )
                }
            )
            for sq in state.get("sub_questions", [])
        ]
        return {"sub_questions": updated}

    return report_assembler
