"""LangGraph state for the research pipeline.

Worker fan-out uses ``operator.add`` reducers so results produced in parallel
branches accumulate into a single list when the branches join.
"""

import operator
from typing import Annotated

from typing_extensions import TypedDict

from app.core.schemas import (
    Citation,
    CriticVerdict,
    RawDocument,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
)


class ResearchState(TypedDict, total=False):
    """Full state threaded through the research graph."""

    job_id: str
    request: ResearchRequest
    topic: str

    blocked: bool
    rejection_reason: str | None

    sub_questions: list[SubQuestion]
    pending_questions: list[SubQuestion]

    search_results: Annotated[list[SearchResult], operator.add]
    raw_documents: Annotated[list[RawDocument], operator.add]
    sources: Annotated[list[Source], operator.add]

    critic_verdict: CriticVerdict | None
    critic_rounds: int
    max_critic_rounds: int

    report: str | None
    citations: list[Citation]
    dropped_citations: int
    errors: Annotated[list[str], operator.add]


class SearchTask(TypedDict):
    """Input slice sent to each ``search_worker`` branch."""

    sub_question: SubQuestion


class ScrapeTask(TypedDict):
    """Input slice sent to each ``scrape_worker`` branch."""

    search_result: SearchResult


class ParseTask(TypedDict):
    """Input slice sent to each ``document_worker`` branch."""

    raw_document: RawDocument


def initial_state(
    request: ResearchRequest, *, job_id: str, max_critic_rounds: int = 2
) -> ResearchState:
    """Build the starting state for a research job."""
    return {
        "job_id": job_id,
        "request": request,
        "topic": request.topic,
        "blocked": False,
        "rejection_reason": None,
        "sub_questions": [],
        "pending_questions": [],
        "search_results": [],
        "raw_documents": [],
        "sources": [],
        "critic_verdict": None,
        "critic_rounds": 0,
        "max_critic_rounds": max_critic_rounds,
        "report": None,
        "citations": [],
        "dropped_citations": 0,
        "errors": [],
    }
