"""Domain schemas for the research pipeline.

These models describe the full lifecycle of a research job: the incoming
request, the planner's sub-questions, the evidence gathered by workers
(search results -> raw documents -> parsed sources), the critic's verdict,
and the final report with citations.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResearchRequest(BaseModel):
    """Payload accepted by ``POST /api/v1/research``."""

    model_config = ConfigDict(str_strip_whitespace=True)

    topic: str = Field(min_length=10, max_length=2_000)
    depth: Literal["quick", "standard", "deep"] = "standard"
    language: str = Field(default="en", min_length=2, max_length=10)


class SubQuestionStatus(StrEnum):
    """Lifecycle of a single planned sub-question."""

    PENDING = "pending"
    ANSWERED = "answered"
    UNRESOLVED = "unresolved"


class SubQuestion(BaseModel):
    """One research sub-question produced by the planner."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    status: SubQuestionStatus = SubQuestionStatus.PENDING


class SearchResult(BaseModel):
    """A candidate source URL returned by the search worker."""

    url: str = Field(min_length=1)
    title: str | None = None
    snippet: str = ""
    sub_question_id: str = Field(min_length=1)
    score: float | None = None


class RawDocument(BaseModel):
    """Untrusted raw content fetched by the scrape worker.

    The raw payload is never interpolated into prompts directly — it must pass
    the sanitization guardrail and be parsed into a :class:`Source` first. It
    also never travels inside graph state: ``content_ref`` points into the
    process-local :class:`DocumentStore` so checkpoints and LangSmith traces
    stay small.
    """

    url: str = Field(min_length=1)
    content_ref: str = ""
    content_type: str = "text/html"
    status_code: int = 200
    byte_size: int = 0
    sub_question_id: str = Field(min_length=1)


class Source(BaseModel):
    """A parsed, sanitized piece of evidence used to back citations."""

    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = Field(min_length=1)
    extracted_text: str = ""
    sub_question_id: str | None = None


class Citation(BaseModel):
    """A claim in the final report and the source quote that supports it."""

    claim: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class CriticVerdict(BaseModel):
    """Structured output of the critic agent for one evaluation round."""

    coverage_score: float = Field(ge=0.0, le=1.0)
    citation_support_score: float = Field(ge=0.0, le=1.0)
    missing_aspects: list[str] = Field(default_factory=list)
    needs_more_research: bool = False
    feedback: str = ""


class JobStatus(StrEnum):
    """Lifecycle of a research job."""

    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"


class JobEvent(BaseModel):
    """Progress event streamed to clients over SSE."""

    job_id: str
    node: str
    status: Literal["started", "completed", "failed", "awaiting_review"]
    detail: str = ""
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewDecision(BaseModel):
    """Human reviewer decision for a job paused at the HITL gate."""

    action: Literal["approve", "edit", "reject"]
    report: str | None = Field(default=None, max_length=100_000)
    notes: str = Field(default="", max_length=2_000)


class ResearchJob(BaseModel):
    """Persisted record of a research job and its final result."""

    id: str = Field(min_length=1)
    status: JobStatus = JobStatus.QUEUED
    request: ResearchRequest
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    report: str | None = None
    cost_usd: float = 0.0
    timings: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
