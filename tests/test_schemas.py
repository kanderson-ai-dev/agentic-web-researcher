"""Validation tests for the domain schemas."""

import pytest
from pydantic import ValidationError

from app.core.schemas import (
    Citation,
    CriticVerdict,
    JobStatus,
    RawDocument,
    ResearchJob,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
    SubQuestionStatus,
)


def test_research_request_valid() -> None:
    request = ResearchRequest(topic="  impact of EU AI Act on startups  ")
    assert request.topic == "impact of EU AI Act on startups"
    assert request.depth == "standard"
    assert request.language == "en"


def test_research_request_rejects_short_topic() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="too short")


def test_research_request_rejects_blank_topic() -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(topic="          ")


def test_sub_question_defaults_to_pending() -> None:
    sq = SubQuestion(id="q1", question="What is the AI Act?")
    assert sq.status is SubQuestionStatus.PENDING


def test_search_result_requires_sub_question_link() -> None:
    with pytest.raises(ValidationError):
        SearchResult(url="https://example.com", sub_question_id="")


def test_source_and_citation_round_trip() -> None:
    source = Source(
        id="s1",
        url="https://example.com/article",
        title="Article",
        content_hash="abc123",
        extracted_text="Some extracted text.",
        sub_question_id="q1",
    )
    citation = Citation(
        claim="The AI Act entered into force in 2024.", source_id="s1", quote="2024"
    )
    assert citation.source_id == source.id


def test_critic_verdict_bounds() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict(coverage_score=1.5, citation_support_score=0.5)


def test_research_job_serialization_round_trip() -> None:
    job = ResearchJob(
        id="job-1",
        request=ResearchRequest(topic="impact of EU AI Act on startups"),
        sub_questions=[SubQuestion(id="q1", question="Scope of the AI Act?")],
        sources=[
            Source(
                id="s1",
                url="https://example.com",
                content_hash="deadbeef",
                extracted_text="text",
            )
        ],
        report="Final report.",
        cost_usd=0.012,
    )
    restored = ResearchJob.model_validate_json(job.model_dump_json())
    assert restored.id == job.id
    assert restored.status is JobStatus.QUEUED
    assert restored.sub_questions[0].status is SubQuestionStatus.PENDING
    assert restored.report == "Final report."


def test_raw_document_carries_payload_reference() -> None:
    doc = RawDocument(
        url="https://example.com",
        content_ref="d-abc123",
        byte_size=2048,
        sub_question_id="q1",
    )
    assert doc.content_ref == "d-abc123"
    assert doc.byte_size == 2048
