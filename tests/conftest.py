"""Shared fixtures and deterministic doubles for the test suite."""

import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest

from app.core.schemas import (
    RawDocument,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
)
from app.graph.deps import GraphDeps, ParseFn, ScrapeFn, SearchFn
from app.services.llm_client import StubLLMClient


@pytest.fixture
def research_request() -> ResearchRequest:
    return ResearchRequest(topic="impact of the EU AI Act on small startups")


@pytest.fixture
def fake_search() -> SearchFn:
    async def _search(sub_question: SubQuestion) -> list[SearchResult]:
        return [
            SearchResult(
                url=f"https://example.com/{sub_question.id}/{i}",
                title=f"Result {i} for {sub_question.id}",
                snippet=f"snippet about {sub_question.question}",
                sub_question_id=sub_question.id,
            )
            for i in range(2)
        ]

    return _search


@pytest.fixture
def fake_scrape() -> ScrapeFn:
    async def _scrape(result: SearchResult) -> RawDocument | None:
        return RawDocument(
            url=result.url,
            content=f"<html><body>Article body for {result.url}</body></html>",
            content_type="text/html",
            status_code=200,
            sub_question_id=result.sub_question_id,
        )

    return _scrape


@pytest.fixture
def fake_parse() -> ParseFn:
    async def _parse(document: RawDocument) -> Source | None:
        digest = hashlib.sha256(document.content.encode()).hexdigest()[:12]
        return Source(
            id=f"s-{digest}",
            url=document.url,
            title=f"Parsed {document.url}",
            fetched_at=datetime.now(UTC),
            content_hash=digest,
            extracted_text=f"Evidence text extracted from {document.url}.",
            sub_question_id=document.sub_question_id,
        )

    return _parse


@pytest.fixture
def stub_deps(
    fake_search: SearchFn, fake_scrape: ScrapeFn, fake_parse: ParseFn
) -> GraphDeps:
    return GraphDeps(
        llm=StubLLMClient(),
        search=fake_search,
        scrape=fake_scrape,
        parse=fake_parse,
        max_search_results=3,
        max_sub_questions=6,
    )


# Re-export the callable types so tests stay annotated consistently.
SearchCallable = Callable[[SubQuestion], Awaitable[list[SearchResult]]]
