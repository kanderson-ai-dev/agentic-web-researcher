"""Tests for the StructuredTool wrappers around service clients."""

import pytest
from pydantic import ValidationError

from app.core.schemas import RawDocument, SearchResult
from app.services.scraper_client import ScraperClient
from app.services.tools import (
    ParseDocumentInput,
    ScrapeUrlInput,
    WebSearchInput,
    build_parse_tool,
    build_scrape_tool,
    build_search_tool,
)


class _FakeSearch:
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                url="https://example.com",
                title="t",
                snippet="s",
                sub_question_id="unassigned",
            )
        ]


async def test_search_tool_invokes_client() -> None:
    tool = build_search_tool(_FakeSearch())
    assert tool.name == "web_search"
    result = await tool.ainvoke({"query": "AI regulation", "max_results": 2})
    assert len(result) == 1
    assert result[0].url == "https://example.com"


def test_search_tool_input_validation() -> None:
    with pytest.raises(ValidationError):
        WebSearchInput(query="")
    with pytest.raises(ValidationError):
        WebSearchInput(query="x", max_results=99)


async def test_scrape_tool_invokes_scraper() -> None:
    class FakeScraper(ScraperClient):
        def __init__(self) -> None:
            pass

        async def fetch(self, url: str, *, sub_question_id: str = "") -> RawDocument | None:
            return RawDocument(
                url=url, content_ref="d-x", sub_question_id=sub_question_id
            )

    tool = build_scrape_tool(FakeScraper())
    result = await tool.ainvoke({"url": "https://example.com", "sub_question_id": "q1"})
    assert result is not None
    assert result.sub_question_id == "q1"


async def test_parse_tool_parses_html() -> None:
    tool = build_parse_tool()
    result = await tool.ainvoke(
        {
            "url": "https://example.com",
            "content": (
                "<html><head><title>T</title></head>"
                "<body><p>Body text here.</p></body></html>"
            ),
            "content_type": "text/html",
            "sub_question_id": "q1",
        }
    )
    assert result is not None
    assert result.title == "T"
    assert "Body text" in result.extracted_text


def test_parse_tool_input_defaults() -> None:
    inp = ParseDocumentInput(url="https://example.com")
    assert inp.content_type == "text/html"
    assert inp.content_bytes_b64 is None


def test_scrape_input_defaults() -> None:
    inp = ScrapeUrlInput(url="https://example.com")
    assert inp.sub_question_id == "unassigned"
