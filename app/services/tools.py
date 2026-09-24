"""Typed LangChain tool wrappers around the service clients.

These expose search/scrape/parse as ``StructuredTool``s with Pydantic input
schemas, so they can be handed to tool-calling agents. The graph's worker
nodes call the same underlying functions through :class:`GraphDeps`.
"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.schemas import RawDocument, SearchResult, Source
from app.services.document_parser import parse_raw_document
from app.services.scraper_client import ScraperClient
from app.services.search_client import SearchClient


class WebSearchInput(BaseModel):
    """Input for the web search tool."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=3, ge=1, le=10)


class ScrapeUrlInput(BaseModel):
    """Input for the URL scrape tool."""

    url: str = Field(min_length=1)
    sub_question_id: str = "unassigned"


class ParseDocumentInput(BaseModel):
    """Input for the document parsing tool."""

    url: str = Field(min_length=1)
    content: str = ""
    content_bytes_b64: str | None = None
    content_type: str = "text/html"
    sub_question_id: str = "unassigned"


def build_search_tool(client: SearchClient) -> StructuredTool:
    """Wrap a :class:`SearchClient` as a StructuredTool."""

    async def _run(query: str, max_results: int = 3) -> list[SearchResult]:
        return await client.search(query, max_results=max_results)

    return StructuredTool.from_function(
        coroutine=_run,
        name="web_search",
        description=(
            "Search the web for a query. Returns candidate source URLs with "
            "titles and snippets."
        ),
        args_schema=WebSearchInput,
    )


def build_scrape_tool(scraper: ScraperClient) -> StructuredTool:
    """Wrap a :class:`ScraperClient` as a StructuredTool."""

    async def _run(url: str, sub_question_id: str = "unassigned") -> RawDocument | None:
        return await scraper.fetch(url, sub_question_id=sub_question_id)

    return StructuredTool.from_function(
        coroutine=_run,
        name="scrape_url",
        description=(
            "Fetch a URL's raw content. Respects robots.txt, rate limits and "
            "size caps; returns null when the fetch is skipped or fails."
        ),
        args_schema=ScrapeUrlInput,
    )


def build_parse_tool() -> StructuredTool:
    """Expose the deterministic document parser as a StructuredTool."""

    async def _run(
        url: str,
        content: str = "",
        content_bytes_b64: str | None = None,
        content_type: str = "text/html",
        sub_question_id: str = "unassigned",
    ) -> Source | None:
        return parse_raw_document(
            RawDocument(
                url=url,
                content=content,
                content_bytes_b64=content_bytes_b64,
                content_type=content_type,
                sub_question_id=sub_question_id or "unassigned",
            )
        )

    return StructuredTool.from_function(
        coroutine=_run,
        name="parse_document",
        description=(
            "Parse raw HTML or base64-encoded PDF content into clean text with "
            "metadata. Returns null when nothing usable is extracted."
        ),
        args_schema=ParseDocumentInput,
    )
