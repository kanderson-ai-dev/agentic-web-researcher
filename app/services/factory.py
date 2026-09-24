"""Composition root: build :class:`GraphDeps` from application settings.

When credentials are absent the factory degrades to deterministic doubles
(``StubLLMClient`` / ``NullSearchClient``) so the app — and CI — run fully
offline. Real providers are wired only when keys are configured.
"""

from app.core.config import Settings
from app.core.schemas import RawDocument, SearchResult, Source, SubQuestion
from app.graph.deps import GraphDeps
from app.services.document_parser import parse_raw_document
from app.services.llm_client import get_llm_client
from app.services.scraper_client import ScraperClient
from app.services.search_client import (
    DuckDuckGoSearchClient,
    NullSearchClient,
    SearchClient,
    TavilySearchClient,
)


def build_graph_deps(settings: Settings) -> GraphDeps:
    """Wire production dependencies for the research graph."""
    llm = get_llm_client(settings)

    search_client: SearchClient
    provider = settings.search_provider.lower()
    if settings.has_search_credentials() and provider in {"auto", "tavily"}:
        assert settings.search_api_key is not None
        search_client = TavilySearchClient(
            api_key=settings.search_api_key.get_secret_value(),
            base_url=settings.search_api_base_url,
        )
    elif provider in {"auto", "duckduckgo"}:
        search_client = DuckDuckGoSearchClient()
    else:
        search_client = NullSearchClient()

    scraper = ScraperClient(
        user_agent=settings.scrape_user_agent,
        delay_seconds=settings.scrape_delay_seconds,
        timeout_seconds=settings.scrape_timeout_seconds,
        max_bytes=settings.scrape_max_bytes,
    )

    async def search(sub_question: SubQuestion) -> list[SearchResult]:
        return await search_client.search(
            sub_question.question, max_results=settings.max_search_results_per_question
        )

    async def scrape(result: SearchResult) -> RawDocument | None:
        return await scraper.fetch(result.url, sub_question_id=result.sub_question_id)

    async def parse(document: RawDocument) -> Source | None:
        return parse_raw_document(document)

    return GraphDeps(
        llm=llm,
        search=search,
        scrape=scrape,
        parse=parse,
        max_search_results=settings.max_search_results_per_question,
        max_sub_questions=settings.max_sub_questions,
        require_human_review=True,
    )
