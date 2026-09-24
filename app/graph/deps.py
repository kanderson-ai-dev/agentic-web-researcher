"""Injected dependencies for the research graph.

Nodes never import concrete tool implementations: they call the callables in
:class:`GraphDeps`, so tests can inject deterministic doubles and production
wires the real clients from ``app.services``.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core.schemas import RawDocument, SearchResult, Source, SubQuestion
from app.services.llm_client import LLMClient

SearchFn = Callable[[SubQuestion], Awaitable[list[SearchResult]]]
ScrapeFn = Callable[[SearchResult], Awaitable[RawDocument | None]]
ParseFn = Callable[[RawDocument], Awaitable[Source | None]]


@dataclass(frozen=True)
class GraphDeps:
    """Everything the graph needs beyond its own state."""

    llm: LLMClient
    search: SearchFn
    scrape: ScrapeFn
    parse: ParseFn
    max_search_results: int = 3
    max_sub_questions: int = 6
