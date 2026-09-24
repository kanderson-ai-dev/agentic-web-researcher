"""Web search client (Tavily-compatible API).

- Real HTTP via ``httpx.AsyncClient`` with ``tenacity`` retries.
- In-memory per-query result cache so re-plan rounds don't re-pay for the
  same search.
- ``NullSearchClient`` degrades cleanly when no API key is configured — the
  rest of the pipeline still runs (it simply gathers no web evidence).
"""

from typing import Protocol

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.schemas import SearchResult


class SearchClient(Protocol):
    """Interface for web search providers."""

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        """Return candidate source URLs for a query."""
        ...


class TavilySearchClient:
    """Tavily ``/search`` API client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.tavily.com",
        timeout_seconds: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._cache: dict[tuple[str, int], list[SearchResult]] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def _post(self, query: str, max_results: int) -> httpx.Response:
        response = await self._client.post(
            f"{self._base_url}/search",
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
        )
        response.raise_for_status()
        return response

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        key = (query, max_results)
        if key in self._cache:
            return self._cache[key]
        response = await self._post(query, max_results)
        payload = response.json()
        results = [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title"),
                snippet=item.get("content", ""),
                sub_question_id="unassigned",
                score=item.get("score"),
            )
            for item in payload.get("results", [])
            if item.get("url")
        ]
        self._cache[key] = results
        return results


class NullSearchClient:
    """Fallback when no search API key is configured: gathers no evidence."""

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        del query, max_results
        return []
