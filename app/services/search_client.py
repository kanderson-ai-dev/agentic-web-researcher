"""Web search client (Tavily-compatible API).

- Real HTTP via ``httpx.AsyncClient`` with ``tenacity`` retries.
- In-memory per-query result cache so re-plan rounds don't re-pay for the
  same search.
- ``NullSearchClient`` degrades cleanly when no API key is configured — the
  rest of the pipeline still runs (it simply gathers no web evidence).
- ``DuckDuckGoSearchClient`` is a free, keyless fallback that parses
  DuckDuckGo's HTML endpoint — real web search without any signup.
"""

import asyncio
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup
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


class DuckDuckGoSearchClient:
    """Free, keyless web search via DuckDuckGo's HTML endpoint.

    Result links arrive as ``//duckduckgo.com/l/?uddg=<encoded>`` redirects;
    the real URL is extracted from the ``uddg`` parameter. A small delay
    between queries keeps usage polite.
    """

    _ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        *,
        timeout_seconds: float = 15.0,
        delay_seconds: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-agent/0.1)"},
            follow_redirects=True,
        )
        self._delay = delay_seconds
        self._last_request = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        elapsed = asyncio.get_running_loop().time() - self._last_request
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request = asyncio.get_running_loop().time()

    @staticmethod
    def _result_url(href: str) -> str | None:
        """Extract the destination URL from a DDG redirect link."""
        if "uddg=" in href:
            target = parse_qs(urlparse(href).query).get("uddg", [None])[0]
            return unquote(target) if target else None
        parsed = urlparse(href if "://" in href else f"https:{href}")
        return parsed.geturl() if parsed.scheme in {"http", "https"} else None

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        await self._throttle()
        try:
            response = await self._client.get(
                self._ENDPOINT, params={"q": query}
            )
        except httpx.HTTPError:
            return []
        if response.status_code != 200:
            return []

        soup = BeautifulSoup(response.text, "lxml")
        results: list[SearchResult] = []
        seen: set[str] = set()
        for anchor in soup.select("a.result__a"):
            href = anchor.get("href", "")
            url = self._result_url(href if isinstance(href, str) else "")
            if not url or url in seen:
                continue
            seen.add(url)
            container = anchor.find_parent("div", class_="result")
            snippet_tag = (
                container.select_one(".result__snippet") if container else None
            )
            results.append(
                SearchResult(
                    url=url,
                    title=anchor.get_text(strip=True) or None,
                    snippet=(
                        snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                    ),
                    sub_question_id="unassigned",
                )
            )
            if len(results) >= max_results:
                break
        return results


class NullSearchClient:
    """Fallback when no search API key is configured: gathers no evidence."""

    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        del query, max_results
        return []
