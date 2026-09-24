"""Resilient, ethics-first web scraper client.

- Respects ``robots.txt`` per origin (cached).
- Sends a realistic, identifiable User-Agent.
- Enforces a minimum delay between requests (mandatory, not optional).
- Caps response size; only ``200`` HTML/PDF payloads produce documents.
- Optional Playwright fallback for JS-heavy pages (lazy import; disabled by
  default and injected as a ``BrowserFetcher`` so tests stay offline).
"""

import asyncio
import base64
from typing import Protocol
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.schemas import RawDocument

_TEXT_TYPES = ("text/html", "text/plain", "application/xhtml")
_BINARY_TYPES = ("application/pdf",)


class BrowserFetcher(Protocol):
    """Optional JS-rendering fallback (e.g. Playwright)."""

    async def fetch(self, url: str) -> str | None:
        """Return rendered HTML for a URL, or None on failure."""
        ...


class PlaywrightBrowserFetcher:
    """Playwright-backed fetcher; imported lazily so it stays an optional dep."""

    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._timeout = timeout_seconds

    async def fetch(self, url: str) -> str | None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Playwright is not installed; install it and run `playwright install` "
                "to enable the browser fallback."
            ) from exc
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page()
                response = await page.goto(url, timeout=self._timeout * 1000)
                if response is None or response.status != 200:
                    return None
                return str(await page.content())
            finally:
                await browser.close()


class ScraperClient:
    """HTTP scraper with robots.txt enforcement and polite rate limiting."""

    def __init__(
        self,
        *,
        user_agent: str,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 15.0,
        max_bytes: int = 2_000_000,
        client: httpx.AsyncClient | None = None,
        browser: BrowserFetcher | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._delay = delay_seconds
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds, follow_redirects=True
        )
        self._browser = browser
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at = 0.0

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- politeness ---------------------------------------------------------

    async def _throttle(self) -> None:
        elapsed = asyncio.get_event_loop().time() - self._last_request_at
        if elapsed < self._delay:
            await asyncio.sleep(self._delay - elapsed)
        self._last_request_at = asyncio.get_event_loop().time()

    async def _is_allowed_by_robots(self, url: str) -> bool:
        origin = self._origin(url)
        if origin not in self._robots:
            parser = RobotFileParser()
            try:
                response = await self._client.get(f"{origin}/robots.txt")
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())
                else:
                    parser.parse([])  # no robots file -> allow all
            except httpx.HTTPError:
                parser.parse([])  # unreachable robots -> allow
            self._robots[origin] = parser
        return self._robots[origin].can_fetch(self._user_agent, url)

    # -- fetching -----------------------------------------------------------

    @staticmethod
    def _origin(url: str) -> str:
        parts = urlparse(url)
        return f"{parts.scheme}://{parts.netloc}"

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def _get(self, url: str) -> httpx.Response:
        return await self._client.get(url, headers={"User-Agent": self._user_agent})

    _ALLOWED_SCHEMES = {"http", "https"}

    async def fetch(self, url: str, *, sub_question_id: str = "unassigned") -> RawDocument | None:
        """Fetch a URL into a RawDocument, or None when skipped/failed."""
        if urlparse(url).scheme not in self._ALLOWED_SCHEMES:
            return None  # SSRF guard: only http(s) URLs may be fetched
        if not await self._is_allowed_by_robots(url):
            return None

        await self._throttle()
        try:
            response = await self._get(url)
        except httpx.HTTPError:
            return await self.fetch_rendered(url, sub_question_id=sub_question_id)
        if response.status_code != 200 or len(response.content) > self._max_bytes:
            return None

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if any(t in content_type for t in _BINARY_TYPES):
            return RawDocument(
                url=url,
                content="",
                content_bytes_b64=base64.b64encode(response.content).decode("ascii"),
                content_type=content_type or "application/pdf",
                status_code=response.status_code,
                sub_question_id=sub_question_id,
            )
        if content_type and not any(t in content_type for t in _TEXT_TYPES):
            return None

        text = response.text
        if not text.strip():
            return await self.fetch_rendered(url, sub_question_id=sub_question_id)
        return RawDocument(
            url=url,
            content=text,
            content_type=content_type or "text/html",
            status_code=response.status_code,
            sub_question_id=sub_question_id,
        )

    async def fetch_rendered(
        self, url: str, *, sub_question_id: str = "unassigned"
    ) -> RawDocument | None:
        """Fetch via the browser fallback when configured; None otherwise."""
        if self._browser is None:
            return None
        html = await self._browser.fetch(url)
        if not html:
            return None
        return RawDocument(
            url=url,
            content=html,
            content_type="text/html",
            status_code=200,
            sub_question_id=sub_question_id,
        )
