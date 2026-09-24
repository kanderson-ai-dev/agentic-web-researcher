"""Tests for the scraper client: robots.txt, UA, size/type limits, fallback."""

import base64

import httpx
import pytest
import respx

from app.services.scraper_client import ScraperClient

ORIGIN = "https://example.com"
ROBOTS_ALLOW = "User-agent: *\nAllow: /\n"
ROBOTS_DENY = "User-agent: *\nDisallow: /private/\n"
HTML = "<html><body><p>Article body</p></body></html>"


def _client(**kwargs) -> ScraperClient:
    defaults = {"user_agent": "test-agent/1.0", "delay_seconds": 0.0}
    return ScraperClient(**{**defaults, **kwargs})


async def test_fetch_returns_document_and_sends_user_agent() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_ALLOW)
        )
        route = router.get(f"{ORIGIN}/page").mock(
            return_value=httpx.Response(
                200, text=HTML, headers={"content-type": "text/html; charset=utf-8"}
            )
        )
        scraper = _client()
        doc = await scraper.fetch(f"{ORIGIN}/page", sub_question_id="q1")
        await scraper.aclose()

    assert doc is not None
    assert doc.content == HTML
    assert doc.sub_question_id == "q1"
    assert route.calls.last.request.headers["user-agent"] == "test-agent/1.0"


async def test_fetch_respects_robots_disallow() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_DENY)
        )
        page_route = router.get(f"{ORIGIN}/private/page")
        scraper = _client()
        doc = await scraper.fetch(f"{ORIGIN}/private/page")
        await scraper.aclose()

    assert doc is None
    assert not page_route.called


async def test_fetch_proceeds_when_robots_missing() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(return_value=httpx.Response(404))
        router.get(f"{ORIGIN}/page").mock(
            return_value=httpx.Response(200, text=HTML, headers={"content-type": "text/html"})
        )
        scraper = _client()
        doc = await scraper.fetch(f"{ORIGIN}/page")
        await scraper.aclose()

    assert doc is not None


async def test_fetch_skips_non_200() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_ALLOW)
        )
        router.get(f"{ORIGIN}/gone").mock(return_value=httpx.Response(404))
        scraper = _client()
        assert await scraper.fetch(f"{ORIGIN}/gone") is None
        await scraper.aclose()


async def test_fetch_skips_unsupported_content_type() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_ALLOW)
        )
        router.get(f"{ORIGIN}/img").mock(
            return_value=httpx.Response(
                200, content=b"\x89PNG", headers={"content-type": "image/png"}
            )
        )
        scraper = _client()
        assert await scraper.fetch(f"{ORIGIN}/img") is None
        await scraper.aclose()


async def test_fetch_pdf_stores_bytes() -> None:
    pdf_bytes = b"%PDF-1.4 fake"
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_ALLOW)
        )
        router.get(f"{ORIGIN}/doc.pdf").mock(
            return_value=httpx.Response(
                200, content=pdf_bytes, headers={"content-type": "application/pdf"}
            )
        )
        scraper = _client()
        doc = await scraper.fetch(f"{ORIGIN}/doc.pdf")
        await scraper.aclose()

    assert doc is not None
    assert doc.content_bytes_b64 is not None
    assert base64.b64decode(doc.content_bytes_b64) == pdf_bytes
    assert doc.content_type == "application/pdf"


async def test_fetch_falls_back_to_browser_on_empty_body() -> None:
    class FakeBrowser:
        async def fetch(self, url: str) -> str | None:
            return "<html><body>rendered</body></html>"

    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_ALLOW)
        )
        router.get(f"{ORIGIN}/js-app").mock(
            return_value=httpx.Response(200, text="", headers={"content-type": "text/html"})
        )
        scraper = _client(browser=FakeBrowser())
        doc = await scraper.fetch(f"{ORIGIN}/js-app")
        await scraper.aclose()

    assert doc is not None
    assert "rendered" in doc.content


async def test_fetch_returns_none_when_no_browser_and_empty_body() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ORIGIN}/robots.txt").mock(
            return_value=httpx.Response(200, text=ROBOTS_ALLOW)
        )
        router.get(f"{ORIGIN}/empty").mock(
            return_value=httpx.Response(200, text="", headers={"content-type": "text/html"})
        )
        scraper = _client()
        assert await scraper.fetch(f"{ORIGIN}/empty") is None
        await scraper.aclose()


# --- Phase 11: SSRF scheme allowlist -----------------------------------------

@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "gopher://x", "javascript:alert(1)", ""],
)
async def test_fetch_rejects_non_http_schemes(url: str) -> None:
    scraper = _client()
    with respx.mock(assert_all_called=False):
        assert await scraper.fetch(url) is None
    await scraper.aclose()
