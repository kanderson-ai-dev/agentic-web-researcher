"""Tests for the web search client (mocked HTTP — no network)."""

from pathlib import Path

import httpx
import pytest
import respx

from app.core.config import Settings
from app.services.search_client import NullSearchClient, TavilySearchClient

BASE = "https://api.tavily.com"


def _payload(n: int = 2) -> dict:
    return {
        "results": [
            {
                "url": f"https://example.com/{i}",
                "title": f"Result {i}",
                "content": f"snippet {i}",
                "score": 0.9 - i * 0.1,
            }
            for i in range(n)
        ]
    }


async def test_search_parses_results() -> None:
    with respx.mock:
        respx.post(f"{BASE}/search").mock(
            return_value=httpx.Response(200, json=_payload(2))
        )
        client = TavilySearchClient(api_key="test-key", base_url=BASE)
        results = await client.search("AI regulation", max_results=5)
        await client.aclose()

    assert len(results) == 2
    assert results[0].url == "https://example.com/0"
    assert results[0].title == "Result 0"
    assert results[0].score == pytest.approx(0.9)


async def test_search_sends_api_key_and_max_results() -> None:
    with respx.mock:
        route = respx.post(f"{BASE}/search").mock(
            return_value=httpx.Response(200, json=_payload(1))
        )
        client = TavilySearchClient(api_key="secret-key", base_url=BASE)
        await client.search("query", max_results=4)
        await client.aclose()

    sent = route.calls.last.request
    import json

    body = json.loads(sent.content)
    assert body["api_key"] == "secret-key"
    assert body["max_results"] == 4


async def test_search_caches_identical_queries() -> None:
    with respx.mock:
        route = respx.post(f"{BASE}/search").mock(
            return_value=httpx.Response(200, json=_payload(1))
        )
        client = TavilySearchClient(api_key="k", base_url=BASE)
        first = await client.search("same query", max_results=3)
        second = await client.search("same query", max_results=3)
        await client.aclose()

    assert route.call_count == 1
    assert first == second


async def test_search_retries_on_server_error() -> None:
    with respx.mock:
        route = respx.post(f"{BASE}/search").mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(200, json=_payload(1)),
            ]
        )
        client = TavilySearchClient(api_key="k", base_url=BASE)
        results = await client.search("retry me", max_results=3)
        await client.aclose()

    assert route.call_count == 2
    assert len(results) == 1


async def test_search_filters_results_without_url() -> None:
    with respx.mock:
        respx.post(f"{BASE}/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"url": "", "title": "no url"},
                        {"url": "https://ok.example.com", "title": "ok"},
                    ]
                },
            )
        )
        client = TavilySearchClient(api_key="k", base_url=BASE)
        results = await client.search("q", max_results=3)
        await client.aclose()

    assert len(results) == 1
    assert results[0].url == "https://ok.example.com"


async def test_null_client_returns_empty() -> None:
    client = NullSearchClient()
    assert await client.search("anything", max_results=5) == []


# --- DuckDuckGo free fallback -------------------------------------------------

DDG_HTML = (Path(__file__).parent / "fixtures" / "ddg_results.html").read_text(
    encoding="utf-8"
)


async def test_duckduckgo_parses_results_and_dedupes() -> None:
    from app.services.search_client import DuckDuckGoSearchClient

    with respx.mock(assert_all_called=True) as router:
        router.get("https://html.duckduckgo.com/html/").mock(
            return_value=httpx.Response(200, text=DDG_HTML)
        )
        client = DuckDuckGoSearchClient(delay_seconds=0)
        results = await client.search("eu ai act startups", max_results=10)
        await client.aclose()

    urls = [r.url for r in results]
    assert urls == [
        "https://example.com/ai-act",
        "https://example.org/startups",
        "https://direct.example.net/page",
    ]
    assert results[0].title == "EU AI Act overview"
    assert "regulates artificial intelligence" in results[0].snippet


async def test_duckduckgo_respects_max_results() -> None:
    from app.services.search_client import DuckDuckGoSearchClient

    with respx.mock() as router:
        router.get("https://html.duckduckgo.com/html/").mock(
            return_value=httpx.Response(200, text=DDG_HTML)
        )
        client = DuckDuckGoSearchClient(delay_seconds=0)
        results = await client.search("q", max_results=2)
        await client.aclose()
    assert len(results) == 2


async def test_duckduckgo_degrades_on_http_error() -> None:
    from app.services.search_client import DuckDuckGoSearchClient

    with respx.mock() as router:
        router.get("https://html.duckduckgo.com/html/").mock(
            return_value=httpx.Response(403)
        )
        client = DuckDuckGoSearchClient(delay_seconds=0)
        assert await client.search("q", max_results=5) == []
        await client.aclose()


async def test_factory_auto_provider_uses_duckduckgo_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto + no key → the deps search callable delegates to DuckDuckGo."""
    from app.core.schemas import SubQuestion
    from app.services.factory import build_graph_deps
    from app.services.search_client import DuckDuckGoSearchClient

    called: dict[str, str] = {}

    async def fake_search(
        self: DuckDuckGoSearchClient, query: str, *, max_results: int
    ) -> list:
        called["query"] = query
        return []

    monkeypatch.setattr(DuckDuckGoSearchClient, "search", fake_search)
    deps = build_graph_deps(Settings())
    await deps.search(SubQuestion(id="sq1", question="q1"))
    assert called["query"] == "q1"


async def test_factory_none_provider_disables_search() -> None:
    from app.core.schemas import SubQuestion
    from app.services.factory import build_graph_deps

    deps = build_graph_deps(Settings(search_provider="none"))
    results = await deps.search(SubQuestion(id="sq1", question="q1"))
    assert results == []
