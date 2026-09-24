"""Frontend static mounts: landing + console served, API never shadowed."""


import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.graph.deps import GraphDeps
from app.main import create_app


@pytest.fixture
def client(tmp_path, stub_deps: GraphDeps) -> TestClient:
    settings = Settings(database_url=f"sqlite:///{tmp_path}/fe.sqlite")
    with TestClient(create_app(settings=settings, deps=stub_deps)) as c:
        yield c


def test_landing_served_at_root(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Autonomous Web Research Agent" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_console_served(client: TestClient) -> None:
    resp = client.get("/console/")
    assert resp.status_code == 200
    assert "Research Console" in resp.text


def test_console_redirects_to_trailing_slash(client: TestClient) -> None:
    resp = client.get("/console", follow_redirects=False)
    assert resp.status_code in {301, 302, 307, 308}
    assert resp.headers["location"].endswith("/console/")


def test_console_assets_served(client: TestClient) -> None:
    assert client.get("/console/styles.css").status_code == 200
    assert client.get("/console/app.js").status_code == 200


def test_api_not_shadowed(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/api/v1/research").status_code in {200, 401}
    assert client.get("/docs").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_missing_static_404(client: TestClient) -> None:
    assert client.get("/nonexistent.js").status_code == 404
