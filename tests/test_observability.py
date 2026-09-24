"""Observability tests: usage/cost tracking, metrics endpoint, logging."""

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.logging import get_logger, redact_secrets
from app.core.schemas import JobStatus, ResearchJob, ResearchRequest
from app.graph.deps import GraphDeps
from app.main import create_app
from app.services.job_runner import JobRunner
from app.services.job_store import JobStore
from app.services.usage import (
    UsageTracker,
    record_usage,
    reset_tracker,
    set_tracker,
)


def test_usage_tracker_cost_math() -> None:
    tracker = UsageTracker()
    tracker.add("planner", 1_000_000, 0)  # 1M input tokens on gpt-4o-mini
    tracker.add("writer", 0, 1_000_000)  # 1M output tokens
    assert tracker.input_tokens == 1_000_000
    assert tracker.output_tokens == 1_000_000
    assert tracker.cost_usd("gpt-4o-mini") == pytest.approx(0.75)


def test_record_usage_noop_without_tracker() -> None:
    record_usage("planner", 10, 10)  # must not raise


def test_record_usage_scoped_to_context() -> None:
    tracker = UsageTracker()
    token = set_tracker(tracker)
    try:
        record_usage("critic", 100, 50)
    finally:
        reset_tracker(token)
    assert tracker.input_tokens == 100
    record_usage("critic", 999, 999)  # outside scope — ignored
    assert tracker.input_tokens == 100


async def test_runner_tracks_cost_and_timings(
    tmp_path: Path, stub_deps: GraphDeps
) -> None:
    store = JobStore(f"sqlite:///{tmp_path}/usage.sqlite")
    await store.init()
    runner = JobRunner(stub_deps, store, llm_model="gpt-4o-mini")

    job = ResearchJob(
        id="job-usage",
        request=ResearchRequest(topic="impact of the EU AI Act on startups"),
    )
    await runner.run(job)

    stored = await store.get(job.id)
    assert stored is not None
    assert stored.status is JobStatus.COMPLETED
    # Stub LLM: plan(400/150) + critique(800/200) + write(1200/400)
    assert stored.cost_usd == pytest.approx(
        (2400 * 0.15 + 750 * 0.60) / 1_000_000, rel=1e-3
    )
    assert stored.timings["total_seconds"] >= 0


async def test_runner_usage_isolated_per_job(
    tmp_path: Path, stub_deps: GraphDeps
) -> None:
    store = JobStore(f"sqlite:///{tmp_path}/iso.sqlite")
    await store.init()
    runner = JobRunner(stub_deps, store)

    request = ResearchRequest(topic="impact of the EU AI Act on startups")
    job_a = ResearchJob(id="job-a", request=request)
    job_b = ResearchJob(id="job-b", request=request)
    await runner.run(job_a)
    await runner.run(job_b)

    a = await store.get("job-a")
    b = await store.get("job-b")
    assert a is not None and b is not None
    assert a.cost_usd == b.cost_usd > 0  # per-job accounting, no accumulation


def test_metrics_endpoint_exposes_prometheus(
    tmp_path: Path, stub_deps: GraphDeps
) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path}/metrics.sqlite")
    with TestClient(create_app(settings=settings, deps=stub_deps)) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "research_jobs_total" in resp.text or "http_request" in resp.text


def test_metrics_counts_completed_jobs(tmp_path: Path, stub_deps: GraphDeps) -> None:
    deps = dataclasses.replace(stub_deps, require_human_review=False)
    settings = Settings(database_url=f"sqlite:///{tmp_path}/m2.sqlite")
    with TestClient(create_app(settings=settings, deps=deps)) as client:
        client.post("/api/v1/research", json={"topic": "impact of the EU AI Act"})
        import time

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            jobs = client.get("/api/v1/research").json()
            if jobs and jobs[0]["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        metrics = client.get("/metrics").text
    assert 'research_jobs_total{status="completed"}' in metrics


def test_redact_secrets_processor() -> None:
    event = {
        "event": "x",
        "openai_api_key": "sk-live",
        "access_token": "abc",
        "authorization": "Bearer x",
        "db_password": "pw",
        "job_id": "job-1",
    }
    out = redact_secrets(None, "info", event)
    assert out["openai_api_key"] == "***"
    assert out["access_token"] == "***"
    assert out["authorization"] == "***"
    assert out["db_password"] == "***"
    assert out["job_id"] == "job-1"


def test_get_logger_logs_without_error() -> None:
    logger = get_logger("test")
    logger.info("smoke", job_id="j1")
