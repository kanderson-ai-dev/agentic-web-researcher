"""Research job endpoints: submit, poll, and stream progress over SSE."""

import asyncio
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.rate_limit import rate_limiter
from app.core.schemas import (
    JobEvent,
    JobStatus,
    ResearchJob,
    ResearchRequest,
    ReviewDecision,
)
from app.core.security import require_auth
from app.graph.guardrails import screen_topic
from app.services.job_runner import JobRunner
from app.services.job_store import JobStore

router = APIRouter(
    prefix="/research", tags=["research"], dependencies=[Depends(require_auth)]
)


def _runner(request: Request) -> JobRunner:
    return request.app.state.job_runner  # type: ignore[no-any-return]


def _store(request: Request) -> JobStore:
    return request.app.state.job_store  # type: ignore[no-any-return]


@router.post("", response_model=ResearchJob, status_code=202)
async def submit_research(payload: ResearchRequest, request: Request) -> ResearchJob:
    """Enqueue a research job; returns the queued job record immediately."""
    settings = request.app.state.settings
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(
        "research",
        client_ip,
        limit=settings.rate_limit_research_per_minute,
        window_seconds=60.0,
    ):
        raise HTTPException(status_code=429, detail="too many requests")
    screened = screen_topic(payload.topic)
    if not screened.allowed:
        raise HTTPException(
            status_code=422, detail="topic rejected by guardrails"
        )
    job = ResearchJob(id=uuid.uuid4().hex, request=payload)
    await _store(request).upsert(job)
    asyncio.create_task(_runner(request).run(job))
    return job


@router.get("/{job_id}", response_model=ResearchJob)
async def get_research(job_id: str, request: Request) -> ResearchJob:
    """Poll a research job's status and (when finished) its result."""
    job = await _store(request).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


_TERMINAL = {JobStatus.COMPLETED, JobStatus.FAILED}


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


@router.get("/{job_id}/stream")
async def stream_research(job_id: str, request: Request) -> StreamingResponse:
    """Stream job progress as Server-Sent Events until it reaches a terminal state."""
    store = _store(request)
    runner = _runner(request)
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    queue = runner.subscribe(job_id)

    async def events() -> AsyncIterator[str]:
        try:
            # Snapshot for late subscribers / already-finished jobs.
            yield _sse(
                JobEvent(
                    job_id=job_id,
                    node="__job__",
                    status="completed" if job.status in _TERMINAL else "started",
                    detail=job.status.value,
                ).model_dump_json()
            )
            if job.status in _TERMINAL:
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield _sse(event.model_dump_json())
                if event.node == "__job__" and event.status in {"completed", "failed"}:
                    return
        finally:
            runner.unsubscribe(job_id, queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("", response_model=list[ResearchJob])
async def list_research(request: Request) -> list[ResearchJob]:
    """List recent research jobs (newest first)."""
    return await _store(request).list_recent()


@router.post("/{job_id}/review", response_model=ResearchJob, status_code=202)
async def review_research(
    job_id: str, decision: ReviewDecision, request: Request
) -> ResearchJob:
    """Submit the human decision for a job paused at the HITL gate."""
    if decision.action == "edit" and not decision.report:
        raise HTTPException(
            status_code=422, detail="report is required when action is 'edit'"
        )
    store = _store(request)
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status is not JobStatus.AWAITING_REVIEW:
        raise HTTPException(
            status_code=409, detail="job is not awaiting review"
        )
    asyncio.create_task(_runner(request).resume(job, decision.model_dump()))
    return job
