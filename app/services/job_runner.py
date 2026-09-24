"""Background execution of research jobs.

``JobRunner`` streams the compiled graph, publishes per-node progress
:class:`JobEvent`s to subscribers (consumed by the SSE endpoint), and persists
the final :class:`ResearchJob` state.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import structlog.contextvars
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.core.logging import get_logger
from app.core.metrics import JOB_COST, RESEARCH_DURATION, RESEARCH_JOBS
from app.core.schemas import JobEvent, JobStatus, ResearchJob
from app.graph.assembly import build_graph
from app.graph.deps import GraphDeps
from app.graph.state import initial_state
from app.services.job_store import JobStore
from app.services.usage import UsageTracker, reset_tracker, set_tracker

_JOB_NODE = "__job__"
_log = get_logger(__name__)


class JobRunner:
    """Runs research graphs in the background and fans out progress events."""

    def __init__(
        self,
        deps: GraphDeps,
        store: JobStore,
        *,
        max_critic_rounds: int = 2,
        checkpoint_db: str | None = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self._deps = deps
        self._graph = build_graph(deps) if checkpoint_db is None else None
        self._store = store
        self._max_critic_rounds = max_critic_rounds
        self._checkpoint_db = checkpoint_db
        self._llm_model = llm_model
        self._listeners: dict[str, set[asyncio.Queue[JobEvent]]] = {}

    # -- subscription (SSE) --------------------------------------------------

    def subscribe(self, job_id: str) -> asyncio.Queue[JobEvent]:
        queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        self._listeners.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[JobEvent]) -> None:
        listeners = self._listeners.get(job_id)
        if listeners is not None:
            listeners.discard(queue)
            if not listeners:
                self._listeners.pop(job_id, None)

    async def _publish(self, event: JobEvent) -> None:
        for queue in self._listeners.get(event.job_id, ()):
            queue.put_nowait(event)

    # -- execution -----------------------------------------------------------

    async def _finish(
        self,
        job: ResearchJob,
        status: JobStatus,
        error: str | None = None,
        *,
        elapsed: float | None = None,
    ) -> None:
        job.status = status
        job.error = error
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
        RESEARCH_JOBS.labels(status=status.value).inc()
        if elapsed is not None:
            RESEARCH_DURATION.observe(elapsed)
        JOB_COST.observe(job.cost_usd)
        _log.info(
            "job.finished",
            status=status.value,
            cost_usd=job.cost_usd,
            elapsed_seconds=elapsed,
        )
        await self._publish(
            JobEvent(
                job_id=job.id,
                node=_JOB_NODE,
                status="completed" if status is JobStatus.COMPLETED else "failed",
                detail=error or "",
            )
        )

    async def run(self, job: ResearchJob) -> None:
        """Execute the research graph, emitting a JobEvent per completed node."""
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
        await self._publish(JobEvent(job_id=job.id, node=_JOB_NODE, status="started"))

        state = initial_state(
            job.request, job_id=job.id, max_critic_rounds=self._max_critic_rounds
        )
        await self._drive(job, state)

    async def resume(self, job: ResearchJob, decision: dict[str, Any]) -> None:
        """Resume a job paused at the HITL gate with the reviewer's decision."""
        if self._checkpoint_db is None:
            await self._finish(
                job, JobStatus.FAILED, "resume unsupported: research job failed"
            )
            return
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
        await self._publish(JobEvent(job_id=job.id, node=_JOB_NODE, status="started"))
        await self._drive(job, Command(resume=decision))

    async def _drive(self, job: ResearchJob, graph_input: Any) -> None:
        """Stream the graph for ``graph_input`` and settle the job status."""
        started = time.monotonic()
        tracker = UsageTracker()
        token = set_tracker(tracker)
        structlog.contextvars.bind_contextvars(job_id=job.id)
        _log.info("job.drive_started")
        graph: Any = None
        config: dict[str, Any] = {}
        try:
            if self._checkpoint_db is not None:
                # Per-job thread_id: checkpoints persist every step for resume/HITL.
                Path(self._checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(
                    self._checkpoint_db
                ) as checkpointer:
                    graph = build_graph(self._deps, checkpointer=checkpointer)
                    config = {"configurable": {"thread_id": job.id}}
                    final = await self._execute(job, graph_input, graph, config)
                    interrupt_payload = await self._pending_interrupt(graph, config)
            else:
                assert self._graph is not None
                graph = self._graph
                final = await self._execute(job, graph_input, graph, config)
                interrupt_payload = None
        except Exception as exc:  # noqa: BLE001 — a failed job must not crash the app
            reset_tracker(token)
            _log.warning("job.failed", error=type(exc).__name__)
            await self._finish(
                job,
                JobStatus.FAILED,
                f"{type(exc).__name__}: research job failed",
                elapsed=time.monotonic() - started,
            )
            return

        job.cost_usd = round(job.cost_usd + tracker.cost_usd(self._llm_model), 6)
        job.timings["total_seconds"] = round(
            job.timings.get("total_seconds", 0.0) + (time.monotonic() - started), 3
        )
        reset_tracker(token)
        if interrupt_payload is not None:
            await self._await_review(job, interrupt_payload)
            return
        if final is None:
            await self._finish(
                job,
                JobStatus.FAILED,
                "empty run: research job failed",
                elapsed=job.timings["total_seconds"],
            )
            return
        job.report = final.get("report")
        job.citations = final.get("citations", [])
        job.sources = final.get("sources", [])
        job.sub_questions = final.get("sub_questions", [])
        await self._finish(
            job, JobStatus.COMPLETED, elapsed=job.timings["total_seconds"]
        )

    @staticmethod
    async def _pending_interrupt(graph: Any, config: dict[str, Any]) -> Any | None:
        """Return the interrupt payload when the run paused inside the graph."""
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return None
        payloads = [
            interrupt.value
            for task in snapshot.tasks
            for interrupt in task.interrupts
        ]
        return payloads[0] if payloads else {}

    async def _await_review(self, job: ResearchJob, payload: Any) -> None:
        """Park the job at AWAITING_REVIEW and notify SSE subscribers."""
        job.status = JobStatus.AWAITING_REVIEW
        job.updated_at = datetime.now(UTC)
        try:
            detail = json.dumps(payload, default=str)
        except TypeError:
            detail = "{}"
        await self._store.upsert(job)
        await self._publish(
            JobEvent(
                job_id=job.id,
                node=_JOB_NODE,
                status="awaiting_review",
                detail=detail,
            )
        )

    async def _execute(
        self,
        job: ResearchJob,
        state: Any,
        graph: Any,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Stream the graph, publishing per-node events; return the final state."""
        final: dict[str, Any] | None = None
        async for mode, chunk in graph.astream(
            state, stream_mode=["updates", "values"], config=config
        ):
            if mode == "updates":
                for node_name in chunk:
                    await self._publish(
                        JobEvent(job_id=job.id, node=node_name, status="completed")
                    )
            else:
                final = cast(dict[str, Any], chunk)
        return final
