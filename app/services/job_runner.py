"""Background execution of research jobs.

``JobRunner`` streams the compiled graph, publishes per-node progress
:class:`JobEvent`s to subscribers (consumed by the SSE endpoint), and persists
the final :class:`ResearchJob` state.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.core.schemas import JobEvent, JobStatus, ResearchJob
from app.graph.assembly import build_graph
from app.graph.deps import GraphDeps
from app.graph.state import initial_state
from app.services.job_store import JobStore

_JOB_NODE = "__job__"


class JobRunner:
    """Runs research graphs in the background and fans out progress events."""

    def __init__(
        self,
        deps: GraphDeps,
        store: JobStore,
        *,
        max_critic_rounds: int = 2,
        checkpoint_db: str | None = None,
    ) -> None:
        self._deps = deps
        self._graph = build_graph(deps) if checkpoint_db is None else None
        self._store = store
        self._max_critic_rounds = max_critic_rounds
        self._checkpoint_db = checkpoint_db
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
        self, job: ResearchJob, status: JobStatus, error: str | None = None
    ) -> None:
        job.status = status
        job.error = error
        job.updated_at = datetime.now(UTC)
        await self._store.upsert(job)
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
        try:
            if self._checkpoint_db is not None:
                # Per-job thread_id: checkpoints persist every step for resume/HITL.
                Path(self._checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
                async with AsyncSqliteSaver.from_conn_string(
                    self._checkpoint_db
                ) as checkpointer:
                    graph = build_graph(self._deps, checkpointer=checkpointer)
                    final = await self._execute(
                        job, state, graph, {"configurable": {"thread_id": job.id}}
                    )
            else:
                assert self._graph is not None
                final = await self._execute(job, state, self._graph, {})
        except Exception as exc:  # noqa: BLE001 — a failed job must not crash the app
            await self._finish(
                job, JobStatus.FAILED, f"{type(exc).__name__}: research job failed"
            )
            return

        if final is None:
            await self._finish(job, JobStatus.FAILED, "empty run: research job failed")
            return
        job.report = final.get("report")
        job.citations = final.get("citations", [])
        job.sources = final.get("sources", [])
        job.sub_questions = final.get("sub_questions", [])
        await self._finish(job, JobStatus.COMPLETED)

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
