"""FastAPI application entrypoint for the Autonomous Web Research Agent."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.routes.research import router as research_router
from app.core.config import Settings, get_settings
from app.graph.deps import GraphDeps
from app.services.factory import build_graph_deps
from app.services.job_runner import JobRunner
from app.services.job_store import JobStore


def create_app(
    *, settings: Settings | None = None, deps: GraphDeps | None = None
) -> FastAPI:
    """Application factory.

    ``settings``/``deps`` are injectable so tests can run fully offline with
    deterministic doubles and a temporary database.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_settings = settings or get_settings()
        store = JobStore(resolved_settings.database_url)
        await store.init()
        app.state.job_store = store
        app.state.job_runner = JobRunner(
            deps or build_graph_deps(resolved_settings),
            store,
            max_critic_rounds=resolved_settings.max_critic_rounds,
        )
        yield

    app = FastAPI(
        title="Autonomous Web Research Agent",
        description=(
            "Hybrid multi-agent research microservice "
            "(planner -> workers -> critic -> writer)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(research_router, prefix="/api/v1")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    return app


app = create_app()
