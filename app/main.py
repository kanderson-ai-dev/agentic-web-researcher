"""FastAPI application entrypoint for the Autonomous Web Research Agent."""

from fastapi import FastAPI

app = FastAPI(
    title="Autonomous Web Research Agent",
    description=(
        "Hybrid multi-agent research microservice (planner -> workers -> critic -> writer)."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
