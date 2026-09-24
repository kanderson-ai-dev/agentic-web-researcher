"""SQLite-backed persistence for research jobs.

The full :class:`ResearchJob` is serialized as JSON — the schema already
covers everything we need to serve ``GET /research/{id}``.
"""

from pathlib import Path

import aiosqlite

from app.core.schemas import ResearchJob

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class JobStore:
    """Async SQLite store, one short-lived connection per operation."""

    def __init__(self, database_url: str) -> None:
        self._path = database_url.removeprefix("sqlite:///")

    async def init(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._path) as db:
            await db.execute(_CREATE_TABLE)
            await db.commit()

    async def upsert(self, job: ResearchJob) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO jobs (id, status, payload, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.status.value,
                    job.model_dump_json(),
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            await db.commit()

    async def get(self, job_id: str) -> ResearchJob | None:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return ResearchJob.model_validate_json(row[0])

    async def list_recent(self, *, limit: int = 50) -> list[ResearchJob]:
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                "SELECT payload FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
        return [ResearchJob.model_validate_json(row[0]) for row in rows]
