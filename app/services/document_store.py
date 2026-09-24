"""Process-local payload store for scraped documents.

Raw fetched payloads (HTML bodies, PDF binaries) can be megabytes each.
LangGraph serializes node inputs/outputs into SQLite checkpoints and
LangSmith traces, so keeping payloads inside ``RawDocument`` state records
would blow through ingest limits and bloat checkpoints. ``RawDocument``
therefore carries only metadata plus a ``content_ref`` resolved through this
store.

The store is process-local and ephemeral: payloads live only for the
duration of a run. If a process restarts mid-run, unresolved references
degrade to ``None`` and the affected source is skipped — the graph still
completes.
"""

import hashlib


class DocumentStore:
    """Content-addressed store mapping ``content_ref`` → raw payload bytes."""

    def __init__(self) -> None:
        self._payloads: dict[str, bytes] = {}

    def put(self, payload: bytes) -> str:
        """Store ``payload`` and return its content-addressed reference."""
        ref = f"d-{hashlib.sha256(payload).hexdigest()[:16]}"
        self._payloads[ref] = payload
        return ref

    def get(self, ref: str) -> bytes | None:
        """Return the payload for ``ref``, or ``None`` if unknown."""
        return self._payloads.get(ref)
