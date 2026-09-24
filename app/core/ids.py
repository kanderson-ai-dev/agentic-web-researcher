"""Stable identifier helpers shared across layers."""

import hashlib


def source_id_for(url: str, content: str) -> str:
    """Stable short id for a source derived from its URL and content."""
    digest = hashlib.sha256(f"{url}|{content}".encode()).hexdigest()[:12]
    return f"s-{digest}"
