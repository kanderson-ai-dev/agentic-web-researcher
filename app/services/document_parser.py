"""Parse raw fetched documents (HTML / PDF) into clean :class:`Source` objects.

The parser only performs mechanical extraction — no LLM calls. Injection-style
content is left untouched here; the ``document_worker`` node applies the
sanitization guardrail afterwards.
"""

import hashlib
import io
import re
from datetime import UTC, datetime

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.core.ids import source_id_for
from app.core.schemas import RawDocument, Source

_WS_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

_STRIP_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


def parse_html(url: str, html: str, sub_question_id: str) -> Source | None:
    """Extract main text and title from an HTML page."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    title = None
    h1 = soup.find("h1")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif h1 is not None:
        title = h1.get_text(strip=True)

    text = soup.get_text(separator="\n")
    text = _BLANK_LINES_RE.sub("\n\n", _WS_RE.sub(" ", text)).strip()
    if not text:
        return None
    return Source(
        id=source_id_for(url, text),
        url=url,
        title=title,
        fetched_at=datetime.now(UTC),
        content_hash=_hash(text),
        extracted_text=text,
        sub_question_id=sub_question_id,
    )


def parse_pdf(url: str, data: bytes, sub_question_id: str) -> Source | None:
    """Extract text and title metadata from a PDF payload."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception:  # noqa: BLE001 — corrupt/unparseable PDFs degrade to None
        return None
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = _BLANK_LINES_RE.sub("\n\n", _WS_RE.sub(" ", "\n".join(pages))).strip()
    if not text:
        return None
    title = None
    if reader.metadata and reader.metadata.title:
        title = reader.metadata.title.strip()
    return Source(
        id=source_id_for(url, text),
        url=url,
        title=title,
        fetched_at=datetime.now(UTC),
        content_hash=_hash(text),
        extracted_text=text,
        sub_question_id=sub_question_id,
    )


def parse_raw_document(document: RawDocument, payload: bytes | None) -> Source | None:
    """Dispatch to the right parser based on content type / payload.

    ``payload`` is the raw fetched body resolved from the document store —
    ``None`` when the reference cannot be resolved (e.g. process restart).
    """
    if payload is None:
        return None
    if "pdf" in document.content_type.lower():
        return parse_pdf(document.url, payload, document.sub_question_id)
    return parse_html(
        document.url,
        payload.decode("utf-8", errors="replace"),
        document.sub_question_id,
    )
