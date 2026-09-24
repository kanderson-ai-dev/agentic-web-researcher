"""Tests for the document parser (HTML + PDF, fixture-based, offline)."""

from pathlib import Path

from app.core.schemas import RawDocument
from app.services.document_parser import parse_raw_document

FIXTURES = Path(__file__).parent / "fixtures"


def _doc(content_type: str = "text/html") -> RawDocument:
    return RawDocument(
        url="https://example.com/page",
        content_type=content_type,
        sub_question_id="q1",
    )


def test_parse_html_extracts_title_and_main_text() -> None:
    html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")
    source = parse_raw_document(_doc(), html.encode("utf-8"))

    assert source is not None
    assert source.title == "AI Adoption Report 2025"
    assert "grew 27%" in source.extracted_text
    assert "Customer support" in source.extracted_text
    # boilerplate is stripped
    assert "cookie banner" not in source.extracted_text
    assert "buy our newsletter" not in source.extracted_text
    assert "Copyright 2025" not in source.extracted_text
    assert "trackPageView" not in source.extracted_text
    assert source.sub_question_id == "q1"
    assert source.content_hash
    assert source.id.startswith("s-")


def test_parse_html_returns_none_for_empty_text() -> None:
    source = parse_raw_document(
        _doc(), b"<html><body><script>x()</script></body></html>"
    )
    assert source is None


def test_parse_missing_payload_returns_none() -> None:
    """Unresolved content_ref (e.g. process restart) degrades to None."""
    assert parse_raw_document(_doc(), None) is None


def test_parse_pdf_extracts_text() -> None:
    pdf = (FIXTURES / "sample.pdf").read_bytes()
    source = parse_raw_document(_doc(content_type="application/pdf"), pdf)

    assert source is not None
    assert "market grew 27 percent" in source.extracted_text


def test_parse_corrupt_pdf_returns_none() -> None:
    source = parse_raw_document(_doc(content_type="application/pdf"), b"not a pdf")
    assert source is None


def test_document_store_roundtrip() -> None:
    from app.services.document_store import DocumentStore

    store = DocumentStore()
    ref = store.put(b"<html>payload</html>")
    assert ref.startswith("d-")
    assert store.get(ref) == b"<html>payload</html>"
    assert store.get("d-missing") is None
    # same content -> same ref (content-addressed)
    assert store.put(b"<html>payload</html>") == ref
