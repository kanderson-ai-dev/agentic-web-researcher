"""Tests for the document parser (HTML + PDF, fixture-based, offline)."""

import base64
from pathlib import Path

from app.core.schemas import RawDocument
from app.services.document_parser import parse_raw_document

FIXTURES = Path(__file__).parent / "fixtures"


def _doc(content: str = "", content_type: str = "text/html", b64: str | None = None) -> RawDocument:
    return RawDocument(
        url="https://example.com/page",
        content=content,
        content_bytes_b64=b64,
        content_type=content_type,
        sub_question_id="q1",
    )


def test_parse_html_extracts_title_and_main_text() -> None:
    html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")
    source = parse_raw_document(_doc(content=html))

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
    source = parse_raw_document(_doc(content="<html><body><script>x()</script></body></html>"))
    assert source is None


def test_parse_pdf_extracts_text() -> None:
    pdf = (FIXTURES / "sample.pdf").read_bytes()
    source = parse_raw_document(
        _doc(
            content_type="application/pdf",
            b64=base64.b64encode(pdf).decode("ascii"),
        )
    )

    assert source is not None
    assert "market grew 27 percent" in source.extracted_text


def test_parse_corrupt_pdf_returns_none() -> None:
    source = parse_raw_document(
        _doc(
            content_type="application/pdf",
            b64=base64.b64encode(b"not a pdf").decode("ascii"),
        )
    )
    assert source is None


def test_parse_invalid_b64_returns_none() -> None:
    source = parse_raw_document(
        _doc(content_type="application/pdf", b64="!!!not-base64!!!")
    )
    assert source is None
