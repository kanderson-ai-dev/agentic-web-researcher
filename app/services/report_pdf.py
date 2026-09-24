"""Render a completed research job to a downloadable PDF.

Uses fpdf2 core fonts (latin-1): markdown is rendered structurally (headings,
bullets, paragraphs) and text is normalized to latin-1 so no font assets are
required.
"""

import re
import unicodedata
from typing import Any

from fpdf import FPDF

from app.core.schemas import ResearchJob

_REPLACEMENTS = {
    "—": "-", "–": "-", "―": "-",
    "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'", "‚": "'",
    "•": "*", "◦": "-", "·": "-",
    "…": "...",
    "→": "->", "←": "<-", "↔": "<->",
    "≥": ">=", "≤": "<=", "≠": "!=",
    "×": "x",
}


_CELL_KW: dict[str, Any] = {
    "wrapmode": "CHAR",
    "new_x": "LMARGIN",
    "new_y": "NEXT",
}


def _latin1(text: str) -> str:
    """Normalize text to latin-1 so core PDF fonts can render it."""
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("latin-1", "replace").decode("latin-1")


def _strip_inline_md(text: str) -> str:
    """Remove inline markdown markers for plain-PDF rendering."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text


def build_pdf(job: ResearchJob) -> bytes:
    """Render the job report + sources to PDF bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    topic = job.request.topic
    pdf.set_font("helvetica", "B", 16)
    pdf.multi_cell(pdf.epw, 9, _latin1(f"Research Report: {topic}"), **_CELL_KW)
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(
        pdf.epw,
        5,
        _latin1(
            f"job {job.id} | depth {job.request.depth} | "
            f"{len(job.sources)} sources | est. cost ${job.cost_usd:.6f}"
        ),
        **_CELL_KW,
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for raw_line in (job.report or "").splitlines():
        line = raw_line.rstrip()
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("### "):
            pdf.set_font("helvetica", "B", 12)
            pdf.multi_cell(pdf.epw, 7, _latin1(_strip_inline_md(line[4:])), **_CELL_KW)
        elif line.startswith("## "):
            pdf.set_font("helvetica", "B", 13)
            pdf.multi_cell(pdf.epw, 8, _latin1(_strip_inline_md(line[3:])), **_CELL_KW)
        elif line.startswith("# "):
            pdf.set_font("helvetica", "B", 15)
            pdf.multi_cell(pdf.epw, 9, _latin1(_strip_inline_md(line[2:])), **_CELL_KW)
        elif line.startswith("- "):
            pdf.set_font("helvetica", "", 10)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(pdf.epw, 6, _latin1(f"- {_strip_inline_md(line[2:])}"), **_CELL_KW)
        else:
            pdf.set_font("helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 6, _latin1(_strip_inline_md(line)), **_CELL_KW)
    if job.sources:
        pdf.ln(4)
        pdf.set_font("helvetica", "B", 13)
        pdf.multi_cell(pdf.epw, 8, "Sources", **_CELL_KW)
        pdf.set_font("helvetica", "", 9)
        for source in job.sources:
            title = source.title or source.url
            pdf.multi_cell(pdf.epw, 5, _latin1(f"- {title}\n  {source.url}"), **_CELL_KW)
    return bytes(pdf.output())
