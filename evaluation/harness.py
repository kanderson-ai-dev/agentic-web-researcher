"""Offline evaluation harness: fixture-backed deps + deterministic eval LLM.

The ``EvalLLMClient`` is corpus-aware by construction (it plans one
sub-question per corpus section and writes sections quoting real source
text) — the offline scorecard therefore measures *pipeline* behaviour
(retrieval → parsing → guardrails → citation verification) deterministically.
Live-model quality is measured by the same metrics when real credentials run
``run_eval`` outside CI.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.schemas import (
    Citation,
    CriticVerdict,
    RawDocument,
    ResearchRequest,
    SearchResult,
    Source,
    SubQuestion,
)
from app.graph.deps import GraphDeps
from app.services.document_parser import parse_raw_document
from app.services.document_store import DocumentStore

CORPUS_DIR = Path(__file__).parent / "corpus"
DATASET_PATH = Path(__file__).parent / "dataset" / "topics.json"


class EvalLLMClient:
    """Deterministic, corpus-aware LLM double for offline evaluation."""

    def __init__(self, aspects: Sequence[str]) -> None:
        self._aspects = list(aspects)

    async def plan(
        self, *, topic: str, max_questions: int, language: str
    ) -> list[SubQuestion]:
        del topic, language
        return [
            SubQuestion(id=f"q{i + 1}", question=f"What are the {aspect}?")
            for i, aspect in enumerate(self._aspects[:max_questions])
        ]

    async def critique(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
    ) -> CriticVerdict:
        del topic, sub_questions
        has_sources = bool(sources)
        return CriticVerdict(
            coverage_score=1.0 if has_sources else 0.0,
            citation_support_score=1.0 if has_sources else 0.0,
            missing_aspects=[],
            needs_more_research=False,
            feedback="eval: satisfied",
        )

    async def write(
        self,
        *,
        topic: str,
        sub_questions: Sequence[SubQuestion],
        sources: Sequence[Source],
        language: str,
    ) -> tuple[str, list[Citation]]:
        del language
        lines = [f"# Research report: {topic}", ""]
        citations: list[Citation] = []
        for sq in sub_questions:
            lines.append(f"## {sq.question}")
            related = [s for s in sources if s.sub_question_id == sq.id] or list(sources)
            for source in related[:2]:
                quote = source.extracted_text[:150].strip()
                lines.append(f"- {quote} ({source.url})")
                citations.append(
                    Citation(
                        claim=f"{sq.question}: supported by {source.url}",
                        source_id=source.id,
                        quote=quote,
                    )
                )
        return "\n".join(lines), citations


def load_dataset() -> dict[str, Any]:
    """Load the versioned evaluation dataset."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def corpus_url(filename: str) -> str:
    """Stable fixture URL for a corpus document."""
    return f"https://eval.local/{filename}"


def build_eval_deps(topic_spec: dict[str, Any]) -> GraphDeps:
    """Build graph deps that serve only the topic's fixture corpus."""
    corpus_files: list[str] = topic_spec["corpus"]

    async def search(sub_question: SubQuestion) -> list[SearchResult]:
        return [
            SearchResult(
                url=corpus_url(name),
                title=name.replace("_", " ").removesuffix(".html"),
                snippet=f"corpus document {name}",
                sub_question_id=sub_question.id,
            )
            for name in corpus_files
        ]

    document_store = DocumentStore()

    async def scrape(result: SearchResult) -> RawDocument | None:
        name = result.url.rsplit("/", 1)[-1]
        path = CORPUS_DIR / name
        if not path.exists():
            return None
        payload = path.read_bytes()
        return RawDocument(
            url=result.url,
            content_ref=document_store.put(payload),
            content_type="text/html",
            byte_size=len(payload),
            sub_question_id=result.sub_question_id,
        )

    async def parse(document: RawDocument) -> Source | None:
        return parse_raw_document(
            document, document_store.get(document.content_ref)
        )

    return GraphDeps(
        llm=EvalLLMClient(topic_spec["expected_aspects"]),
        search=search,
        scrape=scrape,
        parse=parse,
        require_human_review=False,
    )


def topic_request(topic_spec: dict[str, Any]) -> ResearchRequest:
    """Build the job request for a dataset topic."""
    return ResearchRequest(topic=topic_spec["topic"])
