"""Tests for the prompt templates module."""

from app.core.schemas import Source, SubQuestion
from app.graph import prompts


def _source(source_id: str = "s1", text: str = "x" * 100) -> Source:
    return Source(
        id=source_id,
        url=f"https://example.com/{source_id}",
        title=f"Title {source_id}",
        content_hash="h",
        extracted_text=text,
    )


def test_system_prompts_are_role_specific() -> None:
    assert "planner" in prompts.planner_system(max_questions=5, language="en")
    assert "critic" in prompts.critic_system()
    writer = prompts.writer_system(language="en")
    assert "writer" in writer
    assert "verbatim" in writer  # citation discipline is part of the contract


def test_planner_system_binds_limits_and_language() -> None:
    p = prompts.planner_system(max_questions=3, language="es")
    assert "3 questions" in p and "es" in p


def test_format_evidence_labels_ids_and_truncates() -> None:
    evidence = prompts.format_evidence([_source(text="a" * 5000)], per_source_chars=100)
    assert evidence.startswith("[s1] Title s1 (https://example.com/s1)")
    assert len(evidence.split("\n")[1]) == 100


def test_research_prompt_includes_topic_questions_and_fallback() -> None:
    out = prompts.research_prompt(
        topic="AI Act",
        sub_questions=[SubQuestion(id="q1", question="What scope?")],
        evidence="",
    )
    assert "Topic: AI Act" in out
    assert "- What scope?" in out
    assert "(none)" in out
