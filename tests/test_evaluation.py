"""EDD tests: evaluators math + offline harness + scorecard generation."""

import json
from pathlib import Path

import pytest

from app.core.schemas import Source
from evaluation.evaluators import (
    citation_support_rate,
    source_precision,
    topic_coverage,
)
from evaluation.harness import build_eval_deps, load_dataset, topic_request
from evaluation.run_eval import evaluate_topic


def test_citation_support_rate() -> None:
    assert citation_support_rate(kept=9, dropped=1) == pytest.approx(0.9)
    assert citation_support_rate(kept=0, dropped=0) == 0.0
    assert citation_support_rate(kept=0, dropped=5) == 0.0


def test_topic_coverage_case_insensitive() -> None:
    report = "# Report\nThe SCOPE is broad. Obligations apply."
    assert topic_coverage(report, ["scope", "obligations", "penalties"]) == (
        pytest.approx(2 / 3)
    )
    assert topic_coverage(None, ["scope"]) == 0.0
    assert topic_coverage(report, []) == 0.0


def test_source_precision() -> None:
    sources = [
        Source(
            id="s1",
            url="https://a.example",
            content_hash="a" * 16,
            extracted_text="The AI Act regulates startups",
            sub_question_id="q1",
        ),
        Source(
            id="s2",
            url="https://b.example",
            content_hash="b" * 16,
            extracted_text="unrelated content entirely",
            sub_question_id="q1",
        ),
    ]
    assert source_precision(sources, ["ai act"]) == pytest.approx(0.5)
    assert source_precision([], ["x"]) == 0.0


def test_dataset_is_versioned_and_valid() -> None:
    dataset = load_dataset()
    assert dataset["version"]
    assert len(dataset["topics"]) >= 2
    for topic in dataset["topics"]:
        assert topic["expected_aspects"]
        assert topic["corpus"]
        for name in topic["corpus"]:
            assert (Path("evaluation/corpus") / name).exists()


async def test_evaluate_topic_end_to_end() -> None:
    dataset = load_dataset()
    result = await evaluate_topic(dataset["topics"][0])
    metrics = result["metrics"]
    assert 0.0 <= metrics["citation_support_rate"] <= 1.0
    assert 0.0 <= metrics["source_precision"] <= 1.0
    assert 0.0 <= metrics["topic_coverage"] <= 1.0
    # The offline pipeline should score perfectly on its own corpus.
    assert metrics["topic_coverage"] == 1.0
    assert metrics["citation_support_rate"] == 1.0
    assert result["counts"]["sources"] > 0


def test_eval_deps_use_fixture_corpus_only() -> None:
    dataset = load_dataset()
    deps = build_eval_deps(dataset["topics"][0])
    assert deps.require_human_review is False
    request = topic_request(dataset["topics"][0])
    assert request.topic == dataset["topics"][0]["topic"]


async def test_run_eval_writes_scorecard(tmp_path, monkeypatch) -> None:
    import evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "SCORECARD_PATH", tmp_path / "latest.json")
    exit_code = await run_eval.main()
    assert exit_code == 0
    scorecard = json.loads((tmp_path / "latest.json").read_text())
    assert scorecard["passed"] is True
    assert scorecard["aggregate"]["citation_support_rate"] == 1.0
    assert len(scorecard["results"]) == len(load_dataset()["topics"])
