"""EDD evaluation runner: execute the pipeline per dataset topic, score it,
and write a versioned scorecard. Exits non-zero when any gate fails — this is
the CI quality gate.

Usage::

    uv run python -m evaluation.run_eval
"""

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.graph.assembly import build_graph
from app.graph.state import initial_state
from evaluation.evaluators import (
    citation_support_rate,
    source_precision,
    topic_coverage,
)
from evaluation.harness import build_eval_deps, load_dataset, topic_request

SCORECARD_PATH = Path(__file__).parent / "scorecards" / "latest.json"

# Quality gates (mirrors PLAN.md "Objetivo del proyecto").
THRESHOLDS = {
    "citation_support_rate": 0.90,
    "source_precision": 0.80,
    "topic_coverage": 0.85,
}


async def evaluate_topic(topic_spec: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline on one dataset topic and compute its metrics."""
    deps = build_eval_deps(topic_spec)
    graph = build_graph(deps)
    state = initial_state(topic_request(topic_spec), job_id=f"eval-{topic_spec['id']}")
    final = await graph.ainvoke(state)

    kept = len(final.get("citations", []))
    dropped = final.get("dropped_citations", 0)
    return {
        "id": topic_spec["id"],
        "topic": topic_spec["topic"],
        "metrics": {
            "citation_support_rate": citation_support_rate(kept, dropped),
            "source_precision": source_precision(
                final.get("sources", []), topic_spec["source_relevance_terms"]
            ),
            "topic_coverage": topic_coverage(
                final.get("report"), topic_spec["expected_aspects"]
            ),
        },
        "counts": {
            "sources": len(final.get("sources", [])),
            "citations_kept": kept,
            "citations_dropped": dropped,
        },
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each metric across topics."""
    keys = results[0]["metrics"].keys() if results else []
    return {
        key: round(sum(r["metrics"][key] for r in results) / len(results), 4)
        for key in keys
    }


async def main() -> int:
    dataset = load_dataset()
    results = [await evaluate_topic(t) for t in dataset["topics"]]
    summary = aggregate(results)

    gates = {
        metric: {
            "value": summary.get(metric, 0.0),
            "threshold": threshold,
            "passed": summary.get(metric, 0.0) >= threshold,
        }
        for metric, threshold in THRESHOLDS.items()
    }
    passed = all(g["passed"] for g in gates.values())

    scorecard = {
        "dataset_version": dataset["version"],
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "offline-deterministic",
        "results": results,
        "aggregate": summary,
        "gates": gates,
        "passed": passed,
    }
    SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORECARD_PATH.write_text(
        json.dumps(scorecard, indent=2) + "\n", encoding="utf-8"
    )

    for metric, gate in gates.items():
        mark = "PASS" if gate["passed"] else "FAIL"
        print(f"[{mark}] {metric}: {gate['value']:.3f} >= {gate['threshold']}")
    print(f"scorecard written to {SCORECARD_PATH}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
