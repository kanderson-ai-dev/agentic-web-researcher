"""End-to-end tests for the multi-agent research graph (stubbed workers)."""

import dataclasses

from app.core.schemas import ResearchRequest, SubQuestionStatus
from app.graph.assembly import build_graph
from app.graph.deps import GraphDeps
from app.graph.state import initial_state
from app.services.llm_client import StubLLMClient


async def test_graph_runs_end_to_end(stub_deps: GraphDeps, research_request) -> None:
    graph = build_graph(stub_deps)
    final = await graph.ainvoke(initial_state(research_request, job_id="job-1"))

    assert final["report"]
    assert final["critic_rounds"] == 1
    assert len(final["sub_questions"]) == 3
    assert all(
        sq.status is SubQuestionStatus.ANSWERED for sq in final["sub_questions"]
    )
    # 3 sub-questions x 2 search results each
    assert len(final["search_results"]) == 6
    assert len(final["raw_documents"]) == 6
    assert len(final["sources"]) == 6
    assert len(final["citations"]) == 6
    assert final["errors"] == []


async def test_critic_replan_loop_is_bounded(
    stub_deps: GraphDeps, research_request: ResearchRequest
) -> None:
    deps = dataclasses.replace(stub_deps, llm=StubLLMClient(force_needs_more=True))
    graph = build_graph(deps)
    final = await graph.ainvoke(
        initial_state(research_request, job_id="job-2", max_critic_rounds=2)
    )

    # Even though the critic always demands more research, the loop stops at
    # max_critic_rounds and the writer still produces a report.
    assert final["critic_rounds"] == 2
    assert final["report"]


async def test_scrape_failure_is_recorded_not_fatal(
    stub_deps: GraphDeps, research_request: ResearchRequest
) -> None:
    async def failing_scrape(result):  # type: ignore[no-untyped-def]
        if result.url.endswith("/0"):
            raise RuntimeError("simulated fetch failure")
        return await stub_deps.scrape(result)

    deps = dataclasses.replace(stub_deps, scrape=failing_scrape)
    graph = build_graph(deps)
    final = await graph.ainvoke(initial_state(research_request, job_id="job-3"))

    # 3 failures (one per sub-question) are recorded; the rest still succeed.
    assert len(final["errors"]) == 3
    assert len(final["sources"]) == 3
    assert final["report"]


async def test_empty_search_still_produces_report(
    stub_deps: GraphDeps, research_request: ResearchRequest
) -> None:
    async def empty_search(sub_question):  # type: ignore[no-untyped-def]
        return []

    deps = dataclasses.replace(stub_deps, search=empty_search)
    graph = build_graph(deps)
    final = await graph.ainvoke(initial_state(research_request, job_id="job-4"))

    assert final["sources"] == []
    assert final["report"]
    assert all(
        sq.status is SubQuestionStatus.UNRESOLVED for sq in final["sub_questions"]
    )


async def test_replan_adds_new_sub_questions(
    stub_deps: GraphDeps, research_request: ResearchRequest
) -> None:
    """One forced re-plan round enqueues exactly the critic's missing aspects."""

    class OneShotCritic(StubLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        async def critique(self, **kwargs):  # type: ignore[no-untyped-def]
            self._calls += 1
            if self._calls == 1:
                from app.core.schemas import CriticVerdict

                return CriticVerdict(
                    coverage_score=0.5,
                    citation_support_score=0.5,
                    missing_aspects=["enforcement timeline", "enforcement timeline"],
                    needs_more_research=True,
                    feedback="one more round",
                )
            return await super().critique(**kwargs)

    deps = dataclasses.replace(stub_deps, llm=OneShotCritic())
    graph = build_graph(deps)
    final = await graph.ainvoke(
        initial_state(research_request, job_id="job-5", max_critic_rounds=3)
    )

    assert final["critic_rounds"] == 2
    # Dedup: the repeated missing aspect produces a single extra sub-question.
    new_questions = [sq for sq in final["sub_questions"] if sq.id.startswith("r")]
    assert len(new_questions) == 1
    assert new_questions[0].question == "enforcement timeline"
