"""Graph assembly: planner -> workers -> critic -> writer -> assembler.

Fan-out uses LangGraph ``Send`` so each sub-question / URL / document is
processed by an independent worker branch; reducers merge their output back
into the shared state. The critic/re-plan loop is bounded by
``max_critic_rounds`` — termination is guaranteed by construction.
"""

from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.graph.deps import GraphDeps
from app.graph.nodes import (
    make_critic,
    make_document_worker,
    make_human_review,
    make_input_guardrail,
    make_output_guardrail,
    make_planner,
    make_rejection_output,
    make_replan,
    make_report_assembler,
    make_scrape_worker,
    make_search_worker,
    make_writer,
)
from app.graph.state import ResearchState


def _route_to_search(state: ResearchState) -> list[Send] | str:
    """Fan out one search_worker per pending sub-question.

    When nothing is pending the run goes straight to the critic via the
    ``to_critic`` passthrough so the pipeline still terminates instead of
    silently ending on an empty fan-out.
    """
    sends = [
        Send("search_worker", {"sub_question": sq})
        for sq in state.get("pending_questions", [])
    ]
    return sends or "to_critic"


def _route_to_scrapes(state: ResearchState) -> list[Send] | str:
    """Fan out one scrape_worker per unique search result URL."""
    seen: set[str] = set()
    sends: list[Send] = []
    for result in state.get("search_results", []):
        if result.url in seen:
            continue
        seen.add(result.url)
        sends.append(Send("scrape_worker", {"search_result": result}))
    return sends or "to_critic"


def _route_to_parsers(state: ResearchState) -> list[Send] | str:
    """Fan out one document_worker per fetched raw document."""
    sends = [
        Send("document_worker", {"raw_document": doc})
        for doc in state.get("raw_documents", [])
    ]
    return sends or "to_critic"


def _route_after_input_guardrail(state: ResearchState) -> str:
    """Blocked topics short-circuit to the refusal output; nothing else runs."""
    return "rejection_output" if state.get("blocked") else "planner"


def _route_after_critic(state: ResearchState) -> str:
    """Bounded re-plan loop: replan while the critic demands it and rounds remain."""
    verdict = state.get("critic_verdict")
    if (
        verdict is not None
        and verdict.needs_more_research
        and state.get("critic_rounds", 0) < state.get("max_critic_rounds", 1)
    ):
        return "replan"
    return "write"


CompiledResearchGraph = CompiledStateGraph[
    ResearchState, Any, ResearchState, ResearchState
]


def build_graph(
    deps: GraphDeps, *, checkpointer: Any | None = None
) -> CompiledResearchGraph:
    """Compile the research graph.

    ``checkpointer`` is optional; job runners pass a SQLite checkpointer so
    runs can later support HITL interrupts.
    """
    builder = StateGraph(ResearchState)

    async def _aggregate(state: ResearchState) -> dict[str, Any]:
        """Barrier node: runs once all upstream worker branches have merged."""
        return {}

    builder.add_node("input_guardrail", RunnableLambda(make_input_guardrail(deps)))
    builder.add_node("rejection_output", RunnableLambda(make_rejection_output(deps)))
    builder.add_node("planner", RunnableLambda(make_planner(deps)))
    builder.add_node("search_worker", RunnableLambda(make_search_worker(deps)))
    builder.add_node("aggregate_search", RunnableLambda(_aggregate))
    builder.add_node("scrape_worker", RunnableLambda(make_scrape_worker(deps)))
    builder.add_node("aggregate_documents", RunnableLambda(_aggregate))
    builder.add_node("document_worker", RunnableLambda(make_document_worker(deps)))
    builder.add_node("to_critic", RunnableLambda(_aggregate))
    builder.add_node("critic", RunnableLambda(make_critic(deps)))
    builder.add_node("replan", RunnableLambda(make_replan(deps)))
    builder.add_node("writer", RunnableLambda(make_writer(deps)))
    builder.add_node("output_guardrail", RunnableLambda(make_output_guardrail(deps)))
    builder.add_node("human_review", RunnableLambda(make_human_review(deps)))
    builder.add_node("report_assembler", RunnableLambda(make_report_assembler(deps)))

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        _route_after_input_guardrail,
        {"rejection_output": "rejection_output", "planner": "planner"},
    )
    builder.add_edge("rejection_output", END)
    builder.add_conditional_edges(
        "planner", _route_to_search, ["search_worker", "to_critic"]
    )
    builder.add_edge("search_worker", "aggregate_search")
    builder.add_conditional_edges(
        "aggregate_search", _route_to_scrapes, ["scrape_worker", "to_critic"]
    )
    builder.add_edge("scrape_worker", "aggregate_documents")
    builder.add_conditional_edges(
        "aggregate_documents", _route_to_parsers, ["document_worker", "to_critic"]
    )
    builder.add_edge("to_critic", "critic")
    builder.add_edge("document_worker", "critic")
    builder.add_conditional_edges(
        "critic", _route_after_critic, {"replan": "replan", "write": "writer"}
    )
    builder.add_conditional_edges(
        "replan", _route_to_search, ["search_worker", "to_critic"]
    )
    builder.add_edge("writer", "output_guardrail")
    builder.add_edge("output_guardrail", "human_review")
    builder.add_edge("human_review", "report_assembler")
    builder.add_edge("report_assembler", END)

    return builder.compile(checkpointer=checkpointer)
