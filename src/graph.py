"""
LangGraph orchestration — defines the multi-agent workflow as a directed graph.

Key concepts:
- StateGraph: the workflow definition (nodes + edges)
- Node: a function (AgentState -> dict) — your agents
- Edge: unconditional connection between nodes
- Conditional Edge: route to different nodes based on state
- MemorySaver: persists state between graph invocations (enables interrupts)
- interrupt_before: pauses the graph BEFORE a node, letting humans inspect/modify state

Think of it as a flowchart where:
- Boxes = agents (nodes)
- Arrows = routing logic (edges)
- Pause points = human checkpoints
"""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agents.researcher import researcher_agent
from .agents.retriever import retriever_agent
from .agents.writer import writer_agent
from .state import AgentState

logger = logging.getLogger(__name__)


def route_after_researcher(state: AgentState) -> str:
    """
    Conditional routing function: decides what runs after the Researcher.

    Returns the NAME of the next node. LangGraph uses the returned string
    to look up which node to execute next.

    Logic: always run Retriever after Researcher. The Writer runs last.
    In more complex graphs, this function could branch based on research quality,
    question type, or missing information signals.
    """
    completed = state.get("completed_agents", [])
    if "retriever" not in completed:
        return "retriever"
    return "writer"


def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-loop checkpoint node.

    When interrupt_before=["human_review"] is set, LangGraph PAUSES the graph
    before executing this node. The graph state is serialized to MemorySaver.
    External code can then:
      1. Read the current state (inspect the draft report)
      2. Modify state (inject human_feedback)
      3. Resume the graph with graph.invoke(None, config=config)

    In the Streamlit UI, this pause is surfaced as a "Review & Approve" step.
    In a production system, it would trigger a notification (email, Slack)
    and wait for an async webhook response.
    """
    feedback = state.get("human_feedback")
    if feedback:
        logger.info(f"Human review: feedback received — '{feedback[:100]}'")
    else:
        logger.info("Human review: approved with no changes")
    # LangGraph 0.2.52+ requires at least one state field in the returned dict
    return {"human_feedback": feedback or ""}


def build_research_graph():
    """
    Build and compile the multi-agent research graph.

    Graph topology:
        researcher → [conditional] → retriever → writer → human_review → END

    MemorySaver stores state between the interrupt and resume calls.
    For production, replace with SqliteSaver or RedisSaver for persistence
    across process restarts.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("human_review", human_review_node)

    # Entry point
    workflow.set_entry_point("researcher")

    # Conditional routing after researcher
    workflow.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "retriever": "retriever",
            "writer": "writer",
        },
    )

    # Fixed edges
    workflow.add_edge("retriever", "writer")
    workflow.add_edge("writer", "human_review")
    workflow.add_edge("human_review", END)

    # Compile with memory + human interrupt point
    memory = MemorySaver()
    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],  # Pause BEFORE human_review executes
    )

    logger.info("Multi-agent graph compiled successfully")
    return graph


def run_until_interrupt(question: str, thread_id: str = "default") -> tuple:
    """
    Run the graph until the human_review interrupt.

    Returns (graph, state, config) so the caller can:
    - Inspect state["final_report"]
    - Optionally update state["human_feedback"]
    - Resume with graph.invoke(None, config=config)
    """
    graph = build_research_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": question,
        "messages": [],
        "completed_agents": [],
        "web_search_results": None,
        "doc_search_results": None,
        "final_report": None,
        "human_feedback": None,
        # Token accumulators start at 0; operator.add in AgentState adds each agent's usage
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        # Researcher sets this to True on failure so Writer adjusts report
        "search_failed": False,
    }

    logger.info(f"Starting research graph for: '{question[:60]}'")
    state = graph.invoke(initial_state, config=config)
    logger.info("Graph paused at human_review checkpoint")
    return graph, state, config


def resume_after_review(graph, config: dict, feedback: str = "") -> dict:
    """
    Resume the graph after human review, optionally with feedback.

    LangGraph resumes from the persisted checkpoint (MemorySaver).
    Passing None as the first argument tells LangGraph to resume from
    the last interrupt point rather than starting fresh.
    """
    if feedback:
        # Update state before resuming
        graph.update_state(config, {"human_feedback": feedback})
    final_state = graph.invoke(None, config=config)
    return final_state
