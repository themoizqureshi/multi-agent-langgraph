"""
Tests for LangGraph agent nodes and routing logic.

Strategy: test each agent node in isolation (pure functions) and test
the routing logic separately. Avoid running the full graph in unit tests
— that would require live API calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.state import AgentState
from src.graph import route_after_researcher, human_review_node


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(**overrides) -> AgentState:
    """Build a minimal valid AgentState for testing."""
    base: AgentState = {
        "question": "What is RAG?",
        "messages": [],
        "web_search_results": None,
        "doc_search_results": None,
        "final_report": None,
        "completed_agents": [],
        "human_feedback": None,
    }
    base.update(overrides)
    return base


# ── Routing logic tests ───────────────────────────────────────────────────────

def test_route_after_researcher_goes_to_retriever_first():
    state = make_state(completed_agents=["researcher"])
    assert route_after_researcher(state) == "retriever"


def test_route_after_researcher_goes_to_writer_when_retriever_done():
    state = make_state(completed_agents=["researcher", "retriever"])
    assert route_after_researcher(state) == "writer"


def test_route_handles_empty_completed_agents():
    state = make_state(completed_agents=[])
    assert route_after_researcher(state) == "retriever"


# ── Human review node tests ───────────────────────────────────────────────────

def test_human_review_returns_human_feedback():
    state = make_state()
    result = human_review_node(state)
    assert "human_feedback" in result
    assert result["human_feedback"] == ""


def test_human_review_passes_feedback_through():
    state = make_state(human_feedback="Please add more detail.")
    result = human_review_node(state)
    assert result["human_feedback"] == "Please add more detail."


# ── Researcher agent tests ────────────────────────────────────────────────────

def test_researcher_agent_returns_correct_state_keys():
    mock_llm_response = MagicMock()
    mock_llm_response.tool_calls = []
    mock_llm_response.content = "RAG is a technique that combines retrieval with generation."

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value.invoke.return_value = mock_llm_response

    with patch("src.agents.researcher._get_llm", return_value=mock_llm), \
         patch("src.agents.researcher.get_web_search_tool") as mock_tool_fn:
        mock_tool_fn.return_value = MagicMock()

        from src.agents.researcher import researcher_agent
        state = make_state(question="What is RAG?")
        result = researcher_agent(state)

    assert "web_search_results" in result
    assert "completed_agents" in result
    assert "messages" in result
    assert "researcher" in result["completed_agents"]


# ── Retriever agent tests ─────────────────────────────────────────────────────

def test_retriever_agent_handles_no_local_docs():
    with patch("src.agents.retriever.search_local_docs") as mock_search:
        mock_search.return_value = "No local documents available. Local document search skipped."

        from src.agents.retriever import retriever_agent
        state = make_state(question="What is RAG?")
        result = retriever_agent(state)

    assert "doc_search_results" in result
    assert "retriever" in result["completed_agents"]
    assert "No local documents" in result["doc_search_results"]


def test_retriever_agent_returns_local_results():
    with patch("src.agents.retriever.search_local_docs") as mock_search:
        mock_search.return_value = "[Local Doc 1 — doc.pdf, p.3]\nRAG combines retrieval with generation."

        from src.agents.retriever import retriever_agent
        state = make_state(question="What is RAG?")
        result = retriever_agent(state)

    assert "Local Doc 1" in result["doc_search_results"]


# ── Writer agent tests ────────────────────────────────────────────────────────

def test_writer_agent_produces_final_report():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "## Research Report\n\nRAG stands for Retrieval-Augmented Generation..."

    with patch("src.agents.writer._get_llm", return_value=MagicMock()), \
         patch("src.agents.writer.WRITER_PROMPT") as mock_prompt:
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)
        mock_chain.__or__ = MagicMock(return_value=mock_chain)

        from src.agents.writer import writer_agent
        state = make_state(
            question="What is RAG?",
            web_search_results="Web: RAG is a technique...",
            doc_search_results="Local: RAG combines retrieval...",
        )
        result = writer_agent(state)

    assert "final_report" in result
    assert "completed_agents" in result
    assert "writer" in result["completed_agents"]
