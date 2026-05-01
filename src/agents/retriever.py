"""
Retriever Agent — searches local documents in ChromaDB.

Complements the Researcher agent: while Researcher covers the public web,
Retriever covers the user's private document store (uploaded PDFs from Project 1).

If no ChromaDB exists, this agent returns a graceful "no local docs" message
instead of crashing — the Writer can still produce a report from web results alone.
"""

import logging

from langchain_core.messages import AIMessage

from ..state import AgentState
from ..tools.doc_search import search_local_docs

logger = logging.getLogger(__name__)


def retriever_agent(state: AgentState) -> dict:
    """
    Retriever agent node: searches local ChromaDB for relevant document chunks.

    Does not call an LLM — just retrieves and formats the top-k chunks.
    The synthesis step is left to the Writer agent.
    """
    logger.info("Retriever agent: searching local documents")

    question = state["question"]
    doc_results = search_local_docs(question)

    logger.info(f"Retriever agent: found {len(doc_results)} chars of local doc content")

    return {
        "doc_search_results": doc_results,
        "completed_agents": ["retriever"],
        "messages": [AIMessage(content=f"[Retriever] Local document search complete.")],
    }
