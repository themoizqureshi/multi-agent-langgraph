"""
Writer Agent — synthesizes web + local doc results into a final report.

The writer is the only agent that sees the full picture. It receives
web_search_results from the Researcher, doc_search_results from the Retriever,
and synthesizes them into a coherent, cited report.

This separation of concerns mirrors real research workflows:
- Researcher: broad web coverage (the generalist)
- Retriever: deep domain-specific knowledge (the specialist)
- Writer: synthesis and clear communication (the communicator)
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from ..state import AgentState

logger = logging.getLogger(__name__)

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional research writer. Synthesize the provided research into a clear, well-structured report.

Guidelines:
- Lead with a direct answer to the question
- Organize findings into logical sections with headers
- Cite sources by referencing [Web] or [Local Doc] for each claim
- Note any contradictions between web and local sources
- End with a brief summary and confidence level
- Use markdown formatting for readability"""),
    ("human", """Question: {question}

Web Research Results:
{web_results}

Local Document Results:
{doc_results}

Write a comprehensive research report answering the question."""),
])


def writer_agent(state: AgentState) -> dict:
    """
    Writer agent node: synthesizes all research into a final report.

    Combines web_search_results and doc_search_results from state.
    Falls back gracefully if either source is missing.
    """
    logger.info("Writer agent: synthesizing report")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    chain = WRITER_PROMPT | llm | StrOutputParser()

    report = chain.invoke({
        "question": state["question"],
        "web_results": state.get("web_search_results") or "No web results available.",
        "doc_results": state.get("doc_search_results") or "No local documents available.",
    })

    logger.info(f"Writer agent: generated {len(report)}-char report")

    return {
        "final_report": report,
        "completed_agents": ["writer"],
        "messages": [AIMessage(content=f"[Writer] Report complete ({len(report)} chars).")],
    }
