"""
Researcher Agent — searches the web for information using Tavily.

Design principle: each agent is a pure function (AgentState) -> dict.
The dict contains only the state fields this agent updates.
LangGraph merges the returned dict back into the full state.

Single responsibility: the researcher ONLY searches and summarizes findings.
It does not write the final report — that's the writer's job.
This separation makes each agent independently testable and replaceable.
"""

import logging
import os

from langchain_core.messages import AIMessage, HumanMessage

from ..state import AgentState
from ..tools.web_search import get_web_search_tool

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0):
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="google/gemini-2.0-flash-001",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)


RESEARCHER_SYSTEM_PROMPT = """You are a research specialist. Your job is to search the web for accurate, relevant information to answer the user's question.

Guidelines:
- Use the search tool to find recent, authoritative sources
- Summarise the key findings concisely
- Note any contradictions or uncertainty across sources
- Focus on facts, not opinions

Question to research: {question}"""


def researcher_agent(state: AgentState) -> dict:
    """
    Research agent node: calls Tavily and summarises findings.

    Returns a state update dict — only includes fields this node changes.
    LangGraph merges this with the existing state (it does not replace it).
    """
    logger.info("Researcher agent: starting web search")

    llm = _get_llm(temperature=0)
    search_tool = get_web_search_tool(max_results=5)

    # Give the LLM access to the tool
    llm_with_tools = llm.bind_tools([search_tool])

    prompt = RESEARCHER_SYSTEM_PROMPT.format(question=state["question"])
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])

    # Execute any tool calls the LLM made
    search_results = ""
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            results = search_tool.invoke(tool_call["args"])
            search_results += f"\n\nSearch Results:\n{results}"
    else:
        # LLM answered directly without calling the tool (shouldn't happen often)
        search_results = response.content

    logger.info(f"Researcher agent: collected {len(search_results)} chars of web results")

    return {
        "web_search_results": search_results,
        "completed_agents": ["researcher"],
        "messages": [AIMessage(content=f"[Researcher] Web search complete. Found {len(search_results)} chars of results.")],
    }
