"""
Researcher Agent — searches the web for information using Tavily.

Reliability features added in Stage 3:
  - Retry with exponential backoff (tenacity): up to 3 attempts on transient errors.
  - Circuit breaker: after _CIRCUIT_THRESHOLD consecutive run-level failures, the
    circuit opens and web search is skipped entirely until a success resets it.
    Module-level state so it persists across graph runs in the same process.
  - Graceful fallback: on failure (or open circuit), sets search_failed=True so
    the Writer can adjust the report and not fabricate web sources.
  - Cost tracking: LLM call tokens captured and returned as state fields so
    LangGraph accumulates totals across agents via operator.add.
"""

import logging
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from langchain_core.messages import AIMessage, HumanMessage

from ..state import AgentState
from ..tools.web_search import get_web_search_tool
from ..cost_tracker import CostTracker

logger = logging.getLogger(__name__)

# ── Circuit breaker ───────────────────────────────────────────────────────────
# Module-level — persists across graph runs in the same process.
_CIRCUIT_THRESHOLD = 3
_circuit: dict = {"failures": 0, "open": False}


def _circuit_is_open() -> bool:
    return _circuit["open"]


def _record_success() -> None:
    if _circuit["failures"] > 0 or _circuit["open"]:
        logger.info("Circuit breaker RESET: web search re-enabled after success")
    _circuit["failures"] = 0
    _circuit["open"] = False


def _record_failure() -> None:
    _circuit["failures"] += 1
    if _circuit["failures"] >= _CIRCUIT_THRESHOLD:
        _circuit["open"] = True
        logger.warning(
            "Circuit breaker OPEN: web search disabled after %d consecutive failures",
            _CIRCUIT_THRESHOLD,
        )


# ── LLM factory ───────────────────────────────────────────────────────────────

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


# ── Retried search function ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _run_search_with_retry(question: str, tracker: CostTracker) -> str:
    """
    Call Tavily + LLM with tenacity retry (up to 3× with 1s/2s/4s backoff).

    Decorated separately from researcher_agent so the circuit breaker wraps
    the entire retry block — one circuit-failure per agent call, not per attempt.
    """
    llm = _get_llm(temperature=0)
    search_tool = get_web_search_tool(max_results=5)
    llm_with_tools = llm.bind_tools([search_tool])

    prompt = RESEARCHER_SYSTEM_PROMPT.format(question=question)
    response = llm_with_tools.invoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [tracker]},
    )

    search_results = ""
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            results = search_tool.invoke(tool_call["args"])
            search_results += f"\n\nSearch Results:\n{results}"
    else:
        search_results = response.content

    return search_results


# ── Agent node ────────────────────────────────────────────────────────────────

def researcher_agent(state: AgentState) -> dict:
    """
    Research agent node: calls Tavily and summarises findings.

    Returns a state update dict — only fields this node changes.
    LangGraph merges this with the existing state (does not replace it).

    On circuit-open or exhausted retries: sets search_failed=True and returns
    a placeholder so the Writer can fall back to local documents gracefully.
    """
    # ── Circuit breaker check ────────────────────────────────────────────────
    if _circuit_is_open():
        logger.warning("Researcher: circuit breaker OPEN — skipping web search")
        return {
            "web_search_results": (
                "[WEB_SEARCH_UNAVAILABLE] Circuit breaker open after repeated failures. "
                "Report will be based on local documents only."
            ),
            "search_failed": True,
            "completed_agents": ["researcher"],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "messages": [AIMessage(content="[Researcher] Circuit breaker open — web search skipped.")],
        }

    logger.info("Researcher agent: starting web search")
    tracker = CostTracker()

    try:
        search_results = _run_search_with_retry(state["question"], tracker)
        _record_success()
        logger.info("Researcher agent: collected %d chars of web results", len(search_results))

        return {
            "web_search_results": search_results,
            "search_failed": False,
            "completed_agents": ["researcher"],
            "total_input_tokens": tracker.summary.input_tokens,
            "total_output_tokens": tracker.summary.output_tokens,
            "messages": [AIMessage(content=f"[Researcher] Web search complete ({len(search_results)} chars).")],
        }

    except Exception as exc:
        # All retries exhausted — record failure in circuit breaker and fall back.
        _record_failure()
        logger.error("Researcher agent: web search failed after retries — %s", exc)

        return {
            "web_search_results": (
                f"[WEB_SEARCH_FAILED] Tavily unavailable after 3 attempts ({type(exc).__name__}). "
                "Report will be based on local documents only."
            ),
            "search_failed": True,
            "completed_agents": ["researcher"],
            "total_input_tokens": tracker.summary.input_tokens,
            "total_output_tokens": tracker.summary.output_tokens,
            "messages": [AIMessage(content=f"[Researcher] Web search failed: {type(exc).__name__}. Falling back to local docs.")],
        }
