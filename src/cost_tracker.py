"""
Token usage and cost tracking via LangChain callback.

Same pattern as rag-chatbot-langchain/src/cost_tracker.py — each project owns
its own copy so repos stay independent. Wire in at invoke time via
config={"callbacks": [tracker]}; return tracker.summary as state fields so
LangGraph accumulates totals across agents via operator.add.
"""

import logging
from dataclasses import dataclass
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-2.0-flash-001": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "google/gemini-2.0-flash-001": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "default": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
}


@dataclass
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "unknown"

    def display(self) -> str:
        return (
            f"Tokens: {self.input_tokens:,} in / {self.output_tokens:,} out"
            f" | Est. cost: ${self.cost_usd:.6f} | Model: {self.model}"
        )


class CostTracker(BaseCallbackHandler):
    """Captures token usage and estimated USD cost for one LLM call."""

    def __init__(self) -> None:
        self._summary = UsageSummary()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:  # noqa: ANN003
        model = "default"

        # Gemini path: usage_metadata on AIMessage
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None) or {}
                self._summary.input_tokens += usage.get("input_tokens", 0)
                self._summary.output_tokens += usage.get("output_tokens", 0)
                self._summary.total_tokens += usage.get("total_tokens", 0)
                meta = getattr(msg, "response_metadata", {}) or {}
                model = meta.get("model_name", model)

        # OpenRouter / OpenAI path
        llm_out = response.llm_output or {}
        oai_usage = llm_out.get("token_usage", {})
        if oai_usage:
            self._summary.input_tokens += oai_usage.get("prompt_tokens", 0)
            self._summary.output_tokens += oai_usage.get("completion_tokens", 0)
            self._summary.total_tokens += oai_usage.get("total_tokens", 0)
            model = llm_out.get("model_name", model)

        self._summary.model = model
        pricing = _PRICING.get(model, _PRICING["default"])
        self._summary.cost_usd = (
            self._summary.input_tokens * pricing["input"]
            + self._summary.output_tokens * pricing["output"]
        )
        logger.debug("Cost tracker [%s]: %s", model, self._summary.display())

    @property
    def summary(self) -> UsageSummary:
        return self._summary
