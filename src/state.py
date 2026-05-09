"""
Shared agent state — the single source of truth passed through every node.

LangGraph merges state updates returned by each node. The Annotated[List, operator.add]
pattern is critical: it gives messages APPEND semantics rather than overwrite semantics,
preserving the full conversation history across agent steps.

TypedDict makes the schema explicit and type-safe. A poorly designed state is the
most common source of subtle bugs in LangGraph — fields overwritten when they should
append, or missing fields causing KeyError inside an agent.
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state threaded through every node in the research graph.

    Fields use Optional[str] (not str) so the initial state can omit them
    without TypedDict raising a validation error.

    Token fields use Annotated[int, operator.add] so each agent's usage
    accumulates into a running total across the full graph run — LangGraph
    calls operator.add(current, new) when merging state updates.
    """

    # The original user question — never mutated after initialization
    question: str

    # Full message history — append-only via operator.add
    messages: Annotated[List[BaseMessage], operator.add]

    # Results populated by each agent
    web_search_results: Optional[str]
    doc_search_results: Optional[str]

    # Final synthesized report produced by the writer
    final_report: Optional[str]

    # Tracks which agents have completed (used for conditional routing)
    completed_agents: Annotated[List[str], operator.add]

    # Human feedback injected at the review checkpoint
    human_feedback: Optional[str]

    # Per-run token usage — accumulated across all agents via operator.add
    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens: Annotated[int, operator.add]

    # Set to True by researcher when web search fails after all retries;
    # writer uses this to adjust report tone ("based on local docs only")
    search_failed: bool
