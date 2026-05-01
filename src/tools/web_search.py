"""
Tavily web search tool.

Tavily is purpose-built for AI agents — it returns clean, structured results
optimised for LLM consumption (no raw HTML, no ads, no navigation boilerplate).
Free tier: 1000 searches/month.

Comparison to other search tools:
- SerpAPI: returns raw Google results, noisy for LLMs
- DuckDuckGo (free): less accurate, no answer extraction
- Tavily: designed specifically for RAG/agent use, includes extracted answers
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults


def get_web_search_tool(max_results: int = 5) -> TavilySearchResults:
    """
    Return a LangChain-compatible Tavily search tool.

    max_results=5: enough sources for a solid research summary without
    overwhelming the LLM context window with redundant information.
    search_depth="advanced" does more crawling than "basic" — worth the
    extra latency for a research agent that runs once per report.
    """
    return TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_answer=True,       # Tavily's own extracted answer — useful as a summary
        include_raw_content=False, # Full HTML is too noisy; stick to cleaned snippets
    )
