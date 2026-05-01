# Multi-Agent Research Assistant — LangGraph + Gemini + Tavily

> Three specialized agents — Researcher (web), Retriever (local docs), Writer (synthesis) — orchestrated by LangGraph with a human-in-the-loop review checkpoint. The most in-demand architecture pattern in production GenAI right now.

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-red)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange)
![Tavily](https://img.shields.io/badge/Tavily-web_search-green)

---

## Skills Demonstrated

| Category | Technologies / Concepts |
|----------|------------------------|
| **Agent Orchestration** | LangGraph StateGraph, conditional routing, directed graph topology |
| **State Management** | TypedDict, `Annotated[List, operator.add]` append semantics, state merging |
| **Human-in-the-Loop** | `interrupt_before`, MemorySaver checkpointing, `graph.update_state`, resume pattern |
| **Tool Calling** | `llm.bind_tools()`, Tavily API, structured LLM function calls |
| **Multi-Agent Design** | Single-responsibility agents, separation of research vs synthesis |
| **LLM Framework** | LangChain 0.3, `ChatGoogleGenerativeAI`, `ChatPromptTemplate`, LCEL in agents |
| **UI** | Streamlit with stage-based state machine (`input → researching → review → complete`) |
| **Testing** | pytest, isolated agent node tests, routing logic tests |

---

## What This Builds

**The Problem:** Single-agent systems struggle with research tasks that require breadth (public web) and depth (private docs) combined. A single LLM call can't effectively search the web, search local documents, and write a synthesis report — the context gets too long and the LLM loses focus.

**The Solution:** Three specialized agents, each doing one thing well:
- **Researcher**: queries Tavily for current, public web knowledge
- **Retriever**: queries ChromaDB for private, domain-specific document knowledge
- **Writer**: synthesizes both into a structured, cited report

A LangGraph state machine orchestrates the order, persists state between steps, and pauses for human review before finalizing.

**The Outcome:** A research assistant that combines internet knowledge with your private documents, with a human checkpoint before the final report is delivered.

```
User Question
      │
      ▼
[Researcher] — Tavily web search (5 sources)
      │
      ▼
[Retriever] — ChromaDB local doc search (top-4 chunks)
      │
      ▼
[Writer] — Synthesizes web + local results into report
      │
      ▼
[Human Review] ← PAUSES HERE — you approve or add feedback
      │
      ▼
Final Report (Markdown, downloadable)
```

---

## Architecture

```mermaid
graph TD
    START([User Question]) --> R[🔍 Researcher\nTavily web search]
    R --> COND{route_after_researcher}
    COND -->|retriever not done| RET[📚 Retriever\nChromaDB local search]
    COND -->|retriever done| W
    RET --> W[✍️ Writer\nGemini synthesis]
    W --> HR[👤 Human Review\ninterrupt_before checkpoint]
    HR --> END([Final Report])

    style HR fill:#fff3cd,stroke:#ffc107
```

See [docs/architecture.md](docs/architecture.md) for the full interrupt-and-resume flow diagram and state schema breakdown.

---

## Key Engineering Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **State accumulation** | `Annotated[List, operator.add]` | Messages and completed_agents must append, not overwrite. Without this, each agent's history is lost when the next agent runs. |
| **Routing via state** | Check `completed_agents` in routing fn | Decouples the topology from hard-coded order. Adding a new agent only requires updating the routing function, not rewiring edges. |
| **MemorySaver** | In-memory checkpointer | Simplest option for a portfolio project. Production: SqliteSaver (single node) or RedisSaver (distributed). |
| **interrupt_before** | `["human_review"]` | Pauses before the node executes — the node's output is the approved state. If you interrupt_after, you'd need the node to produce an "I'm waiting" result. |
| **temperature=0 for Researcher/Retriever** | Deterministic queries | Search queries benefit from determinism. Writer uses 0.3 for natural-sounding synthesis. |
| **Agents as pure functions** | `(state) -> dict` | Testable without the graph. Each agent's logic can be unit tested in isolation with a fake state. |
| **Gemini for LLM** | Gemini 2.0 Flash | Free, fast, handles tool calling well. Tavily integration via LangChain community tools. |

---

## Tech Stack

| Component | Technology | Version | Why |
|-----------|-----------|---------|-----|
| Agent Orchestration | LangGraph | 0.2.52 | Industry standard for stateful multi-agent workflows |
| LLM | Gemini 2.0 Flash | `langchain-google-genai 2.0.4` | Free API, handles tool calling correctly |
| Web Search | Tavily | tavily-python 0.5.0 | Purpose-built for AI agents: clean results, no HTML noise |
| Local Search | ChromaDB | 0.5.18 | Reuses Project 1's vector store for local document Q&A |
| State Persistence | MemorySaver | bundled with LangGraph | In-memory checkpoint for interrupt/resume within a session |
| UI | Streamlit | 1.40.1 | 4-stage state machine (input→researching→review→complete) |
| Tracing | LangSmith | auto via env var | Per-node trace shows exactly what each agent sent/received |

---

## Quick Start

```bash
# API Keys needed:
# GOOGLE_API_KEY: free at https://aistudio.google.com/apikey
# TAVILY_API_KEY: free (1000 searches/month) at https://tavily.com
# LANGCHAIN_API_KEY: optional, free at https://smith.langchain.com

git clone https://github.com/themoizqureshi/multi-agent-langgraph
cd multi-agent-langgraph

cp .env.example .env
# Fill in GOOGLE_API_KEY and TAVILY_API_KEY

uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

streamlit run app.py
# Opens at http://localhost:8501
```

**Optional:** Point to a local ChromaDB from Project 1:
```bash
# In .env or export:
CHROMA_DB_PATH=../rag-chatbot-langchain/chroma_db
```

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
multi-agent-langgraph/
├── src/
│   ├── state.py             # AgentState TypedDict with Annotated fields
│   ├── graph.py             # build_research_graph(), routing fn, run_until_interrupt(), resume_after_review()
│   ├── agents/
│   │   ├── researcher.py    # Tavily web search via tool calling
│   │   ├── retriever.py     # ChromaDB local document search
│   │   └── writer.py        # Gemini synthesis from both results
│   └── tools/
│       ├── web_search.py    # TavilySearchResults tool factory
│       └── doc_search.py    # ChromaDB retriever factory + search_local_docs()
├── app.py                   # Streamlit UI with 4-stage workflow
├── tests/
│   └── test_graph.py        # Routing logic tests, agent isolation tests
└── docs/
    └── architecture.md      # Mermaid graph + interrupt-and-resume flow
```

---

## Production Considerations

| Concern | Current State | Production Approach |
|---------|--------------|---------------------|
| **State persistence** | MemorySaver (in-memory, lost on restart) | SqliteSaver or RedisSaver; state survives server restarts |
| **Async human review** | Blocking Streamlit session | Pause → send email/Slack notification → resume via webhook; requires async job queue |
| **Parallel research** | Researcher → Retriever (sequential) | Fan-out: both run in parallel; merge before Writer — halves research time |
| **Agent quality evaluation** | No per-agent scoring | LangSmith custom evaluators; RAGAS context_recall on Retriever output |
| **Loop protection** | None (graph structure prevents loops) | Add `iteration_count` to state; routing fn returns `"error_handler"` if count exceeds max |
| **Tool errors** | Unhandled | Wrap tool calls in try/except; return error state; route to a fallback node |

---

## Lessons Learned

- First run: the Writer fired before the Retriever completed because `route_after_researcher` was checking `"retriever"` but `completed_agents` was populated with `"retriever_node"`. LangGraph doesn't validate routing key names against node names — the graph compiled and failed silently at runtime. Defensive logging in the routing function caught it.
- `Annotated[List, operator.add]` on state fields is easy to forget and hard to notice: without it, each agent's output silently overwrites the previous one. The symptom is the Writer only sees the most recent agent's results, not the accumulated history.
- LangSmith per-node traces showed the Researcher was returning 5 Tavily results but 2 were from tangentially related articles. The Writer included them in the synthesis anyway. A relevance filter inside the Researcher node before writing to state would improve output quality significantly.
- `MemorySaver` stores checkpoint state in-process — restarting the Streamlit server loses the interrupt. `SqliteSaver` is the minimum viable persistence for anything that needs to survive a restart or be handed off between sessions.

---

*Part of the [AI Engineer Portfolio](https://github.com/themoizqureshi) — Project 4 of 5.*  
*Previous: [Project 3 — Local LLM + Pinecone + FastAPI](https://github.com/themoizqureshi/local-llm-rag-pinecone)*  
*Next: [Project 5 — LLMOps Pipeline](https://github.com/themoizqureshi/llmops-pipeline)*
