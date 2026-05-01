# Interview Prep — Multi-Agent LangGraph

> This is the most-discussed project in interviews for senior AI roles in 2024-25. Multi-agent orchestration is the frontier — most teams are actively building it, and candidates who can explain the state design and interrupt pattern in detail stand out.

---

## Core Concept Questions

### Q: What is LangGraph and when would you use it over a simple LangChain chain?

> "LangChain chains are linear pipelines — input flows through steps A→B→C and exits. They're perfect for fixed, predictable workflows like RAG Q&A.
>
> LangGraph models workflows as directed graphs with cycles and conditional branches. I use LangGraph when I need: (1) conditional routing — go to different agents depending on what the previous agent found; (2) human-in-the-loop — pause the workflow for human review and resume with feedback; (3) stateful multi-turn conversations — the graph remembers everything between turns via the checkpointer; or (4) agent retry loops — if an agent fails or produces low-quality output, route back and retry.
>
> For a simple RAG Q&A system, a chain is simpler and sufficient. For a research assistant that needs web search, local document search, synthesis, and human approval — a graph is the right abstraction."

---

### Q: Explain the AgentState design. Why use TypedDict and Annotated?

> "The state is a TypedDict — a typed dictionary where every field has an explicit type. TypedDict makes the state schema readable and IDE-completeable. Without it, state is just an opaque dict and you get KeyErrors at runtime instead of type errors at write time.
>
> `Annotated[List[BaseMessage], operator.add]` is the most important pattern in the codebase. When LangGraph merges a node's returned dict back into the state, it needs to know HOW to merge list fields — should it replace the list or append to it? `operator.add` tells it to append.
>
> Without it: if Researcher returns `{'messages': [msg_a]}` and then Writer returns `{'messages': [msg_b]}`, the state would end up with `messages = [msg_b]` — Researcher's message is gone. With `operator.add`, they're appended: `messages = [msg_a, msg_b]`. I use the same pattern for `completed_agents` so each agent's name accumulates in the list, which is what the routing function reads to decide what runs next."

---

### Q: How does the human-in-the-loop interrupt work?

> "When you compile the graph with `interrupt_before=['human_review']`, LangGraph will run every node in the graph EXCEPT `human_review`, then pause and serialize the full state to the checkpointer (MemorySaver in this project, which is an in-memory dict keyed by thread_id).
>
> At this point, `graph.invoke()` returns the current state — which includes the `final_report` the Writer produced. The Streamlit UI reads this, displays it to the user, and shows a feedback text box.
>
> When the user clicks Approve, the UI calls `graph.invoke(None, config)`. The `None` first argument means 'continue from the last checkpoint, don't reinitialize'. LangGraph deserializes the saved state, executes `human_review_node()` (which just reads any feedback), and the graph ends.
>
> The `thread_id` in config is what links the interrupted invocation to the resumed one. Different thread_ids = independent conversations with independent state."

---

### Q: What is tool calling and how does the Researcher agent use it?

> "Tool calling (also called function calling) is when you give an LLM a description of available tools and it decides when to call them and what arguments to pass. LangChain's `.bind_tools([tool])` injects tool schemas into the system prompt in the format the LLM expects.
>
> In the Researcher agent, I call `llm.bind_tools([tavily_search_tool])`. When the LLM processes the research prompt, it generates a structured JSON response like `{'tool_calls': [{'name': 'tavily_search', 'args': {'query': 'RAG systems enterprise 2024'}}]}` instead of plain text. My code then executes that call and feeds the results back.
>
> Why use tool calling instead of just hardcoding the Tavily call? The LLM can reformulate the user's question into better search queries. It might split one question into three targeted queries for better coverage. It also decides whether to call the tool at all — if the user's question can be answered from its training knowledge, it might skip the search. This is more intelligent than always running the same fixed query."

---

### Q: How do you prevent agents from getting stuck in an infinite loop?

> "Three layers of protection:
>
> **1. State-based termination:** The `completed_agents` list in the state tracks which agents have run. The routing function checks this list to decide the next node. Once `['researcher', 'retriever']` are both complete, it routes to writer and then to END — there's no way to loop back unless you explicitly write that edge.
>
> **2. Max iteration guard:** In production, I'd add an `iteration_count: int` field to AgentState and increment it in each node. The routing function would check `if state['iteration_count'] > MAX_ITERATIONS: return 'error_handler'`. The error handler returns a graceful failure message.
>
> **3. LangSmith tracing:** When tracing is enabled, LangSmith records every node execution with its inputs and outputs. An infinite loop shows up immediately as a repeated sequence in the trace tree — you can spot it and debug the routing condition within seconds."

---

### Q: How would you scale this multi-agent system for production?

> "Four main changes:
>
> **Checkpointer**: Replace MemorySaver (in-memory, dies on restart) with SqliteSaver for single-server persistence, or RedisSaver for distributed, multi-instance deployment.
>
> **Parallel research**: Run the Researcher and Retriever in parallel instead of sequentially. LangGraph supports this with `workflow.add_edge(START, 'researcher')` and `workflow.add_edge(START, 'retriever')` as concurrent branches that merge before the Writer.
>
> **Async human review**: Instead of blocking a Streamlit session, the graph pauses and sends a notification (email/Slack) with a link. The human reviews asynchronously and the graph resumes via a webhook. This requires a persistent checkpointer (Redis/Postgres) and an async job queue.
>
> **Evaluation**: Use LangSmith to trace every agent run. Track per-agent quality scores (did the researcher find relevant results? did the writer cite them correctly?) and alert when scores degrade. Connect to Project 2's RAGAS evaluation framework for automated quality gates."

---

### Q: What is the difference between the Researcher and Retriever agents and why separate them?

> "The Researcher and Retriever have fundamentally different information sources and latency profiles.
>
> The Researcher calls the Tavily API — it gets current, broad, public knowledge. It's slower (network call), less predictable (web changes), and covers any topic.
>
> The Retriever queries local ChromaDB — it gets deep, private, specific knowledge from documents you've uploaded. It's faster (local), deterministic (the database doesn't change during a conversation), and only covers what you've ingested.
>
> Separating them means each can be independently optimized, tested, and swapped. If I want to replace Tavily with a different search API, I change one file. If I want to add another local document source (Confluence, Notion), I add it to the Retriever without touching the Researcher.
>
> The Writer then synthesizes both perspectives — public web knowledge vs. private document knowledge — which produces richer reports than either source alone."

---

## Connecting to Your Production Experience

> "At Speridian, I used Vertex AI Agent Builder — Google's managed multi-agent service. It abstracts away what LangGraph makes explicit: the state schema, the routing logic, the checkpoint persistence. Understanding the LangGraph implementation helps me work more effectively with managed services because I know what's happening under the hood.
>
> For example, when Vertex AI Agent Builder has unexpected routing behavior, I can reason about it in terms of 'which conditional edge function might be returning the wrong value' — even though I'm not writing that code directly."

> "The human-in-the-loop pattern here mirrors a real production workflow I've seen at Speridian: a document extraction pipeline where an LLM extracts fields from mortgage documents, and then a human reviewer approves borderline cases before they're committed to the database. The interrupt-and-resume pattern is architecturally identical — the main difference is persistence mechanism (database vs MemorySaver) and notification system (email vs Streamlit UI)."
