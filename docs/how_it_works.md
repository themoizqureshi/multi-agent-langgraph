# How It Works — Multi-Agent Research Assistant

## Why Multi-Agent?

A single-agent system with a single LLM call works fine for simple Q&A. But research tasks have a different structure: they require *different kinds of expertise* applied in *sequence*. 

A single prompt trying to do everything ("search the web, search local docs, synthesize into a report") will either be too long to handle well, or will produce mediocre results at each step because the LLM is context-switching too much.

Multi-agent systems solve this by separating concerns:
- **Researcher**: trained on prompting for web search, focused on breadth
- **Retriever**: specialized in local document similarity search, focused on domain depth
- **Writer**: focused entirely on synthesis and clear communication

Each agent does one thing well. The Writer receives *pre-processed, pre-filtered* results from both agents and can focus entirely on synthesis.

---

## Why LangGraph Instead of a Simple Chain?

LangChain LCEL chains are linear: `A | B | C | D`. They work well for fixed pipelines.

LangGraph models the workflow as a **directed graph** — which adds:

1. **Conditional routing**: the `route_after_researcher` function decides which node runs next based on the current state. This enables branching logic.

2. **State persistence across interrupts**: when the graph pauses at `human_review`, all state is serialized to `MemorySaver`. The human can review the output, close the browser, reopen it, and resume — the graph remembers exactly where it was.

3. **Loops and retries** (not used here but possible): a node can route back to an earlier node (e.g., "if writer quality score < 0.7, route back to researcher for more data").

4. **Parallel execution** (not used here but possible): two nodes can run concurrently if they don't depend on each other's output.

---

## The State Design Decision

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    completed_agents: Annotated[List[str], operator.add]
    # ... other fields as Optional[str]
```

The most important design choice is `Annotated[List, operator.add]`. Without it:

```python
# Without operator.add — WRONG for accumulation
messages: List[BaseMessage]
# Node A returns {"messages": [msg_a]}
# Node B returns {"messages": [msg_b]}
# State after B: messages = [msg_b]  ← msg_a is GONE
```

With `operator.add`, LangGraph uses the `+` operator to merge lists:
```python
# With operator.add — CORRECT
# State after A: messages = [msg_a]
# Node B returns {"messages": [msg_b]}
# LangGraph does: [msg_a] + [msg_b] = [msg_a, msg_b]
# State after B: messages = [msg_a, msg_b]  ← preserved
```

This pattern applies to any state field where you want accumulation instead of replacement.

---

## The Human-in-the-Loop Pattern

The `interrupt_before=["human_review"]` compile option is LangGraph's built-in human-in-the-loop mechanism. Here's what happens step by step:

```python
# 1. Compile with interrupt
graph = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_review"],
)

# 2. Run until interrupt
state = graph.invoke(initial_state, config={"configurable": {"thread_id": "abc"}})
# → Graph runs: researcher → retriever → writer
# → Graph PAUSES before human_review
# → state["final_report"] contains the draft

# 3. Human inspects state["final_report"] in the Streamlit UI

# 4. Optionally inject feedback
graph.update_state(config, {"human_feedback": "Add cost comparison section"})

# 5. Resume
final_state = graph.invoke(None, config=config)
# → human_review_node() executes (reads the feedback)
# → Graph ends → returns final_state
```

The `thread_id` in config is what ties the interrupted graph to its resumed continuation. Different `thread_id` values = different simultaneous conversations (each with their own state).

---

## The Researcher Agent

```python
def researcher_agent(state: AgentState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    search_tool = get_web_search_tool(max_results=5)
    llm_with_tools = llm.bind_tools([search_tool])
    
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            results = search_tool.invoke(tool_call["args"])
            search_results += results
    
    return {"web_search_results": search_results, "completed_agents": ["researcher"], ...}
```

`llm.bind_tools([search_tool])` tells Gemini that it has access to Tavily search. The LLM decides when to call it and what query to pass. This is **tool calling** (also called function calling) — the LLM doesn't just generate text, it generates a structured call to an external function.

**Why not just call Tavily directly without the LLM?** The LLM reformulates the user's question into better search queries. "What are the latest RAG developments?" might become three separate searches: "RAG systems 2024", "retrieval augmented generation enterprise", "RAG benchmarks recent". This multi-query strategy improves recall.

---

## The Routing Function

```python
def route_after_researcher(state: AgentState) -> str:
    completed = state.get("completed_agents", [])
    if "retriever" not in completed:
        return "retriever"
    return "writer"
```

This function returns a string — the *name* of the next node. LangGraph uses `add_conditional_edges` with a dict mapping return values to node names:

```python
workflow.add_conditional_edges(
    "researcher",           # From this node
    route_after_researcher, # Call this function to decide
    {
        "retriever": "retriever",  # If function returns "retriever" → go to retriever node
        "writer": "writer",        # If function returns "writer" → go to writer node
    },
)
```

This is where you'd add more complex routing logic in production — check research quality score, detect if the question is time-sensitive (skip local search), or route to a different agent based on topic classification.

---

## The Streamlit UI Stages

The UI tracks the workflow through 4 stages stored in `st.session_state.stage`:

```
"input"       → User types a question, clicks Start Research
"researching" → Graph runs (agents execute), Streamlit shows spinner
"review"      → Graph paused at interrupt, UI shows draft report + feedback box
"complete"    → Graph resumed and finished, final report displayed
```

`st.session_state` persists the `graph` object and `config` dict between Streamlit reruns, which is what allows the UI to call `resume_after_review(graph, config)` after the human submits feedback.
