"""
Streamlit UI for the multi-agent research assistant.
Run with: streamlit run app.py
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

from src.graph import run_until_interrupt, resume_after_review
from src.agents.researcher import _circuit

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Research Assistant")
st.caption("Powered by LangGraph · Gemini 2.0 Flash · Tavily · ChromaDB")

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = None
if "graph_config" not in st.session_state:
    st.session_state.graph_config = None
if "draft_report" not in st.session_state:
    st.session_state.draft_report = None
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "stage" not in st.session_state:
    st.session_state.stage = "input"  # input → researching → review → complete
if "run_cost" not in st.session_state:
    st.session_state.run_cost = None  # populated after research completes
if "search_failed" not in st.session_state:
    st.session_state.search_failed = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(
        "**How it works:**\n\n"
        "1. Enter a research question\n"
        "2. Researcher searches the web (Tavily)\n"
        "3. Retriever searches local docs (ChromaDB)\n"
        "4. Writer synthesizes a report\n"
        "5. You review and approve (human-in-the-loop)\n"
        "6. Final report is delivered"
    )

    st.markdown("---")
    st.markdown("**Agent Pipeline**")
    agents = [
        ("🔍 Researcher", "Web search via Tavily (retry + circuit breaker)"),
        ("📚 Retriever", "Local docs via ChromaDB"),
        ("✍️ Writer", "Synthesize report"),
        ("👤 Human Review", "You approve or give feedback"),
    ]
    for name, desc in agents:
        st.markdown(f"**{name}** — {desc}")

    st.markdown("---")
    # Circuit breaker status
    if _circuit["open"]:
        st.error(f"⚡ Circuit breaker OPEN — web search disabled after {_circuit['failures']} failures")
    elif _circuit["failures"] > 0:
        st.warning(f"⚠️ Web search: {_circuit['failures']}/{3} failures (circuit closes at 3)")
    else:
        st.success("✅ Web search: healthy")

    # Cost summary for the last run
    if st.session_state.run_cost:
        cost = st.session_state.run_cost
        st.markdown("---")
        st.markdown("**Last run cost**")
        st.caption(
            f"Tokens: {cost['input']:,} in / {cost['output']:,} out  \n"
            f"Est. cost: ${cost['usd']:.6f}"
        )

    if st.button("🔄 Start New Research"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.graph = None
        st.session_state.graph_config = None
        st.session_state.draft_report = None
        st.session_state.final_report = None
        st.session_state.run_cost = None
        st.session_state.search_failed = False
        st.session_state.stage = "input"
        st.rerun()


# ── Stage: Input ──────────────────────────────────────────────────────────────
if st.session_state.stage == "input":
    st.subheader("What would you like to research?")
    question = st.text_area(
        "Research question",
        placeholder="e.g. What are the latest developments in RAG systems for enterprise use?",
        height=100,
    )

    if st.button("🚀 Start Research", type="primary", disabled=not question.strip()):
        st.session_state.stage = "researching"
        st.session_state.current_question = question.strip()
        st.rerun()


# ── Stage: Researching ────────────────────────────────────────────────────────
elif st.session_state.stage == "researching":
    st.subheader(f"📋 Researching: *{st.session_state.current_question}*")

    with st.spinner("Running multi-agent pipeline... (this may take 30-60 seconds)"):
        progress = st.progress(0, text="Researcher agent: searching the web...")
        try:
            graph, state, config = run_until_interrupt(
                question=st.session_state.current_question,
                thread_id=st.session_state.thread_id,
            )
            progress.progress(100, text="Research complete — awaiting your review")

            # Capture cost and fallback status for sidebar display
            pricing = 0.40 / 1_000_000  # output token price (Gemini Flash)
            input_usd = state.get("total_input_tokens", 0) * (0.10 / 1_000_000)
            output_usd = state.get("total_output_tokens", 0) * pricing
            st.session_state.run_cost = {
                "input": state.get("total_input_tokens", 0),
                "output": state.get("total_output_tokens", 0),
                "usd": round(input_usd + output_usd, 6),
            }
            st.session_state.search_failed = state.get("search_failed", False)

            st.session_state.graph = graph
            st.session_state.graph_config = config
            st.session_state.draft_report = state.get("final_report", "No report generated.")
            st.session_state.stage = "review"
            st.rerun()

        except Exception as e:
            st.error(f"Research failed: {e}")
            st.session_state.stage = "input"


# ── Stage: Human Review ───────────────────────────────────────────────────────
elif st.session_state.stage == "review":
    st.subheader("👤 Human Review Checkpoint")
    if st.session_state.search_failed:
        st.warning("Web search was unavailable — report is based on local documents only.")
    st.info("The agents have completed their research. Review the draft report below, then approve or provide feedback.")

    st.markdown("### Draft Report")
    st.markdown(st.session_state.draft_report)

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        feedback = st.text_area(
            "Optional feedback for revision (leave blank to approve as-is)",
            placeholder="e.g. Please add more detail on the cost comparison section.",
            height=80,
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("✅ Approve & Finalize", type="primary"):
            with st.spinner("Finalizing report..."):
                final_state = resume_after_review(
                    st.session_state.graph,
                    st.session_state.graph_config,
                    feedback=feedback or "",
                )
                st.session_state.final_report = st.session_state.draft_report
                st.session_state.stage = "complete"
                st.rerun()


# ── Stage: Complete ───────────────────────────────────────────────────────────
elif st.session_state.stage == "complete":
    st.success("✅ Research complete!")
    st.subheader("📄 Final Report")
    st.markdown(st.session_state.final_report)

    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=st.session_state.final_report,
        file_name="research_report.md",
        mime="text/markdown",
    )
