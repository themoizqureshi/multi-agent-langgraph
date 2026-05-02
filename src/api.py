"""
FastAPI REST API for the multi-agent LangGraph system.

Endpoints:
  GET  /health            — Liveness check
  POST /research          — Start a research job (non-blocking, returns thread_id)
  GET  /status/{thread_id} — Poll job status
  POST /review/{thread_id} — Submit human review (approve or reject with feedback)

Design: background task + polling.
run_until_interrupt() is a blocking ~30-60s call. It runs in a thread executor
so it doesn't block the event loop. The frontend polls /status every 2 seconds.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# thread_id → job state dict
_jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Multi-agent API ready")
    yield
    _jobs.clear()


app = FastAPI(
    title="Multi-Agent Research API",
    description="LangGraph multi-agent: Researcher + Retriever + Writer with human-in-the-loop",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ──────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str


class ReviewRequest(BaseModel):
    feedback: str = ""
    approved: bool = True


# ── Background worker ────────────────────────────────────────────────────────

def _run_research_sync(question: str, thread_id: str) -> None:
    """Blocking — called in a thread executor."""
    from .graph import run_until_interrupt

    try:
        graph, state, config = run_until_interrupt(question, thread_id=thread_id)
        _jobs[thread_id].update({
            "status": "awaiting_review",
            "draft_report": state.get("final_report") or "",
            "web_results": state.get("web_search_results") or "",
            "doc_results": state.get("doc_search_results") or "",
            "_graph": graph,
            "_config": config,
        })
        logger.info(f"Job {thread_id}: awaiting human review")
    except Exception as exc:
        _jobs[thread_id].update({"status": "error", "message": str(exc)})
        logger.error(f"Job {thread_id} failed: {exc}")


def _run_resume_sync(thread_id: str, feedback: str) -> None:
    """Blocking — called in a thread executor."""
    from .graph import resume_after_review

    job = _jobs[thread_id]
    try:
        final_state = resume_after_review(job["_graph"], job["_config"], feedback=feedback)
        _jobs[thread_id].update({
            "status": "complete",
            "final_report": final_state.get("final_report") or "",
        })
        logger.info(f"Job {thread_id}: complete")
    except Exception as exc:
        _jobs[thread_id].update({"status": "error", "message": str(exc)})
        logger.error(f"Job {thread_id} resume failed: {exc}")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": len(_jobs)}


@app.post("/research")
async def start_research(req: ResearchRequest):
    """Kick off a research job. Returns immediately with a thread_id to poll."""
    thread_id = str(uuid.uuid4())
    _jobs[thread_id] = {"status": "running", "question": req.question}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_research_sync, req.question, thread_id)

    return {"thread_id": thread_id, "status": "started"}


@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    """Poll this until status is 'awaiting_review' or 'complete'."""
    job = _jobs.get(thread_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Don't return internal graph objects
    return {k: v for k, v in job.items() if not k.startswith("_")}


@app.post("/review/{thread_id}")
async def submit_review(thread_id: str, req: ReviewRequest):
    """Approve or reject the draft. If approved, the Writer finalises the report."""
    job = _jobs.get(thread_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "awaiting_review":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not awaiting_review")

    _jobs[thread_id]["status"] = "finalizing"
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_resume_sync, thread_id, req.feedback)

    return {"status": "finalizing", "thread_id": thread_id}
