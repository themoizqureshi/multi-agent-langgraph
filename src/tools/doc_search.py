"""
Local document search tool.

Two backends, selected by environment variable:

  RAG_API_URL set  →  calls /chat on the rag-chatbot-langchain API (Stage 5b integration)
  RAG_API_URL unset → falls back to direct ChromaDB similarity search (original behaviour)

The API path is preferred in production; the ChromaDB path works offline/locally.
"""

import logging
import os
from typing import Optional

import httpx
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_RAG_API_URL = os.getenv("RAG_API_URL", "").rstrip("/")


def get_doc_search_tool(persist_directory: str = CHROMA_PATH, k: int = 4):
    """
    Return a ChromaDB retriever over a pre-built local vector store.

    Returns None if no ChromaDB exists yet — the Retriever agent handles
    this gracefully by skipping local search.
    """
    if not os.path.exists(persist_directory):
        logger.warning(f"No ChromaDB found at {persist_directory}. Local doc search disabled.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def _search_via_api(query: str) -> str:
    """Call /chat on the rag-chatbot-langchain API and format the result."""
    try:
        resp = httpx.post(
            f"{_RAG_API_URL}/chat",
            json={"question": query},
            timeout=45.0,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        parts = [f"[RAG API Answer]\n{answer}"]
        for i, s in enumerate(sources[:4]):
            if isinstance(s, dict):
                src = s.get("source", "doc")
                page = s.get("page", "?")
                content = s.get("content", "")
                parts.append(f"[Local Doc {i+1} — {src}, p.{page}]\n{content}")
        return "\n\n---\n\n".join(parts)
    except httpx.HTTPError as e:
        logger.warning(f"RAG API call failed ({e}), falling back to ChromaDB")
        return None


def search_local_docs(query: str, persist_directory: str = CHROMA_PATH) -> str:
    """
    Run a similarity search and return formatted results as a string.

    Tries the RAG API first (if RAG_API_URL is set), then falls back to direct ChromaDB.
    Called directly by the retriever agent.
    """
    if _RAG_API_URL:
        result = _search_via_api(query)
        if result is not None:
            logger.info("doc_search: answered via RAG API")
            return result
        # fall through to ChromaDB on API failure

    retriever = get_doc_search_tool(persist_directory)
    if retriever is None:
        return "No local documents available. Local document search skipped."

    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant local documents found for this query."

    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        results.append(f"[Local Doc {i+1} — {source}, p.{page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)
