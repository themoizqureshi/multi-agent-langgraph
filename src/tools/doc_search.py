"""
Local document search tool backed by ChromaDB.

This tool lets the Retriever agent search the user's local document store —
the same vector database built by Project 1's ingestion pipeline.

Having both web search (Tavily) and local search (ChromaDB) lets the Writer
synthesize public knowledge + private document knowledge in one report.
"""

import logging
import os
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")


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


def search_local_docs(query: str, persist_directory: str = CHROMA_PATH) -> str:
    """
    Run a similarity search and return formatted results as a string.

    Called directly by the retriever agent — returns an empty string
    if no ChromaDB is available (agent handles this case gracefully).
    """
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
