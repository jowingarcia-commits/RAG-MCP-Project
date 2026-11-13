"""MCP tools for document search. Run alone with: python rag_mcp/rag_server.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmcp import FastMCP

from config.settings import DOCUMENT_EXTENSIONS, UPLOAD_DIR, setup_logging
from rag_mcp.embeddings import chunk_text
from rag_mcp.vector_store import VectorStore

setup_logging()
mcp = FastMCP("rag")
store = VectorStore()


def ingest_document(filename: str) -> dict:
    """Index a .txt or .md file from the uploads folder so it can be searched."""
    path = UPLOAD_DIR / Path(filename).name
    if path.suffix.lower() not in DOCUMENT_EXTENSIONS or not path.exists():
        raise FileNotFoundError(f"No .txt or .md file named '{path.name}' in uploads.")
    chunks = chunk_text(path.read_text(encoding="utf-8", errors="ignore"))
    return {"source": path.name, "chunks": store.add(path.name, chunks)}


def search_documents(query: str, top_k: int = 3) -> list[dict]:
    """Find the passages in uploaded documents that best match the query."""
    return store.search(query, min(top_k, 10))


def list_documents() -> list[dict]:
    """List indexed documents and how many chunks each has."""
    return store.sources()


TOOLS = [ingest_document, search_documents, list_documents]
for _tool in TOOLS:
    mcp.tool(_tool)


if __name__ == "__main__":
    mcp.run()
