"""FastMCP aggregator for Pandas, RAG and Tavily."""

import logging
import sys
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import truststore
truststore.inject_into_ssl()

from fastmcp import Client, FastMCP

from config import settings

# Set up logging before the tool modules are imported, so this process writes its own file.
settings.setup_logging("aggregator.log")

from pandas_mcp import pandas_server as pandas_tools  # noqa: E402
from rag_mcp import rag_server as rag_tools  # noqa: E402
from web_search_mcp import web_search_config as web  # noqa: E402

log = logging.getLogger(__name__)

mcp = FastMCP("aggregator")

def tavily_url() -> str:
    """Build the Tavily hosted MCP URL without exposing the key in logs."""
    parts = urlsplit(web.URL)

    if not web.TOKEN:
        raise RuntimeError(
            "WEB_SEARCH_MCP_TOKEN is missing from .env. "
            "Add your Tavily API key and restart the application."
        )

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["tavilyApiKey"] = web.TOKEN

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query),
        parts.fragment,
    ))

async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily's hosted MCP server."""

    query = (query or "").strip()
    if not query:
        raise ValueError("Search query cannot be empty.")

    if not web.SEARCH_TOOL:
        raise RuntimeError("WEB_SEARCH_TOOL is not configured.")

    limit = max(1, min(int(max_results), int(web.MAX_RESULTS)))

    # Tavily supports the API key in the hosted MCP URL.
    # This avoids relying on version-specific custom-header arguments.
    async with Client(tavily_url()) as client:
        # Tavily has renamed its tool before (tavily-search -> tavily_search); accept either.
        names = [t.name for t in await client.list_tools()]
        tool_name = web.SEARCH_TOOL
        if tool_name not in names:
            alt = tool_name.replace("-", "_") if "-" in tool_name else tool_name.replace("_", "-")
            if alt not in names:
                raise RuntimeError(f"Tavily has no tool '{tool_name}'. Available: {names}")
            tool_name = alt
        result = await client.call_tool(
            tool_name,
            {
                "query": query,
                "max_results": limit,
            },
            raise_on_error=False,
        )

    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise RuntimeError(f"Tavily returned an error: {result}")

    blocks = getattr(result, "content", result)

    if isinstance(blocks, str):
        return blocks

    return "\n".join(
        block.text if hasattr(block, "text") else str(block)
        for block in blocks
    )

for tool in (*pandas_tools.TOOLS, *rag_tools.TOOLS, web_search):
    mcp.tool(tool)

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host=settings.AGGREGATOR_HOST,
        port=settings.AGGREGATOR_PORT,
        path=settings.AGGREGATOR_PATH,
    )