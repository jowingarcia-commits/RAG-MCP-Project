"""Connects to the running aggregator server through langchain-mcp-adapters and exposes a sync API."""
import asyncio
import threading

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings


def to_text(output) -> str:
    """Tool output may be a string or a list of content blocks, depending on version."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in output)
    return str(output)


class MCPClient:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._client = MultiServerMCPClient(
            {"aggregator": {"transport": "streamable_http", "url": settings.AGGREGATOR_URL}})
        self._tools = None
        self._lock = threading.Lock()

    def _run(self, coro, timeout: int = 120):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def get_tools(self) -> list:
        """LangChain tools, loaded once."""
        with self._lock:
            if self._tools is None:
                self._tools = self._run(self._client.get_tools())
            return self._tools

    def call_tool(self, tool, arguments: dict) -> str:
        return to_text(self._run(tool.ainvoke(arguments)))
