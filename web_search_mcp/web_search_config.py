"""Settings for Tavily's hosted MCP server. The aggregator connects to it directly."""
import os

from config import settings

URL = settings.WEB_SEARCH_MCP_URL
TOKEN = settings.WEB_SEARCH_MCP_TOKEN  # your Tavily API key, sent in the MCP URL
SEARCH_TOOL = os.getenv("WEB_SEARCH_TOOL") or "tavily-search"
MAX_RESULTS = 10
