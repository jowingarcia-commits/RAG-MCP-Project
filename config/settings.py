"""Central configuration. Values come from .env; empty values fall back to defaults."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or default


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def _flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# LLM (OpenRouter speaks the OpenAI API)
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = _env("MODEL", "openrouter/free")
MAX_TOKENS = int(_env("MAX_TOKENS", "1500"))
MAX_TOOL_ROUNDS = int(_env("MAX_TOOL_ROUNDS", "10"))

# jowin
# LLM (Anthropic Claude API)
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
MODEL = _env("MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(_env("MAX_TOKENS", "1500"))
MAX_TOOL_ROUNDS = int(_env("MAX_TOOL_ROUNDS", "10"))

# MCP: the aggregator is our own server; web search is Tavily's hosted MCP server
AGGREGATOR_HOST = _env("AGGREGATOR_HOST", "127.0.0.1")
AGGREGATOR_PORT = int(_env("AGGREGATOR_PORT", "8000"))
AGGREGATOR_PATH = _env("AGGREGATOR_PATH", "/mcp")
AGGREGATOR_URL = _env("AGGREGATOR_URL", f"http://{AGGREGATOR_HOST}:{AGGREGATOR_PORT}{AGGREGATOR_PATH}")
WEB_SEARCH_MCP_TOKEN = _env("WEB_SEARCH_MCP_TOKEN")
WEB_SEARCH_MCP_URL = _env("WEB_SEARCH_MCP_URL", "https://mcp.tavily.com/mcp/")


# Flask
HOST = _env("FLASK_HOST", "127.0.0.1")
PORT = int(_env("FLASK_PORT", "5000"))
DEBUG = _flag(_env("FLASK_DEBUG", "False"))
SECRET_KEY = _env("SECRET_KEY", "dev-only-change-me")

# RAG
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = _path(_env("VECTOR_STORE_PATH", "./vector_store"))

# Data
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = _path(_env("UPLOAD_FOLDER", "./uploads"))
LOG_DIR = BASE_DIR / "logs"
DEFAULT_DATASET = _env("DEFAULT_DATASET", "titanic.csv")
DATASET_EXTENSIONS = {".csv", ".xlsx"}
DOCUMENT_EXTENSIONS = {".txt", ".md"}
ALLOWED_UPLOADS = DATASET_EXTENSIONS | DOCUMENT_EXTENSIONS
MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", "20"))

for _d in (DATA_DIR, UPLOAD_DIR, LOG_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def setup_logging(filename: str = "app.log") -> None:
    """Log to logs/<filename>, never to stdout. Each process should use its own file."""
    root = logging.getLogger()
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return
    handler = RotatingFileHandler(LOG_DIR / filename, maxBytes=1_000_000, backupCount=3,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    if sys.stdout.isatty():
        root.addHandler(logging.StreamHandler(sys.stderr))
