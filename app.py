"""Flask entry point. Run with: python app.py"""
import truststore
truststore.inject_into_ssl()
import atexit
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

from config import settings

settings.setup_logging()
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = settings.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = settings.MAX_UPLOAD_MB * 1024 * 1024

_histories: dict[str, list[dict]] = {}
_agent = None
_agent_lock = threading.Lock()


def get_agent():
    """Start the MCP connection on first use so the app boots without an API key."""
    global _agent
    with _agent_lock:
        if _agent is None:
            from agent.ai_agent import DataAnalyticsAgent
            from agent.mcp_client import MCPClient
            _agent = DataAnalyticsAgent(MCPClient())
    return _agent


def session_id() -> str:
    session.setdefault("sid", uuid.uuid4().hex)
    return session["sid"]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=settings.MODEL, api_key_set=bool(settings.OPENROUTER_API_KEY))


@app.get("/api/datasets")
def datasets():
    from pandas_mcp.analytics import list_datasets
    return jsonify(datasets=list_datasets())


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Choose a file first."), 400
    name = secure_filename(file.filename)
    if not name or "." not in name:
        return jsonify(error="Rename the file using letters, numbers and a file extension."), 400
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in settings.ALLOWED_UPLOADS:
        allowed = ", ".join(sorted(settings.ALLOWED_UPLOADS))
        return jsonify(error=f"Unsupported file type. Use one of: {allowed}."), 400
    file.save(settings.UPLOAD_DIR / name)
    note = "Ready to analyze."
    if suffix in settings.DOCUMENT_EXTENSIONS:
        from rag_mcp.rag_server import ingest_document
        try:
            chunks = ingest_document(name)["chunks"]
        except Exception as exc:
            log.exception("indexing %s failed", name)
            return jsonify(error=f"Saved {name}, but indexing failed: {exc}"), 500
        note = f"Indexed {chunks} passages for search."
    log.info("uploaded %s", name)
    return jsonify(filename=name, note=note)


@app.errorhandler(413)
def too_large(_exc):
    return jsonify(error=f"File is too large. The limit is {settings.MAX_UPLOAD_MB} MB."), 413


@app.post("/api/chat")
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify(error="Type a question first."), 400


    # jowin
    # if not settings.OPENROUTER_API_KEY:
    #     return jsonify(error="Add OPENROUTER_API_KEY to your .env file and restart the app."), 503
    
    history = _histories.setdefault(session_id(), [])
    try:
        answer, tools = get_agent().chat(history, message)
    except Exception as exc:  # surface the cause to the UI, full trace goes to the log
        log.exception("chat failed")
        return jsonify(error=f"The request failed: {exc}"), 500
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    del history[:-20]
    return jsonify(answer=answer, tools=tools)


@app.post("/api/reset")
def reset():
    _histories.pop(session_id(), None)
    return jsonify(status="cleared")


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def launch_servers() -> None:
    """Start the aggregator MCP server unless something already listens on its port."""
    host, port = settings.AGGREGATOR_HOST, settings.AGGREGATOR_PORT
    if _port_open(host, port):
        log.info("aggregator already running on %s:%s", host, port)
        return
    server_log = open(settings.LOG_DIR / "servers.log", "a")
    env = {**os.environ, "PYTHONPATH": str(settings.BASE_DIR)}
    proc = subprocess.Popen([sys.executable, str(settings.BASE_DIR / "aggregator" / "aggregator_server.py")],
                            cwd=settings.BASE_DIR, env=env, stdout=server_log, stderr=server_log)
    server_log.close()  # the child process keeps its own handle
    atexit.register(proc.terminate)
    for _ in range(60):
        if _port_open(host, port):
            log.info("aggregator started on %s:%s", host, port)
            return
        if proc.poll() is not None:
            raise RuntimeError("The aggregator exited. See logs/servers.log.")
        time.sleep(0.5)
    raise RuntimeError(f"The aggregator did not start on {host}:{port}. See logs/servers.log.")


if __name__ == "__main__":
    launch_servers()
    # No reloader: it would start every MCP server twice.
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG, use_reloader=False)
