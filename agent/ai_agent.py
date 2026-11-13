"""OpenRouter model with MCP tools via LangChain: asks, runs requested tools, returns the answer."""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from agent.mcp_client import MCPClient
from config import settings
from config.prompts import SYSTEM_PROMPT

from langchain_anthropic import ChatAnthropic  

log = logging.getLogger(__name__)


def _content_text(content) -> str:
    """Some models return a list of content blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else b.get("text", "") if isinstance(b, dict) else ""
                       for b in content)
    return ""


class DataAnalyticsAgent:

    # jowin
    # def __init__(self, mcp: MCPClient):
    #     self.mcp = mcp
    #     self.llm = ChatOpenAI(model=settings.MODEL, api_key=settings.OPENROUTER_API_KEY,
    #                           base_url=settings.OPENROUTER_BASE_URL, max_tokens=settings.MAX_TOKENS)

    def __init__(self, mcp: MCPClient):
        self.mcp = mcp
        self.llm = ChatAnthropic(model=settings.MODEL, api_key=settings.ANTHROPIC_API_KEY,
                                 max_tokens=settings.MAX_TOKENS)
        
    def chat(self, history: list[dict], user_message: str) -> tuple[str, list[dict]]:
        """history holds plain-text turns only. Returns (answer, tools_used)."""
        tools = self.mcp.get_tools()
        by_name = {t.name: t for t in tools}
        llm = self.llm.bind_tools(tools)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for m in history:
            cls = HumanMessage if m["role"] == "user" else AIMessage
            messages.append(cls(content=m["content"]))
        messages.append(HumanMessage(content=user_message))

        used: list[dict] = []
        reply = None
        for _ in range(settings.MAX_TOOL_ROUNDS):
            reply = llm.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                log.info("tool %s %s", call["name"], call["args"])
                try:
                    if call["name"] not in by_name:
                        raise ValueError(f"Unknown tool '{call['name']}'. Available: {sorted(by_name)}")
                    output = self.mcp.call_tool(by_name[call["name"]], call["args"])
                except Exception as exc:  # let the model see the error and retry
                    output = f"Error: {exc}"
                used.append({"tool": call["name"], "input": call["args"]})
                messages.append(ToolMessage(content=output[:20000], tool_call_id=call["id"]))

        if reply is None:
            return "No answer: MAX_TOOL_ROUNDS must be at least 1.", used
        # text = _content_text(reply.content)

        c = reply.content
        text = c if isinstance(c, str) else "".join( b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
        
        if reply.tool_calls or not text.strip():
            text = "I ran out of tool steps before finishing. Try a narrower question."
        return text.strip(), used
