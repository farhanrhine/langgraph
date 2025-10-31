from typing import TypedDict, List
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

import requests
from datetime import datetime
from sympy import symbols, sympify, diff, integrate, limit, solve

# -------- State --------
class GraphState(TypedDict):
    messages: List[BaseMessage]

# -------- Tools --------
@tool
def math_tool(expression: str, operation: str = "evaluate") -> str:
    """Symbolic math using SymPy. Supported operations: evaluate, differentiate, integrate, limit, solve."""
    try:
        x = symbols('x')
        expr = sympify(expression)
        if operation == "evaluate":
            return str(expr)
        if operation == "differentiate":
            return str(diff(expr, x))
        if operation == "integrate":
            return str(integrate(expr, x))
        if operation == "limit":
            return str(limit(expr, x, 0))
        if operation == "solve":
            return str(solve(expr))
        return "Unknown operation."
    except Exception as e:
        return f"Math error: {e}"

# Names: duckduckgo_search, wikipedia
search_tool = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def weather_tool(city: str) -> str:
    """Get current weather for a city via wttr.in; resilient to DNS/HTTP failures."""
    url = f"https://wttr.in/{city}?format=3"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"Weather lookup failed: {e}"

@tool
def time_tool() -> str:
    """Return current local time formatted as YYYY-MM-DD HH:MM:SS."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------- LLM --------
llm = ChatOllama(model="gpt-oss:120b-cloud")
llm_with_tools = llm.bind_tools([math_tool, search_tool, wikipedia, weather_tool, time_tool])

# -------- Nodes --------
def llm_agent(state: GraphState):
    """Call the chat model with the full message history; model may return tool_calls."""
    messages = state["messages"]
    ai = llm_with_tools.invoke(messages)
    return {"messages": messages + [ai]}

# Use ToolNode via a lightweight adapter so only the last AIMessage is processed
from langgraph.prebuilt import ToolNode
tool_node = ToolNode([math_tool, search_tool, wikipedia, weather_tool, time_tool])

def tool_executor(state: GraphState):
    """Execute tools for the last AIMessage that has tool_calls, appending ToolMessages."""
    messages = state["messages"]
    last = messages[-1]
    if not (isinstance(last, AIMessage) and getattr(last, "tool_calls", None)):
        return {"messages": messages}
    result = tool_node.invoke({"messages": [last]})
    tool_messages = result.get("messages", [])
    return {"messages": messages + tool_messages}

def route_to_tool_or_end(state: GraphState):
    """Route to tools if last message has tool_calls; otherwise finish."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool_executor"
    return "final_answer"

def final_answer(state: GraphState):
    """Compose a friendly summary from ToolMessages; fallback to last AI content if none."""
    messages = state["messages"]

    weather = None
    math_res = None
    time_val = None
    wiki = None
    search = None

    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.name == "weather_tool":
                weather = msg.content.strip()
            elif msg.name == "math_tool":
                math_res = msg.content.strip()
            elif msg.name == "time_tool":
                time_val = msg.content.strip()
            elif msg.name == "wikipedia":
                wiki = msg.content.strip()
            elif msg.name == "duckduckgo_search":
                search = msg.content.strip()

    final_reply = "Here are the results:\n\n"
    if search:
        final_reply += f"🔍 Search Result:\n{search}\n\n"
    if wiki:
        final_reply += f"📘 Wikipedia Info:\n{wiki}\n\n"
    if weather:
        final_reply += f"🌤 Weather: {weather}\n\n"
    if math_res:
        final_reply += f"🧮 Math Result: {math_res}\n\n"
    if time_val:
        final_reply += f"⏱ Current Time: {time_val}\n\n"

    if final_reply.strip() == "Here are the results:":
        last = messages[-1]
        final_reply = getattr(last, "content", str(last))

    return {"messages": messages + [AIMessage(content=final_reply)]}

# -------- Graph --------
from langgraph.graph import StateGraph, END, START

workflow = StateGraph(GraphState)
workflow.add_node("llm_agent", llm_agent)
workflow.add_node("tool_executor", tool_executor)
workflow.add_node("final_answer", final_answer)

workflow.add_edge(START, "llm_agent")
workflow.add_conditional_edges(
    "llm_agent",
    route_to_tool_or_end,
    {"tool_executor": "tool_executor", "final_answer": "final_answer"},
)
workflow.add_edge("tool_executor", "llm_agent")
workflow.add_edge("final_answer", END)

app = workflow.compile()

if __name__ == "__main__":
    user_q = "What's the weather in Mumbai and what is 12*8? Also tell time now."
    result = app.invoke({"messages": [HumanMessage(content=user_q)]})
    print(getattr(result["messages"][-1], "content", result["messages"][-1]))
