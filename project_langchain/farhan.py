# 1️⃣ Imports 
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict
import requests
from datetime import datetime
from sympy import symbols, sympify, diff, integrate, limit, solve
from langchain.tools import tool
from langchain_core.messages import AIMessage


# 2️⃣ Define Graph State
class GraphState(TypedDict):
    messages: list
    tool_call: dict | None

# 3️⃣ Tools
@tool
def math_tool(expression: str, operation: str = "evaluate") -> str:
    """
    Powerful symbolic mathematics tool using SymPy.

    - operation="evaluate" → evaluates arithmetic expression
    - operation="differentiate" → differentiation wrt x
    - operation="integrate" → indefinite integral wrt x
    - operation="limit" → compute limit as x→0 (default)
    - operation="solve" → solve equation (set expression = 0)
    """
    try:
        x = symbols('x')
        expr = sympify(expression)

        if operation == "evaluate":
            return str(expr)

        elif operation == "differentiate":
            return str(diff(expr, x))

        elif operation == "integrate":
            return str(integrate(expr, x))

        elif operation == "limit":
            return str(limit(expr, x, 0))

        elif operation == "solve":
            return str(solve(expr))

        else:
            return "Unknown operation."

    except Exception as e:
        return f"Math error: {e}"


# 🌐 Search Tool
search_tool = DuckDuckGoSearchRun()

# 🧠 Wikipedia Tool
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) # type: ignore

# 🌤️ Weather Tool (Free - no key)
@tool
def weather_tool(city: str) -> str:
    """Get current weather conditions for any city."""
    url = f"https://wttr.in/{city}?format=3"
    try:
        return requests.get(url, timeout=5).text
    except:
        return "Weather lookup failed."

# ⏱️ Time Tool
@tool
def time_tool() -> str:
    """Returns current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 4️⃣ LLM + tool binding
llm = ChatOllama(
    model="gpt-oss:120b-cloud"
)
llm_with_tools = llm.bind_tools([math_tool, search_tool, wikipedia, weather_tool, time_tool])


## here the main issue i think 
# 5️⃣ LLM Node
def llm_agent(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    tool_call = getattr(response, "tool_calls", None)
    return {"messages": state["messages"] + [response], "tool_call": tool_call}

# 6️⃣ Tool Node
tool_node = ToolNode([math_tool, search_tool, wikipedia, weather_tool, time_tool])

# 7️⃣ Router, or maybe here the main issue

def route_to_tool_or_end(state: GraphState) -> str:
    if state.get("tool_call"):
        return "tool_executor"
    else:
        return "final_answer"
    
# or maybe here the main issue
def final_answer(state: GraphState):
    """Summarize all tool outputs into a clean human-readable answer."""
    messages = state["messages"]

    weather = None
    math = None
    time = None
    wiki = None
    search = None  # <--- ADD THIS

    for msg in messages:
        if hasattr(msg, "name"):
            if msg.name == "weather_tool":
                weather = msg.content.strip()
            elif msg.name == "math_tool":
                math = msg.content.strip()
            elif msg.name == "time_tool":
                time = msg.content.strip()
            elif msg.name == "wikipedia_query_run":
                wiki = msg.content.strip()
            elif msg.name == "duckduckgo_search":  # <--- ADD THIS
                search = msg.content.strip()

    final_reply = "Here are the results:\n\n"

    if search:
        final_reply += f"🔍 **Search Result:**\n{search}\n\n"
    if wiki:
        final_reply += f"📘 **Wikipedia Info:**\n{wiki}\n\n"
    if weather:
        final_reply += f"🌤 **Weather:** {weather}\n\n"
    if math:
        final_reply += f"🧮 **Math Result:** {math}\n\n"
    if time:
        final_reply += f"⏱ **Current Time:** {time}\n\n"

    # If nothing matched → show LLM output directly
    if final_reply.strip() == "Here are the results:":
        final_reply = messages[-1].content

    return {
        "messages": messages + [AIMessage(content=final_reply)],
        "tool_call": None
    }



# 8️⃣ Build Workflow or maybe here the main issue
workflow = StateGraph(GraphState)
workflow.add_node("llm_agent", llm_agent)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("final_answer", final_answer)
workflow.set_entry_point("llm_agent")


workflow.add_conditional_edges(
    "llm_agent",
    route_to_tool_or_end,
    {
        "tool_executor": "tool_executor",
        "final_answer": "final_answer"
    }
)

workflow.add_edge("tool_executor", "llm_agent")
workflow.add_edge("final_answer", END)



# 9️⃣ Compile
app = workflow.compile()

# 🔟 Run Example
result = app.invoke({"messages": ["What's the weather in Mumbai and what is 12*8? Also tell time now."]}) # type: ignore
print(result)
