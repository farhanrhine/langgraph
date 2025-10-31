# 1️⃣ Imports 
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import requests
from datetime import datetime
from sympy import symbols, sympify, diff, integrate, limit, solve

# 2️⃣ Define Graph State - FIX: Use proper message structure
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]

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


# 🌐 Search Tool - FIX: Use DuckDuckGoSearchResults like code #1
search_tool = DuckDuckGoSearchResults()

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

# 5️⃣ LLM Node - FIX: Proper response handling
def call_model(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 6️⃣ Tool Node
tool_node = ToolNode([math_tool, search_tool, wikipedia, weather_tool, time_tool])

# 7️⃣ Router - FIX: Check tool_calls properly
def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 8️⃣ Build Workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# After tools, go back to agent
workflow.add_edge("tools", "agent")

# 9️⃣ Compile
app = workflow.compile()

# # 🔟 Run Example - FIX: Pass proper message format
# if __name__ == "__main__":
#     result = app.invoke({
#         "messages": [("user", "What's the weather in Mumbai and what is 12*8? Also tell time now.")]
#     })
    
#     print("\n" + "="*50)
#     print("FINAL RESULT:")
#     print("="*50)
#     # print(result)
    
#     for msg in result["messages"]:
#         if hasattr(msg, "content"):
#             print(f"\n{msg.__class__.__name__}: {msg.content}")