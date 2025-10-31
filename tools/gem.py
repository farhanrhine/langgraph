## 🚀 Corrected LangGraph Agent Code (Code 2)

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
# Corrected core imports:
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage # <--- Added HumanMessage for user input


# 2️⃣ Define Graph State
class GraphState(TypedDict):
    # State holds a list of messages (HumanMessage, AIMessage, ToolMessage)
    messages: list[BaseMessage]
    # Tool call info is needed to route the graph correctly
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
            # For simple arithmetic, eval() works better than sympify(). For complex expressions, 
            # we rely on SymPy's string conversion.
            return str(expr.evalf(subs={x: x}) if x in expr.free_symbols else expr.evalf())

        elif operation == "differentiate":
            return str(diff(expr, x))

        elif operation == "integrate":
            return str(integrate(expr, x))

        elif operation == "limit":
            # Assuming limit as x->0 if not specified
            return str(limit(expr, x, 0))

        elif operation == "solve":
            return str(solve(expr, x))

        else:
            return "Unknown operation."

    except Exception as e:
        return f"Math error: {e}"


# 🌐 Search Tool
# DuckDuckGoSearchRun has the name 'duckduckgo_search'
search_tool = DuckDuckGoSearchRun()

# 🧠 Wikipedia Tool
# WikipediaQueryRun has the name 'wikipedia_query_run'
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) # type: ignore

# 🌤️ Weather Tool (Free - no key)
# The function name is 'weather_tool'
@tool
def weather_tool(city: str) -> str:
    """Get current weather conditions for any city."""
    url = f"https://wttr.in/{city}?format=3"
    try:
        # Note: This tool is a common source of the ConnectError if network is down.
        return requests.get(url, timeout=5).text
    except Exception as e:
        # Return a clear error message instead of failing the graph
        return f"Weather lookup failed: {e}"

# ⏱️ Time Tool
# The function name is 'time_tool'
@tool
def time_tool() -> str:
    """Returns current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Combine all tools
all_tools = [math_tool, search_tool, wikipedia, weather_tool, time_tool]

# 4️⃣ LLM + tool binding
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    # IMPORTANT: Ensure your Ollama host is correctly specified here if it's not the default
    # base_url="http://localhost:11434" 
)
llm_with_tools = llm.bind_tools(all_tools)


# # 5️⃣ LLM Node
# def llm_agent(state: GraphState):
#     # The LLM is invoked with the ENTIRE message history
#     response = llm_with_tools.invoke(state["messages"])
#     # Tool calls are extracted from the response
#     # LangChain versions can vary how this is returned; using .tool_calls is standard for a bound model
#     tool_call = getattr(response, "tool_calls", None) 
    
#     # Check for empty list or None
#     if not tool_call:
#         tool_call = None
        
#     # Return the updated state with the LLM's new message and any tool call
#     return {"messages": state["messages"] + [response], "tool_call": tool_call}

# 5️⃣ LLM Node (Revised for robustness)
from langchain_core.messages import BaseMessage # Ensure this is imported

def llm_agent(state: GraphState) -> GraphState:
    """Invokes the LLM and extracts tool calls."""
    messages = state["messages"]
    
    # 1. Invoke the LLM with the full message history
    try:
        response: BaseMessage = llm_with_tools.invoke(messages)
    except Exception as e:
        # If LLM connection fails, return an error message to prevent graph crash
        return {
            "messages": messages + [AIMessage(content=f"LLM Invocation Error: {e}")], 
            "tool_call": None
        }

    # 2. Extract tool calls (checking both attributes for compatibility)
    tool_calls = getattr(response, "tool_calls", [])
    
    # If the LLM response is missing tool_calls or it's an empty list/None
    if not tool_calls:
        # If no tool calls, set to None for the router to catch
        return {"messages": messages + [response], "tool_call": None}
    
    # 3. If tool calls exist, package them for the ToolNode
    # ToolNode expects a list of tool call objects, and we pass the LLM response itself 
    # as the new message in the history.
    return {
        "messages": messages + [response], 
        "tool_call": tool_calls
    }

# NOTE: The ToolNode will automatically execute the calls and update the messages 
# state with the ToolMessage results before the graph loops back to llm_agent.


# 6️⃣ Tool Node
tool_node = ToolNode(all_tools)


# 7️⃣ Router
# def route_to_tool_or_end(state: GraphState) -> str:
#     # Router checks if the LLM returned a tool call
#     if state.get("tool_call"):
#         return "tool_executor"
#     else:
#         # If no tool call, the answer is final (or needs a summary)
#         return "final_answer"
def route_to_tool_or_end(state: GraphState) -> str:
    # It checks the 'tool_call' key set by the llm_agent node.
    if state.get("tool_call"):
        return "tool_executor"
    else:
        return "final_answer"
    
    
# 8️⃣ Final Answer Node (This is for summarization/presentation)
def final_answer(state: GraphState):
    """Summarize all tool outputs into a clean human-readable answer."""
    messages = state["messages"]
    
    # Initialize dictionary to hold tool results
    tool_results = {
        "search": None,
        "wiki": None,
        "weather": None,
        "math": None,
        "time": None,
    }

    # Iterate through messages to find ToolMessage contents
    for msg in messages:
        # Tool results are stored in ToolMessage objects
        if isinstance(msg, ToolMessage):
            # Tool names must exactly match the function name or internal tool name
            if msg.name == "weather_tool":
                tool_results["weather"] = msg.content.strip()
            elif msg.name == "math_tool":
                tool_results["math"] = msg.content.strip()
            elif msg.name == "time_tool":
                tool_results["time"] = msg.content.strip()
            elif msg.name == "wikipedia_query_run":
                tool_results["wiki"] = msg.content.strip()
            elif msg.name == "duckduckgo_search":
                tool_results["search"] = msg.content.strip()

    final_reply = "Here are the results:\n\n"

    if tool_results["search"]:
        final_reply += f"🔍 **Search Result:**\n{tool_results['search']}\n\n"
    if tool_results["wiki"]:
        final_reply += f"📘 **Wikipedia Info:**\n{tool_results['wiki']}\n\n"
    if tool_results["weather"]:
        final_reply += f"🌤 **Weather:** {tool_results['weather']}\n\n"
    if tool_results["math"]:
        final_reply += f"🧮 **Math Result:** {tool_results['math']}\n\n"
    if tool_results["time"]:
        final_reply += f"⏱ **Current Time:** {tool_results['time']}\n\n"

    # If the reply is just the header, it means no tools were used or returned
    if final_reply.strip() == "Here are the results:":
        # Show the final LLM output directly
        final_reply = messages[-1].content
        
    # Return the final state with the answer as an AIMessage
    return {
        "messages": messages + [AIMessage(content=final_reply)],
        "tool_call": None
    }


# 9️⃣ Build Workflow
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

# After tool execution, go back to the LLM to process the tool results
workflow.add_edge("tool_executor", "llm_agent") 
workflow.add_edge("final_answer", END)


# 🔟 Compile
app = workflow.compile()

# 11️⃣ Run Example - CORRECTED INPUT FORMAT
# The user's input must be a list containing a HumanMessage object
user_prompt = "What's the weather in Mumbai and what is 12*8? Also tell time now."
initial_state = {"messages": [HumanMessage(content=user_prompt)]}

print("--- Invoking Agent ---")
try:
    # This might still raise a ConnectError if Ollama or wttr.in is inaccessible
    result = app.invoke(initial_state) 
    print("\n--- Final Result ---")
    # The final answer is the last message in the list
    print(result["messages"][-1].content)
except Exception as e:
    print(f"\n--- ERROR DETECTED ---")
    print(f"An error occurred: {e}")
    print("\n⚠️ The most likely cause is still the network connection to your LLM (Ollama) or the weather API (wttr.in).")