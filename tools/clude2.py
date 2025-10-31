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
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

# 2️⃣ Define Graph State
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


search_tool = DuckDuckGoSearchResults()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) # type: ignore

@tool
def weather_tool(city: str) -> str:
    """Get current weather conditions for any city."""
    url = f"https://wttr.in/{city}?format=3"
    try:
        return requests.get(url, timeout=5).text
    except:
        return "Weather lookup failed."

@tool
def time_tool() -> str:
    """Returns current local time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 4️⃣ LLM + tool binding
# llm = ChatOllama(model="gpt-oss:120b-cloud")
llm = ChatOllama(model="qwen3:0.6b")

llm_with_tools = llm.bind_tools([math_tool, search_tool, wikipedia, weather_tool, time_tool])

# 5️⃣ LLM Node with Debug Info
def call_model(state: GraphState):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}🤖 CALLING LLM...")
    print(f"{Fore.CYAN}{'='*60}")
    
    response = llm_with_tools.invoke(state["messages"])
    
    # Check if tools were called
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"\n{Fore.GREEN}✅ LLM DECIDED TO USE TOOLS:")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"{Fore.YELLOW}   {i}. Tool: {Fore.WHITE}{tool_call['name']}")
            print(f"{Fore.YELLOW}      Args: {Fore.WHITE}{tool_call['args']}")
    else:
        print(f"\n{Fore.MAGENTA}💬 LLM RESPONDED WITHOUT TOOLS")
    
    return {"messages": [response]}

# 6️⃣ Tool Node with Debug Info
tools_list = [math_tool, search_tool, wikipedia, weather_tool, time_tool]
tool_node = ToolNode(tools_list)

def execute_tools(state: GraphState):
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.GREEN}🔧 EXECUTING TOOLS...")
    print(f"{Fore.GREEN}{'='*60}")
    
    result = tool_node.invoke(state)
    
    # Show tool results
    if "messages" in result:
        for msg in result["messages"]:
            if hasattr(msg, "name"):
                print(f"\n{Fore.BLUE}📊 Tool Result from: {Fore.WHITE}{msg.name}")
                content = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
                print(f"{Fore.BLUE}   Output: {Fore.WHITE}{content}")
    
    return result

# 7️⃣ Router
def should_continue(state: GraphState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# 8️⃣ Build Workflow
workflow = StateGraph(GraphState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

# 9️⃣ Compile
app = workflow.compile()

# 🔟 Test Function with Statistics
def test_agent(question: str):
    print(f"\n\n{Fore.MAGENTA}{'#'*70}")
    print(f"{Fore.MAGENTA}# QUESTION: {question}")
    print(f"{Fore.MAGENTA}{'#'*70}")
    
    result = app.invoke({"messages": [("user", question)]})
    
    # Analyze results
    tools_used = []
    for msg in result["messages"]:
        if hasattr(msg, "name") and msg.name:
            tools_used.append(msg.name)
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}📈 EXECUTION SUMMARY:")
    print(f"{Fore.CYAN}{'='*70}")
    print(f"{Fore.GREEN}✅ Tools Used: {Fore.WHITE}{len(set(tools_used))}")
    if tools_used:
        for tool in set(tools_used):
            count = tools_used.count(tool)
            print(f"{Fore.YELLOW}   • {tool}: {Fore.WHITE}{count} time(s)")
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}🎯 FINAL ANSWER:")
    print(f"{Fore.CYAN}{'='*70}")
    
    # Get final answer
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            print(f"{Fore.WHITE}{msg.content}")
            break
    
    return result


# 🧪 Run Tests
if __name__ == "__main__":
    # Easy test
    # test_agent("What's the weather in Mumbai and what is 12*8?")
    
    # Medium test
    # test_agent("Differentiate x^3 + 2x^2 and tell me the current time")
    
    # Hard test
    test_agent("Search for latest AI news, check weather in Tokyo, calculate 25*25, and tell time")
    
    # Choose your test from the list above!