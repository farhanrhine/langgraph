# 1️⃣ Imports
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict

# 2️⃣ Define Graph State
class GraphState(TypedDict):
    messages: list
    tool_call: dict | None

# 3️⃣ Tools
@tool
def calculator_tool(a: float, b: float, op: str = "add") -> float:
    """Simple calculator tool"""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "div":
        return a / b
    else:
        return "Invalid operation" # type: ignore

search_tool = DuckDuckGoSearchRun()


# 4️⃣ LLM + tool binding
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

# Change the model to a known compatible one for tool calling
# llm_raw = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation"
# ) # type: ignore

# llm = ChatHuggingFace(llm=llm_raw)


llm = ChatOllama(
    model="gpt-oss:120b-cloud"
)


llm_with_tools = llm.bind_tools([calculator_tool, search_tool])

# 5️⃣ LLM Node
def llm_agent(state: GraphState):
    response = llm_with_tools.invoke(state["messages"])
    tool_call = getattr(response, "tool_calls", None)
    return {"messages": state["messages"] + [response], "tool_call": tool_call}

# 6️⃣ Tool Node
tool_node = ToolNode([calculator_tool, search_tool])

# 7️⃣ Decider — pure routing function
def route_to_tool_or_end(state: GraphState) -> str:
    if state.get("tool_call"):
        return "tool_executor"
    else:
        return END

# 8️⃣ Build Workflow
workflow = StateGraph(GraphState)

workflow.add_node("llm_agent", llm_agent)
workflow.add_node("tool_executor", tool_node)

# ✅ Sequential edge: start with LLM
workflow.set_entry_point("llm_agent")

# ✅ Conditional routing: from LLM → tool or END
workflow.add_conditional_edges(
    "llm_agent", route_to_tool_or_end,
    {"tool_executor": "tool_executor", END: END}
)

# ✅ Loop: After tool → back to LLM
workflow.add_edge("tool_executor", "llm_agent")

# 9️⃣ Compile
app = workflow.compile()

# 🔟 Run
result = app.invoke({"messages": [" 'What is the square root of 144?' And ' who founded SpaceX?' "]}) # type: ignore
print(result)
