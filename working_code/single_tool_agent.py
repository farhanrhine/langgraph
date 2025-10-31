# 1️⃣ Imports
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict

# 2️⃣ Define State
class GraphState(TypedDict):
    user_input: str
    messages: list

# 3️⃣ Define Tool
search_tool = DuckDuckGoSearchResults()

# 4️⃣ LLM + Bind
llm = ChatOllama(
    model="gpt-oss:120b-cloud"
)
llm_with_tools = llm.bind_tools([search_tool])

# 5️⃣ LLM Node
def call_model(state: GraphState):
    response = llm_with_tools.invoke(state["user_input"])
    return {"messages": [response]}

# 6️⃣ Tool Node
tool_node = ToolNode([search_tool])

# 7️⃣ Graph
workflow = StateGraph(GraphState)
workflow.add_node("llm", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge("llm", "tools")
workflow.add_edge("tools", END)
workflow.set_entry_point("llm")
app = workflow.compile()

# 8️⃣ Run
result = app.invoke({"user_input": "Search for today's latest news about Donald trump and what he said about his 3rd terms run for presidental election."}) # type: ignore

print(result)
