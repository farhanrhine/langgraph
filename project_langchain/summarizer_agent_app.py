import streamlit as st
from typing import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# -------------------------
# Step 1 — Define State
# -------------------------
class GraphState(TypedDict):
    user_input: str
    summary_output: str


# -------------------------
# Step 2 — Nodes
# -------------------------
def input_node(state: GraphState) -> dict:
    # No state change here
    return {}

llm = ChatOllama(model="qwen3:0.6b")

def summarizer_node(state: GraphState) -> dict:
    user_text = state["user_input"]
    prompt = f"Summarize this concisely:\n\n{user_text}"
    summary = llm.invoke([HumanMessage(content=prompt)]).content
    return {"summary_output": summary}

def save_node(state: GraphState) -> dict:
    # No change needed
    return {}


# -------------------------
# Step 3 — Build Graph
# -------------------------
workflow = StateGraph(GraphState)

workflow.add_node("input_handler", input_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("saver", save_node)

workflow.add_edge("input_handler", "summarizer")
workflow.add_edge("summarizer", "saver")
workflow.add_edge("saver", END)

workflow.set_entry_point("input_handler")
app_graph = workflow.compile()


# -------------------------
# Step 4 — Streamlit UI
# -------------------------
st.title("🧠 Summarizer Agents (LangGraph + Ollama)")
st.write("Type text below and click **Summarize**")

user_text = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if user_text.strip():
        with st.spinner("Generating summary..."):
            result = app_graph.invoke({"user_input": user_text}) # type: ignore
        st.success("✅ Summary Generated:")
        st.write(result["summary_output"])
    else:
        st.warning("Please enter some text first!")
