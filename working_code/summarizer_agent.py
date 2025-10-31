"""
📌 Mini Project: Summarizer agent
Node 1: Take user input.
Node 2: Summarize with LLM.
Node 3: Save summary to state & return.

"""

# step 1. define state
from typing import TypedDict # TypedDict is a special type hint, that allows you to define a dictionary type with a specific set of keys and their corresponding value types. This provides static type checking for dictionaries, helping to catch errors at development time rather than runtime.
class GraphState(TypedDict):
    user_input : str
    summary_output : str

# step 2 cteate nodes

## node1 take user input
def input_node(state : GraphState) -> dict:
    print(f"USER ASKED : {state['user_input']}")
    return {}

## node2 llm node 
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="qwen3:0.6b")

def summarizer_node(state:GraphState) -> dict:
    user_text = state["user_input"]
    prompt = f"summarize this concisely:\n\n{user_text}"
    summary = llm.invoke([HumanMessage(content=prompt)]).content
    print("\nSUMMARY GENERATION STARTED.......")
    return {"summary_output": summary}

## Node 3 — Save Summary (Output)
def save_node(state: GraphState) -> dict:
    print("\n--------------------FINAL SUMMARY----------------\n")
    print(state["summary_output"])
    return {}

# Step 3 – Build and Compile Graph (orchestrator)
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(GraphState)

## add nodes
# add_node(name_of_node, function_name_of_node)
workflow.add_node("input_handler",input_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("saver",save_node)

## Connect them in sequence using edge
# input_handler ->  summarizer ->  saver ->  END
# add_edge(source_node, target_node):
workflow.add_edge("input_handler", "summarizer")
workflow.add_edge("summarizer","saver")
workflow.add_edge("saver",END)

# ENTRY 
# set_entry_point(name_of_node)
workflow.set_entry_point("input_handler")

# compile all workflow at a single point
app = workflow.compile()

# Step 4 – Run It
result = app.invoke({"user_input": "The sky is blue the grass is green A quiet scene a peaceful dream The wind blows soft the trees stand tall A gentle call embracing all The sun shines bright a golden hue On everything both old and new A silent thought a feeling deep While others wake and others sleep The world moves fast a steady beat From busy street to worn retreat No punctuation stops the flow Just simple truths that spirits know A hundred words a quickened pace To find a rhyme in time and space The story told the poem done Beneath the moon beyond the sun"}) # type: ignore
print("\n Final state returned by Agent......:\n ")
print(result)





