from typing import TypedDict, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ------------------ TOOLS ------------------

@tool
def chat_gpt(question: str) -> str:
    """
    Sends a question to GPT-like logic and returns a simple response.
    """
    return f"[GPT Answer] → {question}"


@tool
def math_tool(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return "Invalid math expression"


@tool
def hello_tool(name: str) -> str:
    """
    Returns a friendly greeting message.
    """
    return f"Hello {name}! Hope you're doing great 🙂"


TOOLS = {
    "chat_gpt": chat_gpt,
    "math_tool": math_tool,
    "hello_tool": hello_tool,
}


# ------------------ STATE ------------------

class State(TypedDict):
    messages: List[Dict[str, Any]]


# ------------------ MODEL ------------------

# model = ChatOllama(
#     model="llama3.2:3b",   # Change model name if needed
#     temperature=0.7
# )
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    # IMPORTANT: Ensure your Ollama host is correctly specified here if it's not the default
    # base_url="http://localhost:11434" 
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{input}")
])

def model_node(state: State):
    last_user_message = state["messages"][-1]["content"]
    response = llm.invoke(last_user_message)
    state["messages"].append({"role": "assistant", "content": response.content})
    return state


# ------------------ TOOL NODE ------------------

def tool_node(state: State):
    user_msg = state["messages"][-1]["content"]

    if user_msg.startswith("/math"):
        expr = user_msg.replace("/math", "").strip()
        result = math_tool(expr)

    elif user_msg.startswith("/hello"):
        name = user_msg.replace("/hello", "").strip()
        result = hello_tool(name)

    else:
        result = chat_gpt(user_msg)

    state["messages"].append({"role": "assistant", "content": result})
    return state


# ------------------ ROUTER ------------------

def router(state: State):
    last = state["messages"][-1]["content"]
    if last.startswith("/"):     # Commands like /math /hello
        return "tools"
    return "model"


# ------------------ GRAPH BUILD ------------------

graph = StateGraph(State)

graph.add_node("model", model_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("model")

graph.add_conditional_edges(
    "model",
    router,
    {
        "tools": "tools",
        END: END,
    }
)

graph.add_edge("tools", END)

app = graph.compile()


# ------------------ RUN EXAMPLES ------------------
if __name__ == "__main__":
    print("\n✅ LangGraph Loaded Successfully!\n")

    while True:
        user = input("You: ")
        state = {"messages": [{"role": "user", "content": user}]}
        output = app.invoke(state)
        print("AI:", output["messages"][-1]["content"], "\n")
