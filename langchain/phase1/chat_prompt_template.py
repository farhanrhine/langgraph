# Step 1: Import
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Step 2: Define Role-Based Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful AI assistent."),
    ("user", "Please write a short paragraph about {topic} in a {style} tone"),
])

# Step 3: Fill in Variables
filled_chat_prompt = chat_prompt.format_messages(topic = "pizza", style="romantic")


# Invoke Model

llm = ChatOllama(
    model="qwen3:0.6b"
)

ai_msg = llm.invoke(filled_chat_prompt)
print(ai_msg.content)