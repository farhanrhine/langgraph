from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:0.6b"
)

messages = [
    ("system", "You are a helpful assistant that translates English to arabic. Translate the user sentence."),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)