from langchain_ollama import ChatOllama
# llm = ChatOllama(
#     model="minimax-m2:cloud"
# )

# llm = ChatOllama(
#     model="gpt-oss:120b-cloud"
# )
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:0.6b")

print(llm.invoke("Hello, who are you?").content)
