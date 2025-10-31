from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="minimax-m2:cloud",
    base_url=os.getenv("OLLAMA_HOST"),
    api_key=os.getenv("OLLAMA_API_KEY") # type: ignore
)

print(llm.invoke("Hello, who are you?").content)
