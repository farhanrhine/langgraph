from langchain_community.document_loaders import WebBaseLoader
import os
os.environ["USER_AGENT"] = "my-langchain-bot/0.1"

loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
documents = loader.load()

print(documents[0].page_content[:1000])
print("--------------------------------------------------\n")
print(documents[0].metadata)

# in RAG, BeautifulSoup helps convert a webpage → readable text Document for your LLM to use.