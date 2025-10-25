from langchain_community.document_loaders import TextLoader
import os


file_path = os.path.join(os.path.dirname(__file__), "the-verdict.txt")

# Load text file
loader = TextLoader(file_path)
documents = loader.load()

print(len(documents)) # how many Document objects were created by LangChain from your file. In your case, it’s 1 because the loader treated the entire text file as a single document.
print(documents[0].page_content[:500])  # show first 500 chars


# documents[0] → the first (and only) Document object.