from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os # helps create platform-independent file paths (very useful on Windows with backslashes).


file_path = os.path.join(os.path.dirname(__file__), "language_understanding_paper.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

# 1. Initialize the splitter
# Chunk size: max size of each chunk
# Chunk overlap: number of characters to share between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[" ", "\n", ""] # Try splitting by space, then newline, etc.
)

# Example: split text into chunks of 500 characters with 50 character overlap

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")
print(f"Loaded {len(documents)} pages")
print(chunks[94].page_content[:500])
