from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os # helps create platform-independent file paths (very useful on Windows with backslashes).
from dotenv import load_dotenv

load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), "language_understanding_paper.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()


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


# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Example: embed one chunk
vector = embeddings.embed_query(chunks[0].page_content)
print(len(vector))  # dimension of embedding

vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
print(len(vectors))  # number of chunks
print(len(vectors[0]))  # dimension of one embedding


# embed_query() → single piece of text
# embed_documents() → multiple documents (chunks)