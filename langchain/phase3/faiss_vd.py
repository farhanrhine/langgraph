from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Define the documents and embedding model
docs = [
    Document(page_content="The moon is the only natural satellite of Earth."),
    Document(page_content="Mars is the fourth planet from the sun.")
]

hf_embed = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# 2. Create the Vector Store from the documents and embeddings
# This step computes and stores the vectors
vectorstore = FAISS.from_documents(docs, hf_embed)

# 3. Create a Retriever (the searchable interface)
retriever = vectorstore.as_retriever(k=1) # Get top 1 result

# 4. Use the retriever to find relevant chunks
relevant_docs = retriever.invoke("Tell me about Earth's satellite.")
print(f"Found: {relevant_docs[0].page_content}")
