from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Define documents
docs = [
    Document(page_content="The moon is the only natural satellite of Earth."),
    Document(page_content="Mars is the fourth planet from the sun.")
]

# 2. Embeddings
hf_embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create FAISS Vector Store
vectorstore = FAISS.from_documents(docs, hf_embed)

# 4. Run similarity search WITH score
query = "Tell me about Earth's satellite."
results = vectorstore.similarity_search_with_score(query, k=2)

# 5. Display
for doc, score in results:
    print(f"Score: {score:.4f} → Content: {doc.page_content}")


"""
Lower score = more similar (because FAISS uses distance metric).

If you want higher = better, you can invert or normalize, but by default:

smaller score → closer vector → more relevant document.

"""