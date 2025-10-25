
from langchain_huggingface import HuggingFaceEmbeddings
# 2. Local/Offline Embedding (e.g., HuggingFace via sentence-transformers)
# This downloads a model (fast) and runs locally (free/private)

hf_embed = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_hf = hf_embed.embed_query("What is LangChain?")

print(f"HuggingFace Vector Dimension: {len(vector_hf)}")
# Output: 384 (all-MiniLM-L6-v2 produces 384 dimensions)
print("================================================")
print(f"first five vector : {vector_hf[:5]}")
print("================================================")
# print(f"whole vector : {vector_hf}")