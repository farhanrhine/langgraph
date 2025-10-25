from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="granite-embedding:latest")  

# vector = embeddings.embed_query(chunks[0].page_content)
vector_ollama = embeddings.embed_query("My name is Farhan")

print(f"HuggingFace Vector Dimension: {len(vector_ollama)}")
print("================================================")
print(f"first five vector : {vector_ollama[:5]}")
print("================================================")
print(f"whole vector : {vector_ollama}")

