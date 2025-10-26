import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# -----------------------------
# 1. Load Environment + PDF
# -----------------------------
load_dotenv()

file_path = os.path.join(os.path.dirname(__file__), "language_understanding_paper.pdf")
loader = PyPDFLoader(file_path)
documents = loader.load()

# -----------------------------
# 2. Split into Chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[" ", "\n", ""]
)
chunks = text_splitter.split_documents(documents)

# -----------------------------
# 3. Create Embeddings + Vectorstore
# -----------------------------
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embed_model)

# -----------------------------
# 4. Define Retriever Function (with Debug)
# -----------------------------
def retrieve_with_debug(question: str):
    """Retrieve chunks and display debug info before answering."""
    results = vectorstore.similarity_search_with_score(question, k=3)
    print("\n--- Retrieved Chunks ---")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\nChunk #{i} | Score: {score:.4f}\n{doc.page_content[:400]}...\n")

    # Prepare data for model
    context_text = "\n\n".join([doc.page_content for doc, _ in results])
    scores_text = "\n".join([f"Chunk {i+1}: {score:.4f}" for i, (_, score) in enumerate(results)])
    return {"context": context_text, "scores": scores_text, "question": question}

# -----------------------------
# 5. Initialize LLM
# -----------------------------

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    temperature=0.0# Low temp for factual answers
) # pyright: ignore[reportCallIssue]
model = ChatHuggingFace(llm=llm)

# -----------------------------
# 6. Define Prompt
# -----------------------------
template = """
You are a Policy Expert. Answer the question ONLY based on the CONTEXT below.
If the context does not contain the answer, say you cannot find it.

--- CONTEXT ---  
{context}  
--- SIMILARITY SCORES ---  
{scores}  

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# -----------------------------
# 7. Combine with RunnableParallel
# -----------------------------
rag_chain = RunnableParallel({
    "retrieved_context": retrieve_with_debug,
    "final_answer": retrieve_with_debug | prompt | model | StrOutputParser()
})

print("\n--- RAG Bot Initialized ---")

# -----------------------------
# 8. Run Q&A
# -----------------------------
q1 = "Find information about the author Alec Radford from OpenAI."
q2 = "What is the new salary structure for engineers?"

for q in [q1, q2]:
    print(f"\n\n--- Question: {q} ---")
    result = rag_chain.invoke(q)

    print("\n✅ Retrieved Context:\n", result["retrieved_context"]["context"][:600], "...")
    print("\n💬 Final Answer:\n", result["final_answer"])

# -----------------------------
# 9. (Optional) Real-time Streaming Example
# -----------------------------
# If you want live typing effect, uncomment this part:

print("\n--- Streaming Mode Example ---")
question = "Explain the role of pre-training in this paper."
retrieved = retrieve_with_debug(question)
formatted_prompt = prompt.format(**retrieved)
print("\nStreaming Answer:\n")
for token in model.stream(formatted_prompt):
    print(token, end="", flush=True)

