from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from operator import itemgetter

load_dotenv()

# 1. Documents
docs = [
    Document(page_content="The moon is the only natural satellite of Earth."),
    Document(page_content="Mars is the fourth planet from the sun.")
]

# 2. Embeddings
hf_embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create FAISS Vector Store
vectorstore = FAISS.from_documents(docs, hf_embed)
retriever = vectorstore.as_retriever(k=1)  # Get top 1 result

###### now 'retriever'is ready==============================================


# 4. LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational"
) # pyright: ignore[reportCallIssue]

model = ChatHuggingFace(llm=llm)

# 5. Define the RAG Prompt Template
template = """
Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. Define the chain structure  with input mapping
# 'context' gets filled by the retriever, 'question' gets filled by user input

rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }| prompt | model | StrOutputParser()
)

# 7. Run
result = rag_chain.invoke({"question": "Tell me about Earth's satellite."})
print(result)
