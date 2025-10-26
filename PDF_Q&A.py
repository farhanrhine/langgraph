############################## THE CLEAN AND WORKING CODE #################################################################################
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # FAISS score is L2 distance (lower = better).
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os # helps create platform-independent file paths (very useful on Windows with backslashes).
load_dotenv()


file_path = os.path.join(os.path.dirname(__file__), "language_understanding_paper.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

# Example: split text into chunks of 500 characters with 50 character overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=[" ", "\n", ""] # Try splitting by space, then newline, etc.
)

chunks = text_splitter.split_documents(documents)

# Create Embeddings and VectorStore
embed_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embed_model)

# Create the Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

###### now 'retriever'is ready==============================================

# --- 2. Build the RAG Chain ---

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    temperature=0.0,
    streaming=True,  # enable streaming
    callbacks=[StreamingStdOutCallbackHandler()]  # send live tokens to console
) # pyright: ignore[reportCallIssue]


model = ChatHuggingFace(llm=llm)

# 5. Define the RAG Prompt Template
template = """
You are a Policy Expert. Answer the question ONLY based on the CONTEXT provided below.
If the context does not contain the answer, state that you cannot find the information.
 --- CONTEXT ---  {context}  ---
    Question: {question}
"""

# RAG Prompt
prompt = ChatPromptTemplate.from_template(template)


rag_chain = (
    {
        "context": itemgetter("question") | retriever, # type: ignore
        "question": itemgetter("question")
    }| prompt | model | StrOutputParser()
)

# --- 3. Run Q&A ---
print("--- RAG Bot Initialized ---")

# Question 1 (Answerable by context)
query1 = "Find information about the author Alec Radford from OpenAI."
a1 = rag_chain.invoke({"question": query1})
print(f"\nQ1: {query1}\nA: {a1}")

# Question 2 (Not answerable by context)
query2 = "What is the new salary structure for engineers?"
a2 = rag_chain.invoke({"question": query2})

print(f"\nQ2: {query2}\nA: {a2}")

# Find information about the author Alec Radford from OpenAI
# What is the new salary structure for engineers?