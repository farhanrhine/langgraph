# "📄 PDF Chatbot with session its works fine. 
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("RAG PDF Chatbot")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# --- Initialize session state ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    # Only process PDF if not already done
    if st.session_state.vectorstore is None:
        with st.spinner("Processing your PDF..."):
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=[" ", "\n", ""]
            )
            chunks = text_splitter.split_documents(documents)

            embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embed_model)
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            st.success("✅ PDF processed and embedded successfully!")

    # --- Build RAG Chain once ---
    if st.session_state.rag_chain is None:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="conversational",
            temperature=0.0,
            streaming=True
        ) # type: ignore

        model = ChatHuggingFace(llm=llm)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        template = """
        You are a helpful assistant. Answer the question ONLY based on the CONTEXT below.
        If the answer is not in the context, say you cannot find it.
        --- CONTEXT --- {context}
        --- QUESTION --- {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Ensure retriever is available (type narrowing for linters)
        retriever = st.session_state.retriever
        assert retriever is not None, "Retriever not initialized. Upload a PDF first."

        st.session_state.rag_chain = (
            {
                # Ensure the retriever receives a string query (the 'question'),
                # then format retrieved documents into a single context string.
                "context": RunnableLambda(itemgetter("question")) | retriever | RunnableLambda(format_docs),
                "question": itemgetter("question")
            }
            | prompt
            | model
            | StrOutputParser()
        )

    # --- Chat Interface ---
    question = st.text_input("Ask a question about your PDF:")

    if st.button("Get Answer") and question.strip() != "":
        response_container = st.empty()
        response_text = ""
        for token in st.session_state.rag_chain.stream({"question": question}):
            response_text += token
            response_container.markdown(f"Answer: {response_text}")

        # Save chat history
        st.session_state.chat_history.append({"question": question, "answer": response_text})

    
