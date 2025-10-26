import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 PDF Chatbot")

# --- Step 1: Upload PDF ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
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
        vectorstore = FAISS.from_documents(chunks, embed_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        st.success("✅ PDF processed and embedded successfully!")

        # --- Step 2: Build RAG Chain ---
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            task="conversational",
            temperature=0.0
        ) # pyright: ignore[reportCallIssue]

        model = ChatHuggingFace(llm=llm)

        template = """
        You are a helpful assistant. Answer the question ONLY based on the CONTEXT below.
        If the answer is not in the context, say you cannot find it.
        --- CONTEXT --- {context}
        --- QUESTION --- {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | prompt
            | model
            | StrOutputParser()
        )

        # --- Step 3: Ask question ---
        question = st.text_input("Ask a question about your PDF:")

        if st.button("Get Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                st.markdown("### 🧠 Answer:")
                answer = rag_chain.invoke({"question": question})
                st.write(answer)
