import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

st.title("Text Summarizer")
st.write("Enter a paragraph below, and the AI will summarize it in one clear, concise sentence.")

# User input
user_input = st.text_area("Input Text", height=200)

if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter some text to summarize!")
    else:
        # Prompt template
        template = """
        You are an expert summarization bot.
        Your task is to summarize the following TEXT in exactly one clear, concise sentence.
        TEXT: "{text_input}" 
        """
        prompt = ChatPromptTemplate.from_template(template)

        # HuggingFace endpoint model
        llm = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task="text-generation",
        ) # pyright: ignore[reportCallIssue]

        model = ChatHuggingFace(llm=llm)
        parser = StrOutputParser()
        chain = prompt | model | parser

        # Run summarization
        summary = chain.invoke({"text_input": user_input})

        st.subheader("Summary")
        st.write(summary)
