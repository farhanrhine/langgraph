import streamlit as st
from langchain_core.messages import HumanMessage
from project_langgraph.multi_tool_reasoning_agent import app  # change filename here
import time


st.title("🤖 Intelligent Agent (LangGraph + ToolAgent + gpt-oss:120b-cloud)")
st.write("Ask anything! The agent will use Math, Weather, Wikipedia, and Search tools based on your query.")

user_input = st.text_input("Enter your question:")

def stream_output(text: str):
    """Yield text gradually to stream in UI."""
    for chunk in text.split():
        yield chunk + " "
        time.sleep(0.05)  # controls speed (optional)

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please type something.")
    else:
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": [HumanMessage(content=user_input)]}) # type: ignore
            final_msg = result["messages"][-1].content

        st.markdown("### Answer:")

        # ⭐ STREAM RESPONSE HERE ⭐
        st.write_stream(stream_output(final_msg))

