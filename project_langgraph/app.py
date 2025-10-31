
import streamlit as st
from langchain_core.messages import HumanMessage
from project_langgraph.multi_tool_reasoning_agent import app  # change filename here
import time

st.title("🤖 Intelligent Agent (LangGraph + ToolAgent + gpt-oss:120b-cloud)")
st.write("Ask anything! The agent will use Math, Weather, Wikipedia, and Search tools based on your query.")


user_input = st.text_area(
    "Enter your question:",
    height=130,
    placeholder='Type your question here...'
)


def stream_output(text: str):
    """Stream markdown in clean, readable blocks."""
    paragraphs = text.split("\n\n")  # split by blank lines
    for block in paragraphs:
        yield block + "\n\n"  # restore spacing
        time.sleep(0.05)  # adjust speed if you want

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please type something.")
    else:
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": [HumanMessage(content=user_input)]})
            final_msg = result["messages"][-1].content

        st.markdown("### Answer:")
        st.write_stream(stream_output(final_msg))
