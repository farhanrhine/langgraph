from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# model
llm = ChatOllama(
    model ="qwen3:0.6b"
)

#template with placeholders {topic} and {style}
template = """ please write a short paragraph of given {topic}.
               The respond must be in a {style} tone."""

# Create the PromptTemplate object
prompt = PromptTemplate.from_template(template)

# Use the format method to fill the variables
filled_prompt = prompt.format(topic = "pizza", style = "romantic")

# invoke model
ai_msg = llm.invoke(filled_prompt)

# print result
print(ai_msg.content)
