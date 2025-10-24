from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(
    model = "qwen3:0.6b"
)

parser = StrOutputParser()

#template with placeholders {topic} and {style}
template = """ please write a short paragraph of given {topic}."""

prompt = ChatPromptTemplate.from_template(template)


# LCEL Chain: (Prompt) -> (Model) -> (Output Parser)
lcel_chain = prompt | llm | parser

result = lcel_chain.invoke({"topic" :"farhan"})

print(result)