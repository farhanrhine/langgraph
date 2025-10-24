from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(
    model = "qwen3:0.6b"
)

template = """what is the capital of {country}?"""
prompt = PromptTemplate.from_template(template)


chain = prompt | model

# chain.invoke expects a dict of inputs
result = chain.invoke(input={"country":"india"})

print(result.content)


