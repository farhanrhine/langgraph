from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="qwen3:0.6b")

# Step 1: Get capital
template1 = "What is the capital of {country}?"
prompt1 = PromptTemplate.from_template(template1)
chain1 = prompt1 | model

# Step 2: Get fact about capital
template2 = "Tell me an interesting fact about {capital}."
prompt2 = PromptTemplate.from_template(template2)
chain2 = prompt2 | model

# Manual chaining
country = "India"

# Step 1
capital_result = chain1.invoke({"country": country})
capital = capital_result.content.strip()  # pyright: ignore[reportAttributeAccessIssue] # strip() returns a new string and does not modify the original string

# Step 2
fact_result = chain2.invoke({"capital": capital})

print("Capital:", capital)
print("Fact:", fact_result.content)
