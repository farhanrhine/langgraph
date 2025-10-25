from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="qwen3:0.6b")
parser = StrOutputParser()




# Step 1: Translate to Pirate-speak
p1 = PromptTemplate.from_template("Translate the following text into french : {text}")
chain_translate = {"text": RunnablePassthrough()} | p1 | llm | parser

# Step 2: Summarize the Pirate-speak
p2 = PromptTemplate.from_template("Summarize the pirate text in one word: {pirate_text}")
# The key: using .map() or simple functions to pass the output of c1 as input to c2
chain_summary = {"pirate_text": chain_translate} | p2 | llm | parser

# Full Chain: Input text -> Pirate -> Summary
result = chain_summary.invoke("I am a AI/ML engineer.")
print(result) # Should be a single word, like "Code-Ahoy!" or similar


"""

Input: "I am a AI/ML engineer."
       |
       v
Step 1: Translate -> French text
       |
       v
Step 2: Summarize -> One word
       |
       v
Output: e.g., "Code-Ahoy!"


"""