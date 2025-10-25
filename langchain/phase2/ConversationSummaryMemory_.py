from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 1. Initialize Summary Memory
# It uses the LLM itself to generate the summary
memory = ConversationSummaryMemory(llm=llm)

# 2. Add some dummy history to generate a summary
memory.save_context({"input": "Hello! My name is Farhan, and I live in Chandigarh."},
                    {"output": "Nice to meet you, Farhan! Paris is a beautiful city."})
memory.save_context({"input": "I have one dog and two cats at home."},
                    {"output": "A small zoo! What are your pets' names?"})

# 3. Load the summary
summary = memory.load_memory_variables({})['history']
print("--- Current Summary History ---\n", summary)