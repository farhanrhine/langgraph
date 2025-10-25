from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:0.6b", temperature=0.7)

# 1. Initialize Memory
memory = ConversationBufferMemory()

# 2. Create a basic conversational chain
conversation = ConversationChain(
    llm=llm,
    memory=memory, # Attach the memory here
    verbose=False # Set to True to see the prompt w/ memory
)

# Turn 1
conversation.invoke("My favorite color is blue.")

# Turn 2
response = conversation.invoke("What color did I say I liked?")
print(response['response'])

# View the stored history
# print("\nHistory:\n", memory.load_memory_variables({}))
