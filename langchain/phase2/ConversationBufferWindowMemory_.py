# Example concept of 'window' memory
from langchain.memory import ConversationBufferWindowMemory

# Only remember the last 3 turns
window_memory = ConversationBufferWindowMemory(k=3)

# After 5 turns, the oldest 2 will be automatically truncated/forgotten.
# The actual tokens sent to the LLM will only include the last 3 turns.

# print(window_memory.load_memory_variables({})['history'])