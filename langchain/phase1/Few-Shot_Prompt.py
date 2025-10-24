from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Load model
llm = ChatOllama(model="qwen3:0.6b")

# 1️ System prompt: sets assistant behavior
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative AI poet who writes fun, romantic, or classical poems.")
])

# 2️ Few-shot examples
example_prompt = ChatPromptTemplate.from_messages([
    ("user", "Write a funny poem about cats."),
    ("assistant", "Cats jump and play all day, chasing mice in a funny way!"),
    ("user", "Write a romantic poem about sunsets."),
    ("assistant", "The sun dips low, painting skies of gold, your hand in mine, a story told.")
])

# 3️ Actual user input (dynamic)
user_prompt = ChatPromptTemplate.from_messages([
    ("user", "Write a {style} poem about {topic} suitable for {audience}.")
])

# 4️ Combine all messages
combined_messages = (
    system_prompt.format_messages() +
    example_prompt.format_messages() +
    user_prompt.format_messages(
        style="romantic",
        topic="pizza",
        audience="teenagers"
    )
)

# 5️ Invoke model
ai_msg = llm.invoke(combined_messages)
print(ai_msg.content)
