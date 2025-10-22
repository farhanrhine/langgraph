from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="qwen3:0.6b"
)

messages = [
    ("system", "You are a helpful assistant that translates English to arabic. Translate the user sentence."),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)