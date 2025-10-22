
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
) # type: ignore
# repo_id="HuggingFaceH4/zephyr-7b-beta",

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india?")
print(result.content)

