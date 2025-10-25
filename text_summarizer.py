from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

template = """
            You are an expert summarization bot.
            Your task is to summariza the following TEXT in exactly one clear, concise sentence.
            TEXT: "{text_input}" 
            """

prompt = ChatPromptTemplate.from_template(template)


input_text = """
The Amazon rainforest is the largest tropical rainforest in the world.
It spans eight countries and one territory, with the majority of it
in Brazil. The ecosystem is home to approximately 390 billion trees,
divided into 16,000 species. Its immense biodiversity is critical to
regulating the world's climate by absorbing vast amounts of carbon dioxide.
Deforestation, primarily driven by cattle ranching and soy production,
remains a significant threat to its survival.
"""



# llm = ChatOllama(
#     model = "qwen3:0.6b",
#     reasoning= True
# )

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
) # type: ignore


model = ChatHuggingFace(llm=llm)


parser = StrOutputParser()

chain = prompt | model  | parser


summary = chain.invoke({"text_input" : input_text})

print(summary)

