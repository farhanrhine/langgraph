from langchain_community.document_loaders import CSVLoader
import os 

file_path = os.path.join(os.path.dirname(__file__), "iris.csv")


loader = CSVLoader(file_path)
documents = loader.load()

print(f"Loaded {len(documents)} rows") # 💡 Each row becomes one Document, containing both column names and values as text.
print(documents[0].page_content)
