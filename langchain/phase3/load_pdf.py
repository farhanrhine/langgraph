from langchain_community.document_loaders import PyPDFLoader
import os # helps create platform-independent file paths (very useful on Windows with backslashes).


file_path = os.path.join(os.path.dirname(__file__), "language_understanding_paper.pdf")

loader = PyPDFLoader(file_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages") # Unlike text files, each page of the PDF becomes its own Document object.That’s why the output says: Loaded 12 pages
print(documents[0].page_content[:1000])
print("--------------------------------------------------\n")
print(documents[0].metadata)
