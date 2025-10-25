from langchain_text_splitters import RecursiveCharacterTextSplitter


text = "LangChain is a framework. It helps connect models to data. It is powerful."

# 1. Initialize the splitter
# Chunk size: max size of each chunk
# Chunk overlap: number of characters to share between chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5,
    separators=[" ", "\n", ""] # Try splitting by space, then newline, etc.
)


# 2. Split the text
# chunks = text_splitter.split_documents(documents)
chunks = splitter.split_text(text)
print(f"Original text split into {len(chunks)} chunks:")
for chunk in chunks:
    print(f"- {chunk}")

# print(chunks[0].page_content[:500])
