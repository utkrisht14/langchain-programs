import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print(os.getcwd())

if __name__ == "__main__":
    print("Ingesting....")

    loader = TextLoader(
        r"C:\Users\utkri\PycharmProjects\LangChainCourse\rag-programs\mediumblog1.txt",
        encoding="utf-8",
        autodetect_encoding=True
    )

    documents = loader.load()

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"created {len(texts)} chunks.")