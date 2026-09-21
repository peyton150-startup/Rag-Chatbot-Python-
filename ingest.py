from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os

from nvidia_embeddings import NVIDIAOpenAIEmbeddings

documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file
        documents.extend(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

chunks = splitter.split_documents(documents)

embeddings = NVIDIAOpenAIEmbeddings()
persist_directory = os.getenv("CHROMA_DB_DIR", "db_nvidia")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

db.persist()

print(
    f"Indexed {len(chunks)} chunks from "
    f"{len(set(d.metadata['source'] for d in documents))} documents "
    f"into {persist_directory}."
)
