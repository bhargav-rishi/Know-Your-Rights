import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_and_add_to_vectorstore(docs_folder="../docs", vectorstore_dir="vectorstore"):
    all_docs = []

    for file in os.listdir(docs_folder):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(docs_folder, file))
            all_docs.extend(loader.load())

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,# Increase chunk size
        chunk_overlap=150,# Add more overlap between chunks
        separators=["\n\n", "\n", ".", " "] # Ensure paragraph-aware breaks
    )
    chunks = splitter.split_documents(all_docs)

    # Use BGE-small Hugging Face embeddings (no API required)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
    # Store the vectors locally using FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vectorstore_dir)

    print("Vector store created and saved.")
