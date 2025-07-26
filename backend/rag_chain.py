from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Load Hugging Face model
pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Build RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def get_rag_answer(question):
    result = qa_chain({"query": question})
    sources = list({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})
    return {
        "answer": result["result"],
        "sources": sources
    }
