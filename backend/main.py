from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from langchain.llms.base import LLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import os
import uvicorn


# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# HuggingFace model wrapper for LangChain
class HuggingFaceLLM(LLM):
    def __init__(self):
        super().__init__()
        # 🔁 Detect GPU if available, else fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Using device: {device}")

        object.__setattr__(self, "pipeline", pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device=0 if device == "cuda" else -1,
            max_new_tokens=800
        ))

    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        result = self.pipeline(prompt)
        return result[0]["generated_text"]

# Load vector store and create retriever
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = HuggingFaceLLM()

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal assistant. The context below comes from legal rights documents. Use ONLY the context to answer the question.

If the answer is not directly stated, say: "I could not find that information in the provided documents."

----------------
{context}
----------------

Question: {question}
Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)


from fastapi import Request
from fastapi.responses import JSONResponse
import asyncio
import concurrent.futures

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    try:
        # Set a timeout duration (in seconds)
        TIMEOUT = 30

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await asyncio.wait_for(
                loop.run_in_executor(pool, lambda: qa_chain({"query": req.question})),
                timeout=TIMEOUT
            )

        print("LLM result:", result.get("result"))
        print("Sources:", result.get("source_documents"))

        return {
            "answer": result["result"],
            "sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk": doc.page_content
                } for doc in result["source_documents"]
            ]
        }

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"answer": "This question took too long to answer. Please try rephrasing or simplifying it.", "sources": []}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"answer": f"Error: {str(e)}", "sources": []}
        )

from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
import os, shutil, tempfile, uuid
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    MAX_FILES = 5
    MAX_MB = 10
    MAX_BYTES = MAX_MB * 1024 * 1024        
      
    # Cleanup old sessions (older than 60 minutes)
    CLEANUP_THRESHOLD_MINUTES = 60
    now = time.time()

    temp_dir = tempfile.gettempdir()
    for folder in os.listdir(temp_dir):
        if folder.startswith("session_"):
                folder_path = os.path.join(temp_dir, folder)
                try:
                    modified_time = os.path.getmtime(folder_path)
                    age_minutes = (now - modified_time) / 60
                    if age_minutes > CLEANUP_THRESHOLD_MINUTES:
                        shutil.rmtree(folder_path)
                        print(f"🧹 Deleted old session folder: {folder_path}")
                except Exception as cleanup_err:
                    print(f"⚠️ Error during cleanup: {cleanup_err}")
    try:
        if len(files) > MAX_FILES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Too many files. Max {MAX_FILES} allowed."}
            )

        session_id = str(uuid.uuid4())
        session_path = os.path.join(tempfile.gettempdir(), f"session_{session_id}")
        os.makedirs(session_path, exist_ok=True)

        uploaded_docs = []

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{file.filename} is not a PDF."}
                )

            content = await file.read()
            if len(content) > MAX_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"{file.filename} exceeds {MAX_MB}MB limit."}
                )

            file.file.seek(0)  # rewind
            file_path = os.path.join(session_path, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            try:
                loader = PyMuPDFLoader(file_path)
                uploaded_docs.extend(loader.load())
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Failed to load {file.filename}: {str(e)}"}
                )
        if not uploaded_docs:
            return JSONResponse(status_code=400, content={"error": "No valid PDFs were uploaded."})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(uploaded_docs)

        uploaded_vectorstore = FAISS.from_documents(chunks, embeddings)

        try:
            vectorstore.merge_from(uploaded_vectorstore)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Error merging vectorstore: {str(e)}"}
            )
    
        return JSONResponse({
            "message": f"Successfully uploaded and indexed {len(files)} file(s).",
            "session_id": session_id,
            "path": session_path
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # default fallback for local dev
    uvicorn.run("main:app", host="0.0.0.0", port=port)

# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Any
# from transformers import pipeline
# from langchain.llms.base import LLM
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# # -----------------------------
# # 🌐 Initialize FastAPI App
# # -----------------------------
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # 📦 Request Schema
# # -----------------------------
# class ChatRequest(BaseModel):
#     question: str

# # -----------------------------
# # 🤖 Custom Local LLM Wrapper
# # -----------------------------
# from langchain_core.language_models.llms import LLM
# from transformers import pipeline
# from typing import List, Any


# class HuggingFaceLLM(LLM):
#     # Required attributes that LangChain core expects
#     @property
#     def _llm_type(self) -> str:
#         return "huggingface_pipeline"

#     def __init__(self):
#         # Don't call super().__init__ — it triggers Pydantic validation
#         object.__setattr__(self, "_pipeline", pipeline(
#             "text2text-generation",
#             model="google/flan-t5-large",
#             tokenizer="google/flan-t5-large",
#             max_new_tokens=512
#         ))

#         # Manually set expected attributes
#         object.__setattr__(self, "callbacks", None)
#         object.__setattr__(self, "verbose", False)
#         object.__setattr__(self, "tags", None)
#         object.__setattr__(self, "metadata", None)
#         object.__setattr__(self, "cache", None)

#     def _call(self, prompt: str, stop: List[str] = None) -> str:
#         response = self._pipeline(prompt)
#         return response[0]["generated_text"]


# # -----------------------------
# # 🔎 Load Vectorstore & Retriever
# # -----------------------------
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# vectorstore = FAISS.load_local(
#     "vectorstore",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# llm = HuggingFaceLLM()

# # -----------------------------
# # 🧾 Prompt Template
# # -----------------------------
# prompt_template = """
# You are a legal assistant. Use ONLY the context below to answer the question.

# If the answer is not directly stated, say: "I could not find that information in the provided documents."

# ----------------
# {context}
# ----------------

# Question: {question}
# Answer:
# """

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=prompt_template
# )

# # -----------------------------
# # 🔗 QA Chain with Sources
# # -----------------------------
# qa_chain = RetrievalQA.from_chain_type(
#     llm=HuggingFaceLLM(),
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt},
#     return_source_documents=True
# )

# # -----------------------------
# # 📨 Chat Endpoint
# # -----------------------------
# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     body = await request.json()
#     question = body.get("question")

#     # 1. Ask the chain
#     response = qa_chain.invoke({"query": question})

#     # 2. Extract components
#     final_answer = response.get("result") or response.get("answer") or "No answer returned."
#     source_docs = response.get("source_documents", [])

#     formatted_sources = []
#     for doc in source_docs:
#         formatted_sources.append({
#             "source": doc.metadata.get("source", "Unknown"),
#             "chunk": doc.page_content,
#             "score": doc.metadata.get("score", None),
#         })

#     return {
#         "answer": final_answer,
#         "sources": formatted_sources,
#     }