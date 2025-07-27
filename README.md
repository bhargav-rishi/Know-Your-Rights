# Know Your Rights – Legal Chatbot

![Built With](https://img.shields.io/badge/Built%20With-FastAPI%20%7C%20LangChain%20%7C%20React%20%7C%20FAISS%20%7C%20Hugging%20Face-blue)  
![Language](https://img.shields.io/badge/Language-Python%20%7C%20JavaScript-orange)  
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

---

Legal jargon can be overwhelming. Our chatbot simplifies it.

**Know Your Rights** is a Retrieval-Augmented Generation (RAG)-based legal assistant that answers your legal questions using government PDFs, ACLU documents, or any uploaded file.


---

## Screenshots

### Homepage (Initial View)
![Homepage](screenshots/01-home.png)
![Homepage](screenshots/02-home.png)

### Legal Question Answering
![Chat in Action](screenshots/03-chat-upload.png)
![Chat in Action](screenshots/04-chat-upload.png)

---

## Features

- Natural Language Legal Q&A  
- Multi-file PDF Uploads for Context  
- FAISS Vectorstore + Hugging Face Embeddings  
- Serverless Deploy (Render + Hugging Face)  
- Light/Dark Mode  
- Upload validation + session caching

---

## Tech Stack

**Backend**:  
- FastAPI  
- LangChain  
- Hugging Face Transformers  
- FAISS (in-memory vectorstore)  
- Sentence Transformers

**Frontend**:  
- React + Vite  
- TailwindCSS  
- Axios  

**Deploy**:  
- Backend → Hugging Face Spaces  
- Frontend → Render

---

## Run Locally

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```


## Use Cases

- Legal rights awareness for students, immigrants, or tenants  
- Document comprehension for law clinics or journalists  
- Prototype for scalable legal tech solutions  
