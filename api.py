import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Tambahkan path agar bisa mengimpor modul RAG
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RAG.main import HybridRAG

# Load environment
load_dotenv()

app = FastAPI(
    title="RAG in a Box API",
    description="REST API untuk integrasi sistem Hybrid RAG ke aplikasi chat eksternal.",
    version="1.0.0"
)

# Inisialisasi Engine RAG (Singleton-like)
rag_engine = None

def get_rag():
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = HybridRAG()
        except Exception as e:
            print(f"Error initializing RAG Engine: {e}")
    return rag_engine

class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    status: str = "success"

@app.on_event("startup")
async def startup_event():
    get_rag()
    print("🚀 RAG API Engine Started!")

@app.get("/")
async def root():
    return {"message": "RAG in a Box API is running", "docs": "/docs"}

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    engine = get_rag()
    if engine is None:
        raise HTTPException(status_code=500, detail="RAG Engine not initialized. Check server logs.")
    
    try:
        answer = engine.query(request.message)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
