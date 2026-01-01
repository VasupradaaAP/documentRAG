"""AI Document Chatbot - FastAPI Backend"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List, Dict
import shutil
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from datetime import datetime

from vector import create_document_store


class AskRequest(BaseModel):
    question: str
    debug: bool = False


class ChatRequest(BaseModel):
    message: str
    debug: bool = False


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunk_count: Optional[int] = None


# ========================================
# INITIALIZE FASTAPI APP
# ========================================

app = FastAPI(title="AI Document Chatbot")
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "unanswered_questions.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("📚 Initializing document store...")
doc_store = create_document_store()
print("✓ Document store ready")

LLM_MODEL_NAME = "google/flan-t5-base"

print(f"🤖 Loading LLM model: {LLM_MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

print(f"✓ LLM loaded on {device.upper()}")


def generate_llm_response(query: str, context: str = "") -> str:
    try:
        if context:
            prompt = f"""Using only the information provided below, answer the question.
If the information is not in the context, respond with: "This information is not available in the provided document(s)."

Context:
{context[:2000]}

Question: {query}

Answer:"""
        else:
            return "This information is not available in the provided documents."

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = llm_model.generate(
            **inputs,
            max_length=200,
            num_beams=4, 
            early_stopping=True,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()
    
    except Exception as e:
        print(f"✗ LLM error: {e}")
        return "Error generating response. Please try again."


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    html_file = BASE_DIR / "templates/page.html"
    if html_file.exists():
        return FileResponse(html_file)
    return JSONResponse({"message": "AI Document Chatbot API", "docs": "/docs"})


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(status_code=204, content={})


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file."""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        pdf_path = UPLOAD_DIR / file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n📥 Received file: {file.filename}")
        success = doc_store.pdf_to_vectors(str(pdf_path))
        
        if success:
            chunk_count = len(doc_store.chunks)
            return {
                "success": True,
                "message": f"Successfully processed {file.filename}",
                "chunk_count": chunk_count
            }
        else:
            return {
                "success": False,
                "message": "Failed to process PDF",
                "chunk_count": 0
            }
    
    except Exception as e:
        print(f"✗ Error in /ingest: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Answer questions using document retrieval (RAG)."""
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        retrieval_result = doc_store.retrieve(question, top_k=3)
        
        if not retrieval_result['success']:
            return {
                "answer": "This information is not available in the provided document(s).",
                "citations": "",
                "debug": None
            }
        
        chunks = retrieval_result['chunks']
        
        if not chunks:
            return {
                "answer": "This information is not available in the provided document(s).",
                "citations": "",
                "debug": None
            }
        
        context = "\n\n".join([chunk['text'] for chunk in chunks])
        answer = generate_llm_response(question, context)
        citations = doc_store.format_citations(chunks)
        
        response = {
            "answer": answer,
            "citations": citations
        }
        
        if request.debug:
            debug_chunks = []
            for i, chunk in enumerate(chunks, 1):
                debug_chunks.append({
                    "rank": i,
                    "document": chunk['document'],
                    "page": chunk['page'],
                    "snippet": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                })
            response["debug"] = debug_chunks
        else:
            response["debug"] = None
        
        return response
    
    except Exception as e:
        print(f"✗ Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/health")
async def health():
    """System health check."""
    return {
        "status": "ok",
        "llm_model": LLM_MODEL_NAME,
        "device": device,
        "documents_indexed": doc_store.is_loaded,
        "chunk_count": len(doc_store.chunks) if doc_store.is_loaded else 0
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    """Ensure Compatibility."""
    ask_request = AskRequest(question=body.message, debug=True)
    result = await ask_question(ask_request)
    
    if "not available in the provided document" in result["answer"]:
        logger.info(f"UNANSWERED QUESTION: {body.message}")
    
    return JSONResponse(content={
        "answer": result["answer"],
        "citations": result.get("citations", ""),
        "chunks": result.get("debug", [])
    })


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("🚀 Starting AI Document Chatbot Server")
    print("="*50)
    print(f"📍 URL: http://0.0.0.0:8000")
    print(f"📖 Docs: http://0.0.0.0:8000/docs")
    print(f"💡 Upload PDFs via POST /ingest")
    print(f"💬 Ask questions via POST /ask")
    print("="*50 + "\n")
    
    uvicorn.run("tech:app", host="0.0.0.0", port=8000, reload=True)