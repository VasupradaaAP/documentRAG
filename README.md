# 🤖 AI Document Chatbot - RAG System

A **production-ready** document-based chatbot powered by RAG (Retrieval Augmented Generation). Answer questions from your PDF files.

## 📋 Project Overview

This is a complete end-to-end RAG (Retrieval Augmented Generation) system that:
1. **Ingests PDF documents** and extracts text page-by-page
2. **Chunks text** with overlap (200 tokens, 50-token overlap) to maintain context
3. **Generates embeddings** using Sentence Transfor
4. **Stores in FAISS** vector database for fast similarity search
5. **Retrieves relevant chunks** for user queries (top-3 by default)
6. **Generates answers** using Hugging Face LLM with citations
7. **Evaluates system** with automated metrics (retrieval, faithfulness, hallucination)

## ▶️ Demo Video
https://drive.google.com/file/d/1pFw0yoALs9GKbQyMfLXtDwCeKzG0auvW/view?usp=sharing

## ✨ Features

### Core Functionality
- **📄 PDF Ingestion**: Upload via API endpoint with validation
- **🔍 Smart Retrieval**: FAISS vector search with L2 distance
- **🧠 Local LLM**: Hugging Face FLAN-T5-Large (780MB, runs on CPU)
- **🎯 Citations**: Source document and page numbers for every answer
- **📊 Chunk Display**: View top 3 retrieved chunks for transparency
- **🚫 Fallback Handling**: Clean "not available" messages when info missing
- **📝 Logging**: Tracks unanswered questions for improvement

### Advanced Features
- **📈 Evaluation System**: 50 test questions across 3 categories
- **🎯 Metrics**: Retrieval hit rate, faithfulness, hallucination detection
- **🔧 Configurable**: Chunk size, overlap, models, top-k retrieval
- **🎨 Clean UI**: Split-screen interface with modern design
- **⚡ Fast**: FAISS enables millisecond similarity search

## 🏗️ Architecture

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       ▼
┌─────────────────────┐
│  Text Extraction    │  PyPDF2, page-by-page
│  (Page Metadata)    │
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│  Text Chunking      │  200 tokens, 50 overlap
│  (Sliding Window)   │  Preserves context at boundaries
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│  Generate           │  Sentence Transformers
│  Embeddings         │  384-dim vectors
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│  FAISS Index        │  IndexFlatL2
│  (Vector Store)     │  L2 distance search
└──────┬──────────────┘
   [QUERY]
       ▼
┌─────────────────────┐
│  Similarity Search  │  Top-K retrieval (k=3)
│  (Retrieve Chunks)  │  Most relevant context
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│  LLM Generation     │  FLAN-T5-Large
│  (Answer + Context) │  Grounded in retrieved chunks
└──────┬──────────────┘
       ▼
┌─────────────────────┐
│  Response           │
│  • Answer           │
│  • Citation         │
│  • Top 3 Chunks     │
└─────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose | Size |
|-----------|-----------|---------|------|
| **Embeddings** | `all-MiniLM-L6-v2` | Convert text to 384-dim vectors | ~90MB |
| **Vector DB** | FAISS IndexFlatL2 | Fast similarity search | In-memory |
| **LLM** | `google/flan-t5-base` | Answer generation | ~250MB |
| **Backend** | FastAPI + Uvicorn | REST API server | - |
| **PDF Processing** | PyPDF2 | Text extraction | - |
| **Frontend** | HTML/CSS/JS | User interface | - |

### Data Flow

1. **Ingestion**: PDF → Pages → Chunks → Embeddings → FAISS
2. **Query**: Question → Embedding → Search → Top-K Chunks
3. **Generation**: Chunks + Question → LLM → Answer + Citations
4. **Display**: Answer + Citation + Chunks → User

## 📦 Installation

### Prerequisites

- **Python 3.8+** (tested on 3.10, 3.11)
- **4GB+ RAM** (for models and embeddings)
- **1GB disk space** (for models)
- **Windows/Linux/macOS**

### Project Setup

```bash
# 1. Navigate to project directory
cd "d:\AI doc-chatbot"

# 2. Create virtual environment (recommended)
python -m venv venv

# 3. Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. First run downloads models (~1GB total, one-time)
python tech.py
```

## 🚀 Usage

### Start the Server

```bash
python tech.py
```

Server will start at: `http://127.0.0.1:8000`

### API Endpoints

#### 1. **POST /ingest** - Upload PDF

```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed your_document.pdf",
  "chunk_count": 42
}
```

#### 2. **POST /ask** - Ask Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the safety procedures?",
    "debug": false
  }'
```

**Response:**
```json
{
  "answer": "The safety procedures include...",
  "citations": "flight_manual (page 3) | flight_manual (page 7)",
  "debug": null
}
```

**With Debug Mode:**
```json
{
  "answer": "The safety procedures include...",
  "citations": "flight_manual (page 3)",
  "debug": [
    {
      "rank": 1,
      "document": "flight_manual",
      "page": 3,
      "snippet": "Safety procedures require all passengers to..."
    }
  ]
}
```

#### 3. **GET /health** - Health Check

```bash
curl "http://127.0.0.1:8000/health"
```

**Response:**
```json
{
  "status": "ok",
  "llm_model": "google/flan-t5-base",
  "device": "cpu",
  "documents_indexed": true,
  "chunk_count": 42
}
```

### Web Interface

1. Open browser: `http://127.0.0.1:8000`
2. Upload PDF using `/ingest` endpoint (via code or API testing tool)
3. Ask questions in the chat interface
4. Receive answers !


## 🔧 Configuration

### Chunking Strategy

**File**: `vector.py`

```python
CHUNK_SIZE = 200
OVERLAP = 50
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
```

### LLM Model Selection

**File**: `tech.py`

```python
LLM_MODEL_NAME = "google/flan-t5-base"
```

**Options:**
- `google/flan-t5-small` (~80MB) - Fast, less accurate
- `google/flan-t5-base` (~250MB) - **Recommended** balance
- `google/flan-t5-large` (~780MB) - Better quality, slower

### Retrieval Settings

**File**: `tech.py` (in `/ask` endpoint)

```python
retrieval_result = doc_store.retrieve(question, top_k=3)
```

Change `top_k` to retrieve more/fewer chunks (default: 3)

## 🐛 Debug Mode

Enable debug mode to see which chunks were retrieved:

```json
{
  "question": "What is the flight speed?",
  "debug": true
}
```

Returns top 3 chunks with:
- Rank (1-3)
- Document name
- Page number
- Text snippet (first 200 chars)


## 📊 Evaluation System

The project includes a comprehensive evaluation framework to measure RAG performance.

### Run Evaluation

```bash
python evaluate.py
```

### What It Tests

**50 Questions Across 3 Categories:**
1. **Simple Factual** (20) - Direct lookups, definitions
2. **Applied** (20) - Scenario-based, operational procedures
3. **Higher-Order** (10) - Multi-step reasoning, trade-offs

### Output Files

- `evaluation_report.txt` - Human-readable report
- `evaluation_report.json` - Detailed data for analysis
- `evaluate_report.md` for full guide


## 🎯 Use Cases

### Aviation Documentation
- Flight manuals, procedures, regulations
- Maintenance documentation
- Safety guidelines

### Technical Documentation  
- Software documentation
- API references
- User manuals

### Legal & Compliance
- Contracts, policies
- Compliance documents
- Regulatory guidelines

### Research & Education
- Research papers
- Textbooks
- Course materials



## 🚀 Deployment

### Local Development
```bash
python tech.py  # Runs on http://127.0.0.1:8000
```

## 🎯 Example Workflow

```bash
# 1. Start server
python tech.py

# 2. Upload PDF (in another terminal)
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@flight_document.pdf"

# 3. Ask question
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \

  -d '{"question": "What is the maximum altitude?"}'
