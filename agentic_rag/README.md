# Agentic RAG Backend

A complete Agentic RAG (Retrieval-Augmented Generation) system built with FastAPI, ChromaDB, OpenAI embeddings, and Google Gemini.

## Architecture

```
Frontend (index.html)
         │
         │ POST /rag-chat
         ▼
┌─────────────────────────────────────┐
│         FastAPI Backend             │
│  ┌─────────┐    ┌───────────────┐   │
│  │ Router  │───>│ DIRECT_ANSWER │   │
│  └────┬────┘    └───────────────┘   │
│       │                              │
│       ├──> RETRIEVE ──┐              │
│       │                │              │
│       └──> CLARIFY     ▼              │
│                     ┌──────────┐      │
│                     │ ChromaDB │      │
│                     │+ OpenAI  │      │
│                     │Embeddings│      │
│                     └────┬─────┘      │
│                          │            │
│                     ┌────▼─────┐      │
│                     │  Gemini  │      │
│                     │  (LLM)   │      │
│                     └──────────┘      │
└─────────────────────────────────────┘
```

## Project Structure

```
agentic_rag/
├── main.py              # FastAPI application
├── rag.py               # Agentic RAG implementation
├── config.py            # Configuration
├── requirements.txt     # Python dependencies
├── knowledge.txt        # Knowledge base (edit this)
├── chroma_store/        # ChromaDB persistence (auto-created)
└── .env                 # Environment variables (create from .env.example)
```

## Prerequisites

- **Python 3.9+**
- **OpenAI API Key** - for embeddings
- **Google API Key** - for Gemini LLM

## Setup

### 1. Navigate to this directory

```bash
cd agentic_rag
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
```

**Required in `.env`:**
```env
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

### 5. Customize Knowledge Base

Edit `knowledge.txt` with your organization-specific information.

### 6. Run Server

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `http://localhost:8000/health` | GET | Health check |
| `http://localhost:8000/ingest` | POST | Ingest knowledge file |
| `http://localhost:8000/rag-chat` | POST | Chat with agentic RAG |
| `http://localhost:8000/session/{id}` | GET/DELETE | Session management |

## Testing

```bash
# Ingest knowledge
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Chat
curl -X POST http://localhost:8000/rag-chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the library hours?"}'
```

## Frontend Connection

The frontend (`../index.html`) expects the backend at `http://localhost:8000/rag-chat`.

See `../FRONTEND_INTEGRATION.md` for detailed frontend integration instructions.

## Troubleshooting

See the main `../README.md` for full troubleshooting guide.
