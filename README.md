# Agentic RAG Chatbot

A complete Agentic RAG (Retrieval-Augmented Generation) system with React frontend and Python backend.

## Project Structure

```
UniBot/
├── agentic_rag/           # 🔧 Backend (self-contained)
│   ├── main.py
│   ├── rag.py
│   ├── config.py
│   ├── requirements.txt
│   ├── knowledge.txt      # ← Edit this with your data
│   ├── .env.example       # ← Copy to .env and add keys
│   ├── chroma_store/      # Vector DB persistence
│   └── README.md
│
├── FRONTEND_INTEGRATION.md # Frontend integration guide
└── index.html             # 🎨 Frontend (React CDN)
```

## Quick Start

### 1. Start Backend

```bash
cd agentic_rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure (add your API keys)
cp .env.example .env
# Edit .env with your keys

# Run server
python main.py
```

### 2. Open Frontend

Open `index.html` in your browser, or serve it:
```bash
# From project root
python -m http.server 3000
# Then visit http://localhost:3000
```

## Configuration

Add to `agentic_rag/.env`:
```env
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-key-here
```

## API Usage

```bash
# Ingest knowledge
curl -X POST http://localhost:8000/ingest -d '{"force":true}'

# Chat
curl -X POST http://localhost:8000/rag-chat \
  -d '{"question": "What are the library hours?"}'
```

## Documentation

- **Backend details**: See `agentic_rag/README.md`
- **Frontend integration**: See `FRONTEND_INTEGRATION.md`

## Features

- ✅ ChromaDB vector store with persistence
- ✅ OpenAI embeddings
- ✅ Google Gemini LLM
- ✅ Smart routing (Direct/Retrieve/Clarify)
- ✅ Session management
- ✅ Source citations
- ✅ File hash checking (no redundant embedding)
