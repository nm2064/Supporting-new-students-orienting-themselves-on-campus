# Agentic RAG Chatbot

A complete Retrieval-Augmented Generation system with a React frontend and a Python backend.

## Project Structure

```text
UniBot/
├── agentic_rag/            # Backend package
│   ├── app.py              # FastAPI app factory
│   ├── api.py              # HTTP route handlers
│   ├── main.py             # Direct-run compatibility entrypoint
│   ├── rag.py              # RAG runtime and service logic
│   ├── settings.py         # Typed environment configuration
│   ├── schemas.py          # API request/response models
│   ├── requirements.txt
│   ├── knowledge.txt       # Edit this with your data
│   ├── .env.example        # Copy to .env and add keys
│   ├── chroma_store/       # Vector DB persistence
│   └── tests/              # Backend regression tests
├── FRONTEND_INTEGRATION.md # Frontend integration guide
└── index.html              # React CDN frontend
```

## Quick Start

```bash
cd UniBot

python -m venv .venv
.venv\Scripts\activate
pip install -r agentic_rag\requirements.txt

copy agentic_rag\.env.example agentic_rag\.env
# Edit agentic_rag\.env with your API keys

python start.py
```

Then open **http://localhost:8000** in your browser.

Both the API and the frontend are served from the same process — no second terminal needed.

> **API docs** are available at http://localhost:8000/docs

## Configuration

Add to `agentic_rag/.env`:

```env
AZURE_OPENAI_API_KEY=your-azure-key
USE_AZURE_OPENAI=true
AZURE_OPENAI_ENDPOINT=https://openaidis.cognitiveservices.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-nano-2
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
CHAT_MODEL=gpt-5-nano
TRANSLATION_MODEL=gpt-5-nano
```

## API Usage

```bash
curl -X POST http://localhost:8000/ingest -d "{\"force\":true}"
curl -X POST http://localhost:8000/rag-chat -d "{\"question\":\"What are the library hours?\"}"
```

## Documentation

- Backend details: `agentic_rag/README.md`
- Frontend integration: `FRONTEND_INTEGRATION.md`

## Website Crawler

Use `crawl_hw_website.py` to crawl Heriot-Watt related sites and extract useful internal links for later review or ingestion.

```bash
.\.venv\Scripts\python.exe crawl_hw_website.py
```

Default site presets:

- `hwu`: Heriot-Watt University (`hw.ac.uk`)
- `watt-living`: Watt Living (`reslife.wattliving.co.uk`)
- `union`: Heriot-Watt Student Union (`hwunion.com`)
- `sports-union`: Heriot-Watt Sports Union (`sportsunion.site.hw.ac.uk`)

Examples:

```bash
.\.venv\Scripts\python.exe crawl_hw_website.py --sites all --max-pages-per-site 150
.\.venv\Scripts\python.exe crawl_hw_website.py --sites hwu watt-living union sports-union
.\.venv\Scripts\python.exe crawl_hw_website.py --sites union --output-prefix hwunion_links
```

Outputs:

- `hw_useful_links.jsonl`: structured link records with score and metadata
- `hw_useful_links.txt`: readable summary of useful links
- `hw_website_crawler.log`: crawl log

## Features

- ChromaDB vector store with persistence
- OpenAI embeddings and chat generation
- Smart routing (Direct/Retrieve/Clarify)
- Session management
- Source citations
- File hash checking to avoid redundant ingestion
- Structured logging and typed settings
- Backend regression tests
