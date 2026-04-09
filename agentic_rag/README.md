# Agentic RAG Backend

FastAPI backend for the UniBot project, with RAG chat, session handling, and campus maps endpoints.

## Architecture

```text
Frontend (index.html)
        |
        | POST /rag-chat
        v
FastAPI app factory (app.py)
  |- API routes (api.py)
  |- Maps routes (maps/)
  |- RAG service runtime (rag.py)
  |- Typed settings (settings.py)
  `- Structured logging (logging_utils.py)
```

## Project Structure

```text
agentic_rag/
├── app.py               # FastAPI app factory
├── api.py               # Core HTTP routes
├── main.py              # Direct-run compatibility entrypoint
├── rag.py               # RAG service runtime
├── settings.py          # Environment parsing and validation
├── schemas.py           # API models
├── config.py            # Backward-compatible config exports
├── maps/                # Places search and routing
├── tests/               # Regression tests
├── requirements.txt     # Python dependencies
├── knowledge.txt        # Knowledge base
├── chroma_store/        # Chroma persistence
└── .env                 # Create from .env.example
```

## Setup

```bash
cd UniBot
python -m venv .venv
.venv\Scripts\activate
pip install -r agentic_rag\requirements.txt
copy agentic_rag\.env.example agentic_rag\.env
```

Required in `agentic_rag/.env`:

```env
AZURE_OPENAI_API_KEY=your-azure-key
USE_AZURE_OPENAI=true
AZURE_OPENAI_ENDPOINT=https://openaidis.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-nano-2
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
CHAT_MODEL=gpt-5-nano
TRANSLATION_MODEL=gpt-5-nano
```

Notes:
- `AZURE_OPENAI_CHAT_DEPLOYMENT` should be your GPT-5 nano deployment name.
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` must be a separate embedding deployment; do not point embeddings at GPT-5 nano.

## Run

Preferred:

```bash
python -m agentic_rag
```

Alternative:

```bash
uvicorn agentic_rag.app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health`
- `POST /ingest`
- `POST /rag-chat`
- `GET /session/{id}`
- `DELETE /session/{id}`
- `GET /api/places`
- `GET /api/places/{id}`
- `POST /api/route`

## Testing

```bash
python -m pytest agentic_rag/tests -q -p no:cacheprovider
```
