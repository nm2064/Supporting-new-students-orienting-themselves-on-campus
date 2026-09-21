# Architecture

The browser application in `index.html` uses React loaded from a CDN. FastAPI serves it at `/` and exposes the chat, ingestion, session and map endpoints from the same process.

```text
Browser (React)
    |
FastAPI application (agentic_rag/app.py)
    |-- Chat / ingestion / sessions (api.py)
    |      `-- RAG runtime (rag.py)
    |             |-- Knowledge files -> chunks -> Chroma vector store
    |             `-- OpenAI or Azure OpenAI embeddings and generation
    `-- Campus maps (maps/router.py)
           |-- Campus places JSON
           `-- OpenRouteService walking directions, cache and rate limits
```

`settings.py` resolves configuration and local paths; `schemas.py` defines the chat API payloads. `main.py` and `__main__.py` retain the supported module entry points. `start.py` manages local setup and starts the same backend.

Conversation state and route caches are runtime state. Chroma persists locally under `agentic_rag/chroma_store/` by default. These stores, credentials and logs are not published.

The application is a dissertation prototype. Its current configuration permits broad CORS access and exposes session and ingestion endpoints without authentication. The launcher binds to localhost; deployment for shared use would require access controls and an explicit hosting configuration.

The root `data/` folder contains research inputs and intermediate outputs. Runtime knowledge remains inside the backend package, so moving the collection materials does not change the configured knowledge paths. Archive code has its own historical dependencies and is excluded from the active test suite.
