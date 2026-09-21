# UniBot backend

FastAPI backend for retrieval-augmented chat, multilingual responses, conversation sessions and campus navigation. See the [project setup guide](../README.md#how-to-run-unibot-on-your-computer) for installation and configuration.

## Run and test

From the repository root with the environment activated:

```console
python -m agentic_rag
python -m pytest agentic_rag/tests -q -p no:cacheprovider
```

Alternatively, `python start.py` manages local setup and launches the application. The frontend is available at `/`, and the interactive API reference at `/docs`.

## Components

- `app.py`: application factory, lifecycle and frontend serving.
- `api.py`, `schemas.py`: ingestion, chat, health and session contracts.
- `rag.py`: document processing, retrieval and generation.
- `settings.py`, `config.py`: environment configuration and compatibility exports.
- `maps/`: campus places, routing provider, caching and rate limits.
- `tests/`: regression tests with substitute external services.

## Local data and credentials

Copy `.env.example` to `.env` and supply your own credentials. Paths in that configuration are relative to this package unless absolute. The example uses `knowledge_extracted_pages.json`; the minimal fallback in settings is `knowledge.txt`. Keep `data/campus_places.json` available for navigation.

`knowledge_curated_links.txt` and the extracted-page files are preserved research artifacts. Use `POST /ingest` after selecting the knowledge file. The generated `chroma_store/` directory is excluded from Git.

See [architecture](../docs/architecture.md), [frontend integration](../docs/frontend-integration.md) and [data provenance](../data/README.md) for the wider project context.
