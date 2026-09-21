# UniBot — Supporting New Students on Campus

UniBot is an AI-assisted student onboarding application developed for a BSc Computer Science dissertation at Heriot-Watt University. It helps new students, particularly international students, find university services, understand administrative procedures and navigate campus through multilingual, context-aware conversations grounded in university information.

The repository brings together the implementation, collected knowledge, evaluation materials and earlier retrieval experiments. The original research and planning history is retained in the [Gantt chart](Gantt%20Chart/README.md).

## Features

- Retrieval-augmented answers with source references, follow-up suggestions and conversation context.
- Multilingual chat with language detection and translation support.
- Campus place search and walking directions through OpenRouteService.
- A React frontend served alongside the FastAPI backend.
- Saved research datasets, an evaluation runner and backend regression tests.

## Screenshots

| Chat | Campus navigation | Multilingual chat |
| --- | --- | --- |
| ![UniBot chat](images/unibot_chat_interface.png) | ![Campus map](images/unibot_map_interface.png) | ![Multilingual chat](images/unibot_multilingual_chat.png) |

## Quick start

Use Python 3.11 or newer. The application requires configured OpenAI or Azure OpenAI services; walking routes also require an OpenRouteService key. The browser interface loads external JavaScript, fonts and map resources, so an internet connection is required.

From the repository root:

```console
python start.py
```

The launcher creates `.venv`, installs the backend dependencies and creates `agentic_rag/.env` from the example if it is missing. On the first run, edit that file with your service configuration, then run the command again. The launcher checks dependencies on subsequent starts as well.

Open **http://localhost:8000** for the application or **http://localhost:8000/docs** for the interactive API reference.

### Manual setup

```console
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```console
python -m pip install -r agentic_rag/requirements.txt
```

Copy `agentic_rag/.env.example` to `agentic_rag/.env`, complete the configuration and run:

```console
python -m agentic_rag
```

To install both backend and optional data-collection dependencies, use `python -m pip install -r requirements.txt` instead.

### Configuration

The example file enables Azure OpenAI. Set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT` and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` to your own resource values. Chat and embedding deployments must be configured separately.

For OpenAI directly, set `USE_AZURE_OPENAI=false` and supply `OPENAI_API_KEY`. Set `ORS_API_KEY` to enable walking directions. Keep credentials in your local `.env`; it is excluded from Git.

The example selects `agentic_rag/knowledge_extracted_pages.json`. Relative knowledge, map-data and vector-store paths resolve from `agentic_rag/`. With the server running, use `POST /ingest` in the API reference to build or refresh the local vector store. The data is a saved research snapshot and may contain information that has since changed.

## Repository layout

| Location | Purpose |
| --- | --- |
| `agentic_rag/` | Backend, runtime knowledge, campus places and regression tests |
| `index.html`, `images/` | Frontend and project screenshots |
| `scripts/` | Website crawlers and text-extraction tools |
| `data/raw/`, `data/processed/` | Collected source records and reviewed/extracted datasets |
| `evaluation/` | Evaluation runner and preserved April results |
| `docs/` | Architecture, frontend guide, presentation and development history |
| `archive/` | Inactive retrieval prototypes and reference materials |
| `Gantt Chart/` | Original dissertation planning documents |

## Development and evaluation

```console
python -m pytest agentic_rag/tests -q -p no:cacheprovider
```

The regression tests use substitute services and do not require live model calls. The separate evaluation runner sends real queries to a running, configured application and may incur API usage:

```console
python evaluation/run_evaluation_tests.py
```

New results go to ignored `evaluation/results/latest.json`. The preserved [April baseline](evaluation/results/baseline-2026-04-09.json) is kept separately. See the [evaluation notes](evaluation/README.md) for interpretation and limitations.

## Documentation

- [Architecture](docs/architecture.md)
- [Backend reference](agentic_rag/README.md)
- [Frontend integration](docs/frontend-integration.md)
- [Data collection and provenance](data/README.md)
- [Development history and reconstruction method](docs/development-history.md)
- [Project presentation](docs/presentation.html) — open the HTML file in a browser
- [Archived experiments](archive/README.md)

Collected university content and third-party reference materials retain their original ownership and acknowledgements. No new licence is asserted for those materials.
