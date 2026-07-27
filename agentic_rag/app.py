"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import router as api_router
from .config import validate_config
from .logging_utils import configure_logging
from .maps.router import router as maps_router
from .settings import get_settings

# Root of the whole project (one level above this package)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("agentic_rag.app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting Agentic RAG backend")
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except Exception as error:
        logger.warning("Configuration validation failed: %s", error)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic RAG API",
        description="Agentic RAG system with ChromaDB and OpenAI-powered multilingual generation",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(maps_router)

    # --- Serve frontend static files ---
    # Mount the images directory so index.html can reference /images/...
    images_dir = PROJECT_ROOT / "images"
    if images_dir.exists():
        app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    # Serve index.html at the root URL
    index_file = PROJECT_ROOT / "index.html"

    @app.get("/", include_in_schema=False)
    async def serve_frontend() -> FileResponse:
        return FileResponse(str(index_file), media_type="text/html")

    return app


app = create_app()
