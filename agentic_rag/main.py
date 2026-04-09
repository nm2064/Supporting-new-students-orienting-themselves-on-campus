"""Compatibility entrypoint for running the backend directly."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic_rag.app import app
from agentic_rag.logging_utils import configure_logging
from agentic_rag.settings import get_settings


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "agentic_rag.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    run()
