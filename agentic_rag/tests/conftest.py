import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def reset_runtime_state():
    from agentic_rag.maps.router import reset_maps_service_cache
    from agentic_rag.rag import reset_rag
    from agentic_rag.settings import reload_settings

    reload_settings()
    reset_rag()
    reset_maps_service_cache()
    yield
    reset_rag()
    reset_maps_service_cache()
    reload_settings()
