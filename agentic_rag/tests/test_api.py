from fastapi.testclient import TestClient

from agentic_rag.app import create_app
from agentic_rag.maps.router import get_maps_service
from agentic_rag.rag import RagResponse, get_rag


class _VectorStoreStub:
    collection = object()


class FakeRag:
    def __init__(self):
        self.vector_store = _VectorStoreStub()
        self.last_chat = None

    def ingest(self, force: bool = False):
        return {
            "status": "success",
            "message": "ok",
            "chunks": 1,
            "file_hash": "abc123",
        }

    def chat(
        self,
        query: str,
        session_id: str | None = None,
        ui_language: str = "en",
        auto_detect: bool = True,
        campus: str | None = None,
        location_context: dict | None = None,
    ) -> RagResponse:
        self.last_chat = {
            "query": query,
            "session_id": session_id,
            "ui_language": ui_language,
            "auto_detect": auto_detect,
            "campus": campus,
            "location_context": location_context,
        }
        return RagResponse(
            answer_markdown="Hello",
            effective_language="en",
            answer="Hello",
            sources=[],
            used_retrieval=False,
            conversation_id=session_id or "generated-id",
            session_id=session_id or "generated-id",
        )

    def get_session_history(self, session_id: str):
        return [{"role": "user", "content": "hello"}]

    def clear_session(self, session_id: str):
        return None


class FakeMapsService:
    def search_places(self, query: str, campus: str | None = None, limit: int = 20):
        return []

    def get_place(self, place_id: str):
        return None

    async def get_route(self, request):
        raise AssertionError("route endpoint is not part of this test")


def _client_with_overrides():
    app = create_app()
    rag = FakeRag()
    app.dependency_overrides[get_rag] = lambda: rag
    app.dependency_overrides[get_maps_service] = lambda: FakeMapsService()
    return TestClient(app), rag


def test_health_endpoint_reports_backend_state():
    client, _ = _client_with_overrides()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "degraded"}
    assert payload["chroma_initialized"] is True
    assert "configuration_valid" in payload


def test_rag_chat_accepts_legacy_fields():
    client, rag = _client_with_overrides()

    response = client.post(
        "/rag-chat",
        json={
            "question": "Where is the library?",
            "session_id": "legacy-session",
            "ui_language": "en",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer_markdown"] == "Hello"
    assert rag.last_chat == {
        "query": "Where is the library?",
        "session_id": "legacy-session",
        "ui_language": "en",
        "auto_detect": True,
        "campus": None,
        "location_context": None,
    }
