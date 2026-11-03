from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agentic_rag.public_access import PublicAccessMiddleware


def client(monkeypatch, tmp_path, limit=200):
    monkeypatch.setenv("PUBLIC_SESSION_SECRET", "test-secret-not-for-production")
    monkeypatch.setenv("PUBLIC_ORIGIN", "https://example.test")
    monkeypatch.setenv("PUBLIC_LIMIT_DB", str(tmp_path / "limits.sqlite3"))
    monkeypatch.setenv("PUBLIC_DAILY_REQUEST_LIMIT", str(limit))
    app = FastAPI()
    app.add_middleware(PublicAccessMiddleware)

    @app.post("/rag-chat/stream")
    @app.post("/rag-chat")
    async def chat(request: Request):
        return {"session": request.state.public_session}

    return TestClient(app, base_url="https://example.test")


def test_maintenance_routes_are_not_public(monkeypatch, tmp_path):
    c = client(monkeypatch, tmp_path)
    for path in ["/ingest", "/docs", "/openapi.json", "/session/example", "/health"]:
        assert c.get(path).status_code == 404


def test_rejects_cross_origin_and_large_request(monkeypatch, tmp_path):
    c = client(monkeypatch, tmp_path)
    assert c.post("/rag-chat", headers={"Origin": "https://other.test"}).status_code == 403
    assert c.post("/rag-chat", content="x" * 16385).status_code == 413


def test_session_is_private_stable_and_signed(monkeypatch, tmp_path):
    c = client(monkeypatch, tmp_path)
    first = c.post("/rag-chat", json={"session_id": "attacker-supplied"})
    session = first.json()["session"]
    assert session != "attacker-supplied"
    assert "HttpOnly" in first.headers["set-cookie"]
    assert "Secure" in first.headers["set-cookie"]
    assert c.post("/rag-chat").json()["session"] == session
    second = client(monkeypatch, tmp_path)
    assert second.post("/rag-chat", headers={"Cookie": f"unibot_session={session}.forged"}).json()["session"] != session


def test_daily_limit_survives_app_restart(monkeypatch, tmp_path):
    c = client(monkeypatch, tmp_path, limit=2)
    assert c.post("/rag-chat").status_code == 200
    assert c.post("/rag-chat").status_code == 200
    fresh = client(monkeypatch, tmp_path, limit=2)
    assert fresh.post("/rag-chat").status_code == 429


def test_per_ip_limit(monkeypatch, tmp_path):
    c = client(monkeypatch, tmp_path)
    for _ in range(10):
        assert c.post("/rag-chat").status_code == 200
    assert c.post("/rag-chat").status_code == 429


def test_reset_changes_cookie_and_clears_server_conversation(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from agentic_rag import rag
    fake = Mock()
    monkeypatch.setattr(rag, 'rag_instance', fake)
    c = client(monkeypatch, tmp_path)
    previous = c.post('/rag-chat').json()['session']
    assert c.post('/reset-chat').status_code == 200
    fake.clear_session.assert_called_once_with(previous)
    assert c.post('/rag-chat').json()['session'] != previous


def test_starting_a_new_conversation_does_not_use_up_the_ai_request_allowance(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from agentic_rag import rag
    monkeypatch.setattr(rag, 'rag_instance', Mock())
    c = client(monkeypatch, tmp_path, limit=2)
    for _ in range(15):  # e.g. a visitor reloading the page many times
        assert c.post('/reset-chat').status_code == 200
    assert c.post('/rag-chat').status_code == 200
    assert c.post('/rag-chat').status_code == 200


def test_stream_keeps_signed_session_origin_checks_and_quota(monkeypatch, tmp_path):
    c = client(monkeypatch, tmp_path, limit=1)
    assert c.post('/rag-chat/stream', headers={'Origin': 'https://other.test'}).status_code == 403
    first = c.post('/rag-chat/stream')
    assert first.status_code == 200
    assert 'HttpOnly' in first.headers['set-cookie']
    assert c.post('/rag-chat/stream').status_code == 429
