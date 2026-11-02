"""Seam S1: the LLM gateway interface (complete JSON, embed) against a fake OpenAI client."""

import pytest

from agentic_rag.llm import LLMGateway
from agentic_rag.tests.fakes import FakeOpenAIClient, luna_settings


def test_luna_json_completion_uses_responses_api_with_configured_reasoning_and_budget():
    client = FakeOpenAIClient(output_text='{"answer": "yes"}')
    gateway = LLMGateway(luna_settings(chat_reasoning_effort="medium", chat_max_tokens=2500), client=client)

    result = gateway.chat_json("Return JSON.", "Is the library open?")

    assert result == {"answer": "yes"}
    request = client.response_requests[0]
    assert request["model"] == "gpt-6-luna"
    assert request["reasoning"] == {"effort": "medium"}
    assert request["max_output_tokens"] == 2500
    assert request["text"] == {"format": {"type": "json_object"}}
    assert request["store"] is False
    assert "temperature" not in request


def test_azure_v1_endpoint_builds_standard_client_without_legacy_api_version(monkeypatch):
    built: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: built.append(kwargs) or FakeOpenAIClient())

    LLMGateway(luna_settings()).chat_json("Return JSON.", "ping")

    assert built[0]["base_url"].endswith("/openai/v1/")
    assert built[0]["api_key"] == "unit-test-key"
    assert "api_version" not in built[0]


def test_embeddings_use_the_embedding_deployment_not_the_chat_deployment():
    client = FakeOpenAIClient()
    gateway = LLMGateway(luna_settings(), client=client)

    assert gateway.embed_texts(["library"]) == [[0.5, 0.5]]
    assert client.embedding_requests[0]["model"] == "text-embedding-3-small"


def test_incomplete_model_response_is_an_error_not_an_answer():
    client = FakeOpenAIClient(output_text='{"truncated":', status="incomplete")
    gateway = LLMGateway(luna_settings(), client=client)

    with pytest.raises(RuntimeError):
        gateway.chat_json("Return JSON.", "Check")


def test_json_completion_always_tells_the_model_to_reply_in_json():
    client = FakeOpenAIClient()
    gateway = LLMGateway(luna_settings(), client=client)

    gateway.chat_json("Translate these instructions.", "Turn left")

    sent_text = " ".join(message["content"] for message in client.response_requests[0]["input"])
    assert "json" in sent_text.lower()
