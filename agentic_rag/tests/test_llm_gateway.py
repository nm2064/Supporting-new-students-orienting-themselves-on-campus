"""Seam S1: the LLM gateway interface (complete JSON, embed) against a fake OpenAI client."""

import pytest

from agentic_rag.llm import LLMGateway
from agentic_rag.tests.fakes import FakeOpenAIClient, luna_settings, nvidia_settings


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


# ---- NVIDIA hosted models (OpenAI-compatible chat completions) -----------------


def test_nvidia_json_completion_uses_chat_completions_in_json_mode_with_the_answer_budget():
    client = FakeOpenAIClient(output_text='{"answer_markdown": "Open 24 hours [1]."}')
    gateway = LLMGateway(nvidia_settings(chat_max_tokens=1600), client=client)

    result = gateway.chat_json("Return JSON.", "When is the library open?", max_tokens=3000)

    assert result == {"answer_markdown": "Open 24 hours [1]."}
    request = client.completion_requests[0]
    assert request["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 3000
    assert client.response_requests == []


def test_nvidia_client_points_at_the_nvidia_endpoint_with_the_nvidia_key(monkeypatch):
    built: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: built.append(kwargs) or FakeOpenAIClient())

    LLMGateway(nvidia_settings()).chat_json("Return JSON.", "ping")

    assert built[0]["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert built[0]["api_key"] == "nvapi-unit-test"


def test_nvidia_embeddings_mark_documents_as_passages_and_questions_as_queries():
    client = FakeOpenAIClient()
    gateway = LLMGateway(nvidia_settings(), client=client)

    gateway.embed_texts(["The library is open 24 hours."])
    gateway.embed_texts(["When is the library open?"], input_type="query")

    passage_request, query_request = client.embedding_requests
    assert passage_request["model"] == "nvidia/nemotron-3.5-embed-1b"
    assert passage_request["extra_body"] == {"input_type": "passage", "truncate": "END"}
    assert query_request["extra_body"] == {"input_type": "query", "truncate": "END"}


@pytest.mark.parametrize(
    "raw_reply",
    [
        '<think>The evidence says 24 hours.</think>\n{"answer_markdown": "Open 24 hours [1]."}',
        'Here you go:\n```json\n{"answer_markdown": "Open 24 hours [1]."}\n```',
    ],
)
def test_json_is_recovered_when_the_model_wraps_it_in_reasoning_or_a_code_fence(raw_reply):
    gateway = LLMGateway(nvidia_settings(), client=FakeOpenAIClient(output_text=raw_reply))

    assert gateway.chat_json("Return JSON.", "When is the library open?") == {
        "answer_markdown": "Open 24 hours [1]."
    }


def test_missing_nvidia_key_is_reported_by_name():
    from agentic_rag.settings import ConfigurationError

    with pytest.raises(ConfigurationError, match="NVIDIA_API_KEY"):
        nvidia_settings(nvidia_api_key="").validate_required()


class FlakyEmbeddingClient(FakeOpenAIClient):
    """NVIDIA's hosted embedder intermittently answers 500 'service inference failed'."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures_left = failures

    def _create_embedding(self, **params):
        if self.failures_left:
            self.failures_left -= 1
            raise RuntimeError("Error code: 500 - service inference failed")
        return super()._create_embedding(**params)


def test_transient_embedding_failures_are_retried_until_they_succeed():
    client = FlakyEmbeddingClient(failures=3)
    gateway = LLMGateway(nvidia_settings(), client=client, retry_delay_s=0)

    assert gateway.embed_texts(["The library is open 24 hours."]) == [[0.5, 0.5]]


def test_embedding_gives_up_after_repeated_failures():
    gateway = LLMGateway(nvidia_settings(), client=FlakyEmbeddingClient(failures=50), retry_delay_s=0)

    with pytest.raises(RuntimeError, match="service inference failed"):
        gateway.embed_texts(["The library is open 24 hours."])
