"""Test doubles for system boundaries (OpenAI SDK, embeddings, translation)."""

from dataclasses import replace
from types import SimpleNamespace

from agentic_rag.settings import get_settings


class FakeOpenAIClient:
    """Stands in for the OpenAI SDK client at the system boundary; records requests."""

    def __init__(self, output_text='{"ok": true}', status="completed"):
        self.response_requests: list[dict] = []
        self.completion_requests: list[dict] = []
        self.embedding_requests: list[dict] = []
        self._output_text = output_text
        self._status = status
        self.responses = SimpleNamespace(create=self._create_response)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create_completion))
        self.embeddings = SimpleNamespace(create=self._create_embedding)

    def _create_response(self, **params):
        self.response_requests.append(params)
        return SimpleNamespace(output_text=self._output_text, status=self._status)

    def _create_completion(self, **params):
        self.completion_requests.append(params)
        message = SimpleNamespace(content=self._output_text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _create_embedding(self, **params):
        self.embedding_requests.append(params)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5, 0.5]) for _ in params["input"]])


def luna_settings(**overrides):
    return replace(
        get_settings(),
        use_azure_openai=True,
        azure_openai_endpoint="https://example.services.ai.azure.com/openai/v1",
        azure_openai_api_key="unit-test-key",
        azure_openai_chat_deployment="gpt-6-luna",
        azure_openai_embedding_deployment="text-embedding-3-small",
        **overrides,
    )
