from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agentic_rag.rag import OpenAIBackend
from agentic_rag.settings import get_settings


def backend():
    settings = replace(get_settings(), use_azure_openai=True,
                       azure_openai_endpoint="https://example.services.ai.azure.com/openai/v1",
                       azure_openai_api_key="unit-test-key",
                       azure_openai_chat_deployment="gpt-6-luna",
                       azure_openai_embedding_deployment="text-embedding-3-small")
    return OpenAIBackend(settings)


def test_azure_v1_uses_standard_client_without_legacy_api_version():
    b = backend()
    with patch('openai.OpenAI') as client:
        b._get_client()
    params = client.call_args.kwargs
    assert params['base_url'].endswith('/openai/v1/')
    assert params['api_key'] == 'unit-test-key'
    assert 'api_version' not in params


def test_luna_json_response_uses_valid_responses_schema():
    b = backend()
    b._client = Mock()
    b._client.responses.create.return_value = SimpleNamespace(output_text='{"ready":true}', status='completed')
    assert b.chat_json('Return JSON.', 'Check', 'unused') == {'ready': True}
    params = b._client.responses.create.call_args.kwargs
    assert params['model'] == 'gpt-6-luna'
    assert params['text'] == {'format': {'type': 'json_object'}}
    assert params['reasoning'] == {'effort': 'none'}
    assert params['store'] is False
    assert 'temperature' not in params
    assert 'max_tokens' not in params
    assert params['input'][0]['role'] == 'system'


def test_embedding_deployment_is_separate_from_chat():
    b = backend()
    b._client = Mock()
    b._client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    assert b.embed_texts(['test']) == [[0.1, 0.2]]
    assert b._client.embeddings.create.call_args.kwargs['model'] == 'text-embedding-3-small'


def test_incomplete_response_is_not_presented_as_a_valid_answer():
    b = backend()
    b._client = Mock()
    b._client.responses.create.return_value = SimpleNamespace(output_text='{"truncated":', status='incomplete')
    with pytest.raises(RuntimeError):
        b.chat_json('Return JSON.', 'Check', 'unused')
