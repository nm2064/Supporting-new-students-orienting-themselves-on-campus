"""LLM gateway: the one module that knows how to talk to OpenAI and Azure OpenAI.

Callers ask for a JSON completion or embeddings. Provider selection (Azure v1,
legacy Azure, OpenAI), Responses vs Chat Completions, token-parameter names,
temperature support and reasoning effort all stay behind this interface.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from .logging_utils import log_event
from .settings import AppSettings

logger = logging.getLogger("agentic_rag.llm")

_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")


class LLMGateway:
    """Lazy OpenAI / Azure OpenAI client with model quirks handled in one place."""

    def __init__(self, settings: AppSettings, client: Any | None = None):
        self.settings = settings
        self._client: Any | None = client
        self._lock = threading.Lock()

    # ---- client construction -------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            self.settings.validate_required()
            if self.settings.use_azure_openai and self.settings.azure_openai_endpoint:
                if self._azure_v1():
                    from openai import OpenAI

                    self._client = OpenAI(
                        api_key=self.settings.effective_openai_api_key,
                        base_url=self.settings.azure_openai_endpoint.rstrip("/") + "/",
                        timeout=45.0,
                        max_retries=1,
                    )
                else:
                    from openai import AzureOpenAI

                    self._client = AzureOpenAI(
                        api_key=self.settings.effective_openai_api_key,
                        api_version=self.settings.azure_openai_api_version,
                        azure_endpoint=self.settings.azure_openai_endpoint,
                        timeout=45.0,
                        max_retries=1,
                    )
                log_event(
                    logger,
                    "openai_client_initialized",
                    provider="azure_openai",
                    endpoint=self.settings.azure_openai_endpoint,
                )
            else:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.settings.effective_openai_api_key)
                log_event(logger, "openai_client_initialized", provider="openai")

        return self._client

    def _azure_v1(self) -> bool:
        return self.settings.use_azure_openai and self.settings.azure_openai_endpoint.rstrip("/").endswith(
            "/openai/v1"
        )

    def _use_responses(self, model: str) -> bool:
        return self._azure_v1() or model.lower().startswith("gpt-6")

    # ---- model naming and quirks ---------------------------------------------

    def _chat_model_name(self, fallback_model: str | None) -> str:
        if self.settings.use_azure_openai and self.settings.azure_openai_chat_deployment:
            return self.settings.azure_openai_chat_deployment
        if self.settings.use_azure_openai and self.settings.azure_openai_deployment:
            return self.settings.azure_openai_deployment
        return fallback_model or self.settings.chat_model

    def _embedding_model_name(self) -> str:
        if self.settings.use_azure_openai and self.settings.azure_openai_embedding_deployment:
            return self.settings.azure_openai_embedding_deployment
        if self.settings.use_azure_openai and self.settings.azure_openai_deployment:
            return self.settings.azure_openai_deployment
        return self.settings.embedding_model

    @staticmethod
    def _is_reasoning_model(model_name: str) -> bool:
        return model_name.lower().startswith(_REASONING_MODEL_PREFIXES)

    # ---- requests -------------------------------------------------------------

    def _response_text(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        params: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if self._is_reasoning_model(model) and self.settings.chat_reasoning_effort:
            params["reasoning"] = {"effort": self.settings.chat_reasoning_effort}
        if json_mode:
            params["text"] = {"format": {"type": "json_object"}}
        response = self._get_client().responses.create(**params)
        text = (response.output_text or "").strip()
        if getattr(response, "status", "completed") != "completed" or not text:
            raise RuntimeError("Model response did not complete with usable text")
        return text

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._get_client().embeddings.create(
            model=self._embedding_model_name(),
            input=texts,
        )
        return [item.embedding for item in response.data]

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        chat_model = self._chat_model_name(model)
        budget = max_tokens or self.settings.chat_max_tokens
        # JSON mode is rejected unless the input itself asks for JSON.
        if "json" not in f"{system_prompt} {user_prompt}".lower():
            system_prompt = f"{system_prompt}\nRespond with a single JSON object."
        if self._use_responses(chat_model):
            content = self._response_text(system_prompt, user_prompt, chat_model, budget, json_mode=True)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object from the model")
            return parsed

        params: dict[str, Any] = {
            "model": chat_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._is_reasoning_model(chat_model):
            params["max_completion_tokens"] = budget
        else:
            params["max_tokens"] = budget
            params["temperature"] = 0.1

        response = self._get_client().chat.completions.create(**params)
        content = (response.choices[0].message.content or "{}").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Model returned invalid JSON. Falling back to empty object.")
            return {}
        return parsed if isinstance(parsed, dict) else {}
