"""LLM gateway: the one module that knows how to talk to OpenAI and Azure OpenAI.

Callers ask for a JSON completion or embeddings. Provider selection (Azure v1,
legacy Azure, OpenAI), Responses vs Chat Completions, token-parameter names,
temperature support and reasoning effort all stay behind this interface.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import openai

from .logging_utils import log_event
from .settings import AppSettings

logger = logging.getLogger("agentic_rag.llm")

_REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")

_PERMANENT_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.NotFoundError,
)


class LLMGateway:
    """Lazy OpenAI / Azure OpenAI client with model quirks handled in one place."""

    EMBEDDING_ATTEMPTS = 6

    def __init__(self, settings: AppSettings, client: Any | None = None, retry_delay_s: float = 1.0):
        self.settings = settings
        self._client: Any | None = client
        self._lock = threading.Lock()
        self._retry_delay_s = retry_delay_s

    # ---- client construction -------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            self.settings.validate_required()
            if self._nvidia():
                from openai import OpenAI

                # Large hosted models can take a while on thorough answers;
                # Caddy allows 120s, so stay under that.
                self._client = OpenAI(
                    api_key=self.settings.nvidia_api_key,
                    base_url=self.settings.nvidia_base_url,
                    timeout=90.0,
                    max_retries=1,
                )
                log_event(logger, "openai_client_initialized", provider="nvidia")
            elif self.settings.use_azure_openai and self.settings.azure_openai_endpoint:
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

    def _nvidia(self) -> bool:
        return self.settings.provider == "nvidia"

    def _azure_v1(self) -> bool:
        return (
            self.settings.provider == "azure"
            and self.settings.azure_openai_endpoint.rstrip("/").endswith("/openai/v1")
        )

    def _use_responses(self, model: str) -> bool:
        if self._nvidia():
            return False
        return self._azure_v1() or model.lower().startswith("gpt-6")

    # ---- model naming and quirks ---------------------------------------------

    def _chat_model_name(self, fallback_model: str | None) -> str:
        azure = self.settings.provider == "azure"
        if azure and self.settings.azure_openai_chat_deployment:
            return self.settings.azure_openai_chat_deployment
        if azure and self.settings.azure_openai_deployment:
            return self.settings.azure_openai_deployment
        return fallback_model or self.settings.chat_model

    def _embedding_model_name(self) -> str:
        azure = self.settings.provider == "azure"
        if azure and self.settings.azure_openai_embedding_deployment:
            return self.settings.azure_openai_embedding_deployment
        if azure and self.settings.azure_openai_deployment:
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

    def embed_texts(self, texts: list[str], input_type: str = "passage") -> list[list[float]]:
        """Embed ``texts``. ``input_type`` is "passage" for indexed documents and
        "query" for search questions; asymmetric retrieval models need it."""
        params: dict[str, Any] = {"model": self._embedding_model_name(), "input": texts}
        if self._nvidia():
            params["extra_body"] = {"input_type": input_type, "truncate": "END"}
        # Hosted embedders fail intermittently with 5xx; retry those with backoff.
        # Errors that retrying cannot fix (bad key, bad request) are raised at once.
        for attempt in range(1, self.EMBEDDING_ATTEMPTS + 1):
            try:
                response = self._get_client().embeddings.create(**params)
                return [item.embedding for item in response.data]
            except _PERMANENT_ERRORS:
                raise
            except Exception as error:
                if attempt == self.EMBEDDING_ATTEMPTS:
                    raise
                logger.warning("Embedding attempt %d failed (%s); retrying.", attempt, type(error).__name__)
                time.sleep(self._retry_delay_s * attempt)
        raise AssertionError("unreachable")

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
        if self._is_reasoning_model(chat_model) and not self._nvidia():
            params["max_completion_tokens"] = budget
        else:
            params["max_tokens"] = budget
            params["temperature"] = 0.1

        response = self._get_client().chat.completions.create(**params)
        choice = response.choices[0]
        parsed = _parse_json_object(choice.message.content or "")
        if parsed is None:
            logger.warning(
                "Model returned no usable JSON (finish_reason=%s).", getattr(choice, "finish_reason", None)
            )
            return {}
        return parsed


_THINKING = re.compile(r"(?s)<think>.*?</think>")


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model reply, tolerating reasoning text and code fences."""
    text = _THINKING.sub("", text).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
