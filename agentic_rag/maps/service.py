"""Maps service orchestration: places lookup, routing, caching, and localization."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from ..settings import get_settings
from .cache import TTLCache, build_route_cache_key
from .models import Locale, PlaceDetail, PlaceSummary, RouteRequest, RouteResponse, RouteStep
from .places_store import PlacesStore
from .providers.base import RouteProvider
from .providers.openrouteservice_provider import OpenRouteServiceProviderError


class InstructionTranslator(Protocol):
    async def translate_steps(self, steps: list[RouteStep], locale: Locale) -> list[RouteStep]:
        ...


class NoopInstructionTranslator:
    async def translate_steps(self, steps: list[RouteStep], locale: Locale) -> list[RouteStep]:
        return steps


class OpenAIInstructionTranslator:
    """Fallback translator for provider instruction localization."""

    _LANG_NAME = {
        Locale.en: "English",
        Locale.zh_hans: "Simplified Chinese",
        Locale.hi: "Hindi",
        Locale.ar: "Arabic",
    }

    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        effective_api_key = api_key or settings.effective_openai_api_key
        self._uses_azure = settings.use_azure_openai and bool(settings.azure_openai_endpoint)
        self.model = self._resolve_model_name(settings, model)
        self._client = self._build_client(settings, effective_api_key)

    @staticmethod
    def _resolve_model_name(settings, fallback_model: str | None) -> str:
        if settings.use_azure_openai and settings.azure_openai_chat_deployment:
            return settings.azure_openai_chat_deployment
        if settings.use_azure_openai and settings.azure_openai_deployment:
            return settings.azure_openai_deployment
        return fallback_model or settings.translation_model

    def _build_client(self, settings, api_key: str | None):
        if not api_key:
            return None
        if self._uses_azure:
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=api_key,
                api_version=settings.azure_openai_api_version,
                azure_endpoint=settings.azure_openai_endpoint,
            )

        from openai import OpenAI

        return OpenAI(api_key=api_key)

    @staticmethod
    def _supports_custom_temperature(model_name: str) -> bool:
        return "gpt-5" not in model_name.lower()

    @staticmethod
    def _uses_completion_token_limit(model_name: str) -> bool:
        normalized = model_name.lower()
        return normalized.startswith(("gpt-5", "o1", "o3", "o4"))

    async def translate_steps(self, steps: list[RouteStep], locale: Locale) -> list[RouteStep]:
        if locale == Locale.en or not steps or self._client is None:
            return steps

        instructions = [step.instruction for step in steps]
        translated = await asyncio.to_thread(self._translate_texts, instructions, locale)
        if not translated or len(translated) != len(steps):
            return steps

        output: list[RouteStep] = []
        for original, text in zip(steps, translated):
            output.append(
                RouteStep(
                    instruction=text.strip() or original.instruction,
                    distance_m=original.distance_m,
                    duration_s=original.duration_s,
                )
            )
        return output

    def _translate_texts(self, texts: list[str], locale: Locale) -> list[str]:
        language_name = self._LANG_NAME.get(locale, "English")
        prompt = (
            "Translate each navigation instruction into "
            f"{language_name}. Return strict JSON as {{\"instructions\": [..]}} "
            "with the same order and count."
        )
        payload = json.dumps(texts, ensure_ascii=False)

        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ],
                "response_format": {"type": "json_object"},
            }
            if self._uses_completion_token_limit(self.model):
                params["max_completion_tokens"] = 800
            else:
                params["max_tokens"] = 800
            if self._supports_custom_temperature(self.model):
                params["temperature"] = 0

            response = self._client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            items = parsed.get("instructions")
            if isinstance(items, list) and all(isinstance(item, str) for item in items):
                return items
        except Exception:
            return texts

        return texts


@dataclass
class RoutingUnavailable(Exception):
    reason: str
    provider: str
    detail: str | None = None


class MapsService:
    def __init__(
        self,
        places_store: PlacesStore,
        provider: RouteProvider,
        cache: TTLCache[RouteResponse],
        timeout_s: float = 8.0,
        translator: InstructionTranslator | None = None,
    ):
        self.places_store = places_store
        self.provider = provider
        self.cache = cache
        self.timeout_s = timeout_s
        self.translator = translator or NoopInstructionTranslator()

    def search_places(self, query: str, campus: str | None = None, limit: int = 20) -> list[PlaceSummary]:
        return self.places_store.search(query=query, campus=campus, limit=limit)

    def get_place(self, place_id: str) -> PlaceDetail | None:
        return self.places_store.get(place_id)

    async def get_route(self, request: RouteRequest) -> tuple[RouteResponse, bool]:
        key = build_route_cache_key(request.origin, request.destination, request.profile, request.locale)
        cached = self.cache.get(key)
        if cached is not None:
            return cached.model_copy(deep=True), True

        try:
            route = await asyncio.wait_for(
                self.provider.get_route(
                    origin=request.origin,
                    destination=request.destination,
                    profile=request.profile,
                    locale=request.locale.value,
                ),
                timeout=self.timeout_s,
            )
        except TimeoutError as error:
            raise RoutingUnavailable(reason="timeout", provider=self.provider.name, detail=str(error)) from error
        except OpenRouteServiceProviderError as error:
            raise RoutingUnavailable(reason="provider_error", provider=self.provider.name, detail=str(error)) from error
        except Exception as error:
            raise RoutingUnavailable(reason="unexpected_error", provider=self.provider.name, detail=str(error)) from error

        route.steps = await self.translator.translate_steps(route.steps, request.locale)
        self.cache.set(key, route)
        return route.model_copy(deep=True), False
