"""Maps service orchestration: places lookup, routing, caching, and localization."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Protocol

from ..llm import LLMGateway
from ..settings import get_settings
from .cache import TTLCache, build_route_cache_key
from .models import Locale, PlaceDetail, PlaceSummary, RouteRequest, RouteResponse, RouteStep
from .places_store import PlacesStore
from .providers.base import RouteProvider
from .providers.openrouteservice_provider import OpenRouteServiceProviderError

logger = logging.getLogger("agentic_rag.maps.service")


class InstructionTranslator(Protocol):
    async def translate_steps(self, steps: list[RouteStep], locale: Locale) -> list[RouteStep]:
        ...


class NoopInstructionTranslator:
    async def translate_steps(self, steps: list[RouteStep], locale: Locale) -> list[RouteStep]:
        return steps


class OpenAIInstructionTranslator:
    """Fallback translator for provider instruction localization, via the LLM gateway."""

    _LANG_NAME = {
        Locale.en: "English",
        Locale.zh_hans: "Simplified Chinese",
        Locale.hi: "Hindi",
        Locale.ar: "Arabic",
    }

    def __init__(self, gateway: LLMGateway | None = None):
        settings = get_settings()
        if gateway is None and settings.effective_openai_api_key:
            gateway = LLMGateway(settings)
        self._gateway = gateway

    async def translate_steps(self, steps: list[RouteStep], locale: Locale) -> list[RouteStep]:
        if locale == Locale.en or not steps or self._gateway is None:
            return steps

        instructions = [step.instruction for step in steps]
        translated = await asyncio.to_thread(self._translate_texts, instructions, locale)
        if not translated or len(translated) != len(steps):
            return steps

        return [
            RouteStep(
                instruction=text.strip() or original.instruction,
                distance_m=original.distance_m,
                duration_s=original.duration_s,
            )
            for original, text in zip(steps, translated)
        ]

    def _translate_texts(self, texts: list[str], locale: Locale) -> list[str]:
        language_name = self._LANG_NAME.get(locale, "English")
        try:
            parsed = self._gateway.chat_json(
                system_prompt=(
                    "Translate each navigation instruction into "
                    f"{language_name}. Return strict JSON as {{\"instructions\": [..]}} "
                    "with the same order and count."
                ),
                user_prompt=json.dumps(texts, ensure_ascii=False),
                max_tokens=800,
            )
        except Exception as error:
            logger.warning("Route instruction translation failed: %s", type(error).__name__)
            return texts
        items = parsed.get("instructions")
        if isinstance(items, list) and all(isinstance(item, str) for item in items):
            return items
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
