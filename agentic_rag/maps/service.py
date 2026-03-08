"""Maps service orchestration: places lookup, routing, caching, and localization."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from config import OPENAI_API_KEY, TRANSLATION_MODEL

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

    def __init__(self, model: str = TRANSLATION_MODEL, api_key: str = OPENAI_API_KEY):
        self.model = model
        self._client = OpenAI(api_key=api_key) if api_key else None

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
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ],
            )
            content = response.choices[0].message.content or ""
            parsed = json.loads(content)
            items = parsed.get("instructions")
            if isinstance(items, list) and all(isinstance(x, str) for x in items):
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
        except TimeoutError as exc:
            raise RoutingUnavailable(reason="timeout", provider=self.provider.name, detail=str(exc)) from exc
        except OpenRouteServiceProviderError as exc:
            raise RoutingUnavailable(reason="provider_error", provider=self.provider.name, detail=str(exc)) from exc
        except Exception as exc:
            raise RoutingUnavailable(reason="unexpected_error", provider=self.provider.name, detail=str(exc)) from exc

        route.steps = await self.translator.translate_steps(route.steps, request.locale)
        self.cache.set(key, route)
        return route.model_copy(deep=True), False
