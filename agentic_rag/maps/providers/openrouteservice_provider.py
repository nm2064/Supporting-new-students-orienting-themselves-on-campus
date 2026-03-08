"""OpenRouteService provider implementation."""
from __future__ import annotations

import asyncio
from typing import Any

import requests

from ..models import Coord, Profile, RouteGeometry, RouteResponse, RouteStep
from .base import RouteProvider


class OpenRouteServiceProviderError(Exception):
    """Raised when OpenRouteService returns an error payload."""


class OpenRouteServiceProvider(RouteProvider):
    name = "openrouteservice"

    _PROFILE_MAP = {
        Profile.foot: "foot-walking",
        Profile.cycling: "cycling-regular",
        Profile.driving: "driving-car",
    }

    _LOCALE_MAP = {
        "en": "en",
        "zh-Hans": "zh-cn",
        "hi": "en",
        "ar": "en",
    }

    def __init__(self, api_key: str, base_url: str = "https://api.openrouteservice.org"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _has_real_api_key(self) -> bool:
        key = (self.api_key or "").strip()
        if not key:
            return False
        placeholder_fragments = (
            "your-openrouteservice-key",
            "replace-with",
            "changeme",
            "example",
        )
        lowered = key.lower()
        return not any(fragment in lowered for fragment in placeholder_fragments)

    async def get_route(
        self,
        origin: Coord,
        destination: Coord,
        profile: Profile,
        locale: str,
    ) -> RouteResponse:
        if not self._has_real_api_key():
            raise OpenRouteServiceProviderError("Missing ORS_API_KEY")

        ors_profile = self._PROFILE_MAP[profile]
        ors_language = self._LOCALE_MAP.get(locale, "en")
        payload = {
            "coordinates": [[origin.lng, origin.lat], [destination.lng, destination.lat]],
            "instructions": True,
            "language": ors_language,
        }
        url = f"{self.base_url}/v2/directions/{ors_profile}/geojson"

        def _call() -> dict[str, Any]:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": self.api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            if response.status_code >= 400:
                detail = response.text[:500]
                raise OpenRouteServiceProviderError(
                    f"ORS request failed with status {response.status_code}: {detail}"
                )
            return response.json()

        data = await asyncio.to_thread(_call)
        return self.parse_geojson_route(data, provider_name=self.name)

    @staticmethod
    def parse_geojson_route(data: dict[str, Any], provider_name: str = "openrouteservice") -> RouteResponse:
        features = data.get("features") or []
        if not features:
            raise OpenRouteServiceProviderError("ORS response missing features")

        feature = features[0]
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        summary = properties.get("summary") or {}

        segments = properties.get("segments") or []
        steps: list[RouteStep] = []
        for segment in segments:
            for step in segment.get("steps") or []:
                instruction = (step.get("instruction") or "Continue").strip()
                steps.append(
                    RouteStep(
                        instruction=instruction,
                        distance_m=float(step.get("distance") or 0),
                        duration_s=float(step.get("duration") or 0),
                    )
                )

        return RouteResponse(
            distance_m=float(summary.get("distance") or 0),
            duration_s=float(summary.get("duration") or 0),
            geometry=RouteGeometry(
                type="LineString",
                coordinates=geometry.get("coordinates") or [],
            ),
            steps=steps,
            provider=provider_name,
        )
