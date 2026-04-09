import asyncio
from pathlib import Path

from agentic_rag.maps.cache import TTLCache
from agentic_rag.maps.models import (
    Coord,
    Locale,
    Profile,
    RouteGeometry,
    RouteRequest,
    RouteResponse,
    RouteStep,
)
from agentic_rag.maps.places_store import PlacesStore
from agentic_rag.maps.service import MapsService


class FakeProvider:
    name = "fake-provider"

    def __init__(self):
        self.calls = 0

    async def get_route(self, origin, destination, profile, locale):
        self.calls += 1
        return RouteResponse(
            distance_m=1000,
            duration_s=700,
            geometry=RouteGeometry(type="LineString", coordinates=[[origin.lng, origin.lat], [destination.lng, destination.lat]]),
            steps=[RouteStep(instruction="Go straight", distance_m=1000, duration_s=700)],
            provider=self.name,
        )


class SpyTranslator:
    def __init__(self):
        self.calls = 0

    async def translate_steps(self, steps, locale):
        self.calls += 1
        return [
            RouteStep(
                instruction=f"[{locale.value}] {steps[0].instruction}",
                distance_m=steps[0].distance_m,
                duration_s=steps[0].duration_s,
            )
        ]


FIXTURE_PATH = Path(__file__).resolve().parent / "places_fixture.json"


def test_maps_service_cache_and_translation():
    dataset = FIXTURE_PATH
    provider = FakeProvider()
    translator = SpyTranslator()
    service = MapsService(
        places_store=PlacesStore(str(dataset)),
        provider=provider,
        cache=TTLCache(ttl_s=300, max_items=100),
        timeout_s=1,
        translator=translator,
    )

    request = RouteRequest(
        profile=Profile.foot,
        origin=Coord(lat=55.9, lng=-3.3),
        destination=Coord(lat=55.91, lng=-3.29),
        locale=Locale.ar,
    )

    first, first_cache_hit = asyncio.run(service.get_route(request))
    second, second_cache_hit = asyncio.run(service.get_route(request))

    assert first_cache_hit is False
    assert second_cache_hit is True
    assert provider.calls == 1
    assert translator.calls == 1
    assert first.steps[0].instruction.startswith("[ar]")
    assert second.steps[0].instruction.startswith("[ar]")
