import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag.maps import router as maps_router
from agentic_rag.maps.cache import TTLCache
from agentic_rag.maps.models import (
    PlaceDetail,
    RouteGeometry,
    RouteRequest,
    RouteResponse,
    RouteStep,
)
from agentic_rag.maps.places_store import PlacesStore
from agentic_rag.maps.service import MapsService, NoopInstructionTranslator, RoutingUnavailable


class FakeService:
    def __init__(self, fail_route: bool = False):
        self.fail_route = fail_route
        self.last_locale = None
        self._place = PlaceDetail(
            id="edin-main-library",
            name="Main Library",
            lat=55.90985,
            lng=-3.32015,
            campus="edinburgh",
            type="library",
            tags=["study"],
            metadata={"building_code": "LIB"},
        )

    def search_places(self, query: str, campus: str | None = None, limit: int = 20):
        if query and query.lower() not in self._place.name.lower():
            return []
        if campus and campus.lower() != self._place.campus:
            return []
        return [self._place]

    def get_place(self, place_id: str):
        if place_id == self._place.id:
            return self._place
        return None

    async def get_route(self, request: RouteRequest):
        if self.fail_route:
            raise RoutingUnavailable(reason="provider_error", provider="fake")

        self.last_locale = request.locale.value
        instruction = "انعطف يمينًا" if request.locale.value == "ar" else "Turn right"
        return (
            RouteResponse(
                distance_m=1200,
                duration_s=900,
                geometry=RouteGeometry(
                    type="LineString",
                    coordinates=[
                        [request.origin.lng, request.origin.lat],
                        [request.destination.lng, request.destination.lat],
                    ],
                ),
                steps=[RouteStep(instruction=instruction, distance_m=1200, duration_s=900)],
                provider="fake",
            ),
            False,
        )


class CountingProvider:
    name = "counting"

    def __init__(self):
        self.calls = 0

    async def get_route(self, origin, destination, profile, locale):
        self.calls += 1
        return RouteResponse(
            distance_m=500,
            duration_s=300,
            geometry=RouteGeometry(
                type="LineString",
                coordinates=[[origin.lng, origin.lat], [destination.lng, destination.lat]],
            ),
            steps=[RouteStep(instruction="Continue", distance_m=500, duration_s=300)],
            provider=self.name,
        )


def _client_for_service(service):
    app = FastAPI()
    app.include_router(maps_router.router)
    app.dependency_overrides[maps_router.get_maps_service] = lambda: service
    return TestClient(app)


FIXTURE_PATH = Path(__file__).resolve().parent / "places_fixture.json"


def test_route_success_returns_canonical_schema():
    client = _client_for_service(FakeService())

    response = client.post(
        "/api/route",
        json={
            "profile": "foot",
            "origin": {"lat": 55.9, "lng": -3.3},
            "destination": {"lat": 55.91, "lng": -3.29},
            "locale": "en",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "fake"
    assert payload["geometry"]["type"] == "LineString"
    assert payload["steps"][0]["instruction"] == "Turn right"


def test_route_failure_returns_graceful_503():
    client = _client_for_service(FakeService(fail_route=True))

    response = client.post(
        "/api/route",
        json={
            "profile": "foot",
            "origin": {"lat": 55.9, "lng": -3.3},
            "destination": {"lat": 55.91, "lng": -3.29},
            "locale": "en",
        },
    )

    assert response.status_code == 503
    assert "Routing temporarily unavailable" in response.json()["detail"]


def test_places_search_and_lookup():
    client = _client_for_service(FakeService())

    search_response = client.get("/api/places", params={"query": "library", "campus": "edinburgh"})
    assert search_response.status_code == 200
    assert len(search_response.json()) == 1

    detail_response = client.get("/api/places/edin-main-library")
    assert detail_response.status_code == 200
    assert detail_response.json()["name"] == "Main Library"

    missing_response = client.get("/api/places/unknown")
    assert missing_response.status_code == 404


def test_route_cache_hit_on_second_identical_call():
    dataset_path = FIXTURE_PATH
    provider = CountingProvider()
    real_service = MapsService(
        places_store=PlacesStore(str(dataset_path)),
        provider=provider,
        cache=TTLCache(ttl_s=300, max_items=100),
        timeout_s=1,
        translator=NoopInstructionTranslator(),
    )

    client = _client_for_service(real_service)
    payload = {
        "profile": "foot",
        "origin": {"lat": 55.9, "lng": -3.3},
        "destination": {"lat": 55.91, "lng": -3.29},
        "locale": "en",
    }

    r1 = client.post("/api/route", json=payload)
    r2 = client.post("/api/route", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert provider.calls == 1


def test_route_arabic_locale_response():
    fake_service = FakeService()
    client = _client_for_service(fake_service)

    response = client.post(
        "/api/route",
        json={
            "profile": "foot",
            "origin": {"lat": 55.9, "lng": -3.3},
            "destination": {"lat": 55.91, "lng": -3.29},
            "locale": "ar",
        },
    )

    assert response.status_code == 200
    assert response.json()["steps"][0]["instruction"] == "انعطف يمينًا"
