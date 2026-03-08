"""FastAPI router for maps places and routing endpoints."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

import config as app_config

from .cache import TTLCache
from .models import PlaceDetail, PlaceSummary, RouteRequest, RouteResponse
from .places_store import PlacesStore
from .providers.openrouteservice_provider import OpenRouteServiceProvider
from .rate_limit import FixedWindowRateLimiter, client_ip
from .service import MapsService, NoopInstructionTranslator, OpenAIInstructionTranslator, RoutingUnavailable

logger = logging.getLogger("maps")
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api", tags=["maps"])

_maps_service: MapsService | None = None
_rate_limiter = FixedWindowRateLimiter()


def _cfg(name: str, default):
    return getattr(app_config, name, default)


def _log_route(
    *,
    ip: str,
    profile: str,
    locale: str,
    provider: str,
    latency_ms: int,
    cache_hit: bool,
    error_code: str | None,
) -> None:
    payload = {
        "event": "route_request",
        "ts": datetime.now(timezone.utc).isoformat(),
        "ip": ip,
        "profile": profile,
        "locale": locale,
        "provider": provider,
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "error_code": error_code,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def get_maps_service() -> MapsService:
    global _maps_service
    if _maps_service is not None:
        return _maps_service

    places_store = PlacesStore(_cfg("MAPS_DATA_FILE", "agentic_rag/data/campus_places.json"))
    provider = OpenRouteServiceProvider(
        api_key=_cfg("ORS_API_KEY", ""),
        base_url=_cfg("ORS_BASE_URL", "https://api.openrouteservice.org"),
    )
    translator = (
        OpenAIInstructionTranslator()
        if _cfg("MAP_ENABLE_TRANSLATION_FALLBACK", True)
        else NoopInstructionTranslator()
    )
    cache = TTLCache[RouteResponse](
        ttl_s=int(_cfg("MAP_CACHE_TTL_S", 300)),
        max_items=int(_cfg("MAP_CACHE_MAX_ITEMS", 1000)),
    )
    _maps_service = MapsService(
        places_store=places_store,
        provider=provider,
        cache=cache,
        timeout_s=float(_cfg("MAP_ROUTE_TIMEOUT_S", 8)),
        translator=translator,
    )
    return _maps_service


def _enforce_rate_limit(request: Request, key: str, limit: int) -> str:
    ip = client_ip(request)
    allowed, retry_after = _rate_limiter.allow(f"{key}:{ip}", limit=limit, window_seconds=60)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    return ip


@router.get("/places", response_model=list[PlaceSummary])
async def search_places(
    request: Request,
    query: str = Query(default=""),
    campus: str | None = Query(default=None),
    service: MapsService = Depends(get_maps_service),
):
    _enforce_rate_limit(
        request,
        key="places",
        limit=int(_cfg("MAP_RATE_LIMIT_PLACES_PER_MIN", 60)),
    )
    return service.search_places(query=query, campus=campus)


@router.get("/places/{place_id}", response_model=PlaceDetail)
async def get_place(
    place_id: str,
    request: Request,
    service: MapsService = Depends(get_maps_service),
):
    _enforce_rate_limit(
        request,
        key="places",
        limit=int(_cfg("MAP_RATE_LIMIT_PLACES_PER_MIN", 60)),
    )
    place = service.get_place(place_id)
    if place is None:
        raise HTTPException(status_code=404, detail="Place not found")
    return place


@router.post("/route", response_model=RouteResponse)
async def get_route(
    route_request: RouteRequest,
    request: Request,
    service: MapsService = Depends(get_maps_service),
):
    ip = _enforce_rate_limit(
        request,
        key="route",
        limit=int(_cfg("MAP_RATE_LIMIT_ROUTE_PER_MIN", 30)),
    )
    started = time.perf_counter()

    try:
        response, cache_hit = await service.get_route(route_request)
        _log_route(
            ip=ip,
            profile=route_request.profile.value,
            locale=route_request.locale.value,
            provider=response.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cache_hit=cache_hit,
            error_code=None,
        )
        return response
    except RoutingUnavailable as exc:
        _log_route(
            ip=ip,
            profile=route_request.profile.value,
            locale=route_request.locale.value,
            provider=exc.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cache_hit=False,
            error_code=exc.reason,
        )
        detail = (
            "Routing temporarily unavailable. "
            "You can still view the destination on the map."
        )
        if exc.reason == "provider_error":
            provider_detail = (exc.detail or "").lower()
            if "missing ors_api_key" in provider_detail:
                detail = (
                    "Routing is unavailable: OpenRouteService key is missing or still set to a placeholder. "
                    "Set ORS_API_KEY in agentic_rag/.env, then restart the backend."
                )
            elif "401" in provider_detail or "unauthorized" in provider_detail or "forbidden" in provider_detail:
                detail = (
                    "Routing is unavailable: OpenRouteService rejected the API key (401/403). "
                    "Check ORS_API_KEY in agentic_rag/.env and restart the backend."
                )
        raise HTTPException(status_code=503, detail=detail) from exc
