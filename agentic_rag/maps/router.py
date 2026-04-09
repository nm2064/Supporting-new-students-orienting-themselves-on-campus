"""FastAPI router for maps places and routing endpoints."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..logging_utils import log_event
from ..settings import get_settings
from .cache import TTLCache
from .models import PlaceDetail, PlaceSummary, RouteRequest, RouteResponse
from .places_store import PlacesStore
from .providers.openrouteservice_provider import OpenRouteServiceProvider
from .rate_limit import FixedWindowRateLimiter, client_ip
from .service import (
    MapsService,
    NoopInstructionTranslator,
    OpenAIInstructionTranslator,
    RoutingUnavailable,
)

logger = logging.getLogger("agentic_rag.maps.router")

router = APIRouter(prefix="/api", tags=["maps"])

_rate_limiter = FixedWindowRateLimiter()


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
    log_event(
        logger,
        "route_request",
        ts=datetime.now(timezone.utc).isoformat(),
        ip=ip,
        profile=profile,
        locale=locale,
        provider=provider,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error_code=error_code,
    )


@lru_cache(maxsize=1)
def get_maps_service() -> MapsService:
    settings = get_settings()
    translator = (
        OpenAIInstructionTranslator()
        if settings.map_enable_translation_fallback
        else NoopInstructionTranslator()
    )
    return MapsService(
        places_store=PlacesStore(str(settings.maps_data_file)),
        provider=OpenRouteServiceProvider(
            api_key=settings.ors_api_key,
            base_url=settings.ors_base_url,
        ),
        cache=TTLCache[RouteResponse](
            ttl_s=settings.map_cache_ttl_s,
            max_items=settings.map_cache_max_items,
        ),
        timeout_s=settings.map_route_timeout_s,
        translator=translator,
    )


def reset_maps_service_cache() -> None:
    get_maps_service.cache_clear()


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
    settings = get_settings()
    _enforce_rate_limit(
        request,
        key="places",
        limit=settings.map_rate_limit_places_per_min,
    )
    return service.search_places(query=query, campus=campus)


@router.get("/places/{place_id}", response_model=PlaceDetail)
async def get_place(
    place_id: str,
    request: Request,
    service: MapsService = Depends(get_maps_service),
):
    settings = get_settings()
    _enforce_rate_limit(
        request,
        key="places",
        limit=settings.map_rate_limit_places_per_min,
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
    settings = get_settings()
    ip = _enforce_rate_limit(
        request,
        key="route",
        limit=settings.map_rate_limit_route_per_min,
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
    except RoutingUnavailable as error:
        _log_route(
            ip=ip,
            profile=route_request.profile.value,
            locale=route_request.locale.value,
            provider=error.provider,
            latency_ms=int((time.perf_counter() - started) * 1000),
            cache_hit=False,
            error_code=error.reason,
        )
        detail = (
            "Routing temporarily unavailable. "
            "You can still view the destination on the map."
        )
        if error.reason == "provider_error":
            provider_detail = (error.detail or "").lower()
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
        raise HTTPException(status_code=503, detail=detail) from error
