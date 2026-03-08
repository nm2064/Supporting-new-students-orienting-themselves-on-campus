"""Pydantic models for maps endpoints."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Profile(str, Enum):
    foot = "foot"
    cycling = "cycling"
    driving = "driving"


class Locale(str, Enum):
    en = "en"
    zh_hans = "zh-Hans"
    hi = "hi"
    ar = "ar"


class Coord(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class PlaceSummary(BaseModel):
    id: str
    name: str
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    type: str
    campus: str


class PlaceDetail(PlaceSummary):
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    profile: Profile = Profile.foot
    origin: Coord
    destination: Coord
    locale: Locale = Locale.en


class RouteGeometry(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[list[float]] = Field(default_factory=list)


class RouteStep(BaseModel):
    instruction: str
    distance_m: float = Field(ge=0)
    duration_s: float = Field(ge=0)


class RouteResponse(BaseModel):
    distance_m: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    geometry: RouteGeometry
    steps: list[RouteStep] = Field(default_factory=list)
    provider: str
