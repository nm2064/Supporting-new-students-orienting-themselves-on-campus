"""Route provider interface."""
from __future__ import annotations

from typing import Protocol

from ..models import Coord, Profile, RouteResponse


class RouteProvider(Protocol):
    """Provider adapter contract for routing backends."""

    name: str

    async def get_route(
        self,
        origin: Coord,
        destination: Coord,
        profile: Profile,
        locale: str,
    ) -> RouteResponse:
        ...
