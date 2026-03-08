"""In-memory TTL cache for route responses."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

from .models import Coord, Locale, Profile

T = TypeVar("T")


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Thread-safe in-memory TTL cache."""

    def __init__(self, ttl_s: int = 300, max_items: int = 1000):
        self._ttl_s = ttl_s
        self._max_items = max_items
        self._items: dict[str, _CacheEntry[T]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._items.get(key)
            if not entry:
                return None
            if entry.expires_at <= now:
                self._items.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: T, ttl_s: int | None = None) -> None:
        now = time.monotonic()
        ttl = ttl_s if ttl_s is not None else self._ttl_s
        with self._lock:
            self._prune_expired(now)
            if len(self._items) >= self._max_items:
                oldest_key = min(self._items, key=lambda k: self._items[k].expires_at)
                self._items.pop(oldest_key, None)
            self._items[key] = _CacheEntry(value=value, expires_at=now + ttl)

    def _prune_expired(self, now: float) -> None:
        expired = [key for key, entry in self._items.items() if entry.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)


def build_route_cache_key(
    origin: Coord,
    destination: Coord,
    profile: Profile | str,
    locale: Locale | str,
    precision: int = 5,
) -> str:
    """Create stable cache key from rounded coordinates and route options."""
    profile_value = profile.value if isinstance(profile, Profile) else str(profile)
    locale_value = locale.value if isinstance(locale, Locale) else str(locale)
    return ":".join(
        [
            f"{round(origin.lat, precision):.{precision}f}",
            f"{round(origin.lng, precision):.{precision}f}",
            f"{round(destination.lat, precision):.{precision}f}",
            f"{round(destination.lng, precision):.{precision}f}",
            profile_value,
            locale_value,
        ]
    )
