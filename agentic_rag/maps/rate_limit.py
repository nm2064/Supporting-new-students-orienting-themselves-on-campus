"""Basic fixed-window rate limiting utilities."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from fastapi import Request


@dataclass
class _WindowState:
    window_start: float
    count: int


class FixedWindowRateLimiter:
    """In-memory per-key fixed-window limiter."""

    def __init__(self):
        self._state: dict[str, _WindowState] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_seconds: int = 60) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            entry = self._state.get(key)
            if not entry or (now - entry.window_start) >= window_seconds:
                self._state[key] = _WindowState(window_start=now, count=1)
                return True, 0

            if entry.count >= limit:
                retry_after = int(max(1, window_seconds - (now - entry.window_start)))
                return False, retry_after

            entry.count += 1
            return True, 0


def client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"
