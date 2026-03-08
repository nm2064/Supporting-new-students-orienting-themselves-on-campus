import pytest
from pydantic import ValidationError

from maps.cache import build_route_cache_key
from maps.models import Coord, Locale, Profile, RouteRequest
from maps.rate_limit import FixedWindowRateLimiter


def test_coord_validation_rejects_out_of_range_latitude():
    with pytest.raises(ValidationError):
        Coord(lat=95, lng=0)


def test_route_request_rejects_unknown_profile_and_locale():
    with pytest.raises(ValidationError):
        RouteRequest(
            profile="transit",
            origin={"lat": 55.9, "lng": -3.3},
            destination={"lat": 55.91, "lng": -3.29},
            locale="fr",
        )


def test_build_route_cache_key_rounding_is_stable():
    origin = Coord(lat=55.91234567, lng=-3.32123456)
    destination = Coord(lat=55.90000444, lng=-3.30000444)

    key_a = build_route_cache_key(origin, destination, Profile.foot, Locale.en)
    key_b = build_route_cache_key(
        Coord(lat=55.91234568, lng=-3.32123459),
        Coord(lat=55.90000440, lng=-3.30000449),
        Profile.foot,
        Locale.en,
    )

    assert key_a == key_b


def test_fixed_window_rate_limiter_blocks_after_limit():
    limiter = FixedWindowRateLimiter()
    key = "route:127.0.0.1"

    allowed1, _ = limiter.allow(key, limit=2, window_seconds=60)
    allowed2, _ = limiter.allow(key, limit=2, window_seconds=60)
    allowed3, retry_after = limiter.allow(key, limit=2, window_seconds=60)

    assert allowed1 is True
    assert allowed2 is True
    assert allowed3 is False
    assert retry_after > 0
