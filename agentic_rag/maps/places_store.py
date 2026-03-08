"""Campus places dataset loader and search utilities."""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

from .models import PlaceDetail, PlaceSummary


class PlacesStore:
    """In-memory store backed by versioned JSON dataset."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._places: list[PlaceDetail] = []
        self._by_id: dict[str, PlaceDetail] = {}
        self.load()

    def load(self) -> None:
        if not self.dataset_path.exists():
            self._places = []
            self._by_id = {}
            return

        with self.dataset_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raise ValueError("Places dataset must be a JSON array")

        places: list[PlaceDetail] = []
        for item in raw:
            place = PlaceDetail.model_validate(item)
            places.append(place)

        self._places = places
        self._by_id = {place.id: place for place in places}

    def search(self, query: str = "", campus: str | None = None, limit: int = 20) -> list[PlaceSummary]:
        q = query.strip().lower()
        campus_filter = (campus or "").strip().lower()

        filtered = []
        for place in self._places:
            if campus_filter and place.campus.lower() != campus_filter:
                continue
            score = self._score(place, q)
            if q and score <= 0:
                continue
            filtered.append((score, place))

        filtered.sort(key=lambda item: item[0], reverse=True)
        return [
            PlaceSummary(
                id=place.id,
                name=place.name,
                lat=place.lat,
                lng=place.lng,
                type=place.type,
                campus=place.campus,
            )
            for _, place in filtered[:limit]
        ]

    def get(self, place_id: str) -> PlaceDetail | None:
        return self._by_id.get(place_id)

    def _score(self, place: PlaceDetail, query: str) -> float:
        if not query:
            return 1.0

        haystack_parts = [place.name.lower(), place.type.lower(), place.campus.lower()] + [
            tag.lower() for tag in place.tags
        ]
        haystack = " | ".join(haystack_parts)

        if query in place.name.lower():
            return 1.0
        if query in haystack:
            return 0.85
        return SequenceMatcher(None, query, place.name.lower()).ratio()
