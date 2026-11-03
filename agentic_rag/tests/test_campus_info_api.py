"""Seam: the campus information HTTP endpoints (/api/events, /api/services), over a real corpus file."""

import json
from datetime import datetime

from fastapi.testclient import TestClient

from agentic_rag.app import create_app
from agentic_rag.campus_info import CampusInfo, get_campus_info


def _event(title, start, end, campus, url):
    return {
        "title": f"{title} (student event)",
        "url": url,
        "category": "student-events",
        "extracted_text": f"{title} is a Heriot-Watt student event.",
        "event": {"title": title, "start": start, "end": end, "location": f"Somewhere, {campus} campus",
                  "campus": campus, "url": url},
    }


def client_for(tmp_path, pages, now=datetime(2026, 11, 3, 12, 0)):
    corpus = tmp_path / "knowledge.json"
    corpus.write_text(json.dumps({"pages": pages}), encoding="utf-8")
    app = create_app()
    app.dependency_overrides[get_campus_info] = lambda: CampusInfo(corpus, now=lambda: now)
    return TestClient(app)


def test_events_lists_upcoming_student_events_soonest_first_and_can_filter_by_campus(tmp_path):
    client = client_for(tmp_path, [
        _event("Careers fair", "2026-11-11T11:00:00", "2026-11-11T15:00:00", "edinburgh", "https://e/2"),
        _event("Yoga", "2026-11-02T09:00:00", "2026-11-02T10:00:00", "edinburgh", "https://e/past"),
        _event("Pub quiz", "2026-11-05T19:00:00", "2026-11-05T21:00:00", "edinburgh", "https://e/1"),
        _event("Desert trip", "2026-11-06T08:00:00", "2026-11-06T18:00:00", "dubai", "https://e/dubai"),
        {"title": "Library", "url": "https://lib", "extracted_text": "Not an event."},
    ])

    every = client.get("/api/events").json()
    edinburgh = client.get("/api/events", params={"campus": "edinburgh"}).json()

    assert [e["title"] for e in every] == ["Pub quiz", "Desert trip", "Careers fair"]
    assert [e["title"] for e in edinburgh] == ["Pub quiz", "Careers fair"]
    assert edinburgh[0] == {
        "title": "Pub quiz", "start": "2026-11-05T19:00:00", "end": "2026-11-05T21:00:00",
        "location": "Somewhere, edinburgh campus", "campus": "edinburgh", "url": "https://e/1",
    }


def test_services_are_real_cited_pages_and_support_campus_filter(tmp_path):
    client = client_for(tmp_path, [
        {"title": "Edinburgh wellbeing support", "url": "https://www.hw.ac.uk/support", "extracted_text": "Edinburgh wellbeing team"},
        {"title": "Dubai careers", "url": "https://www.hw.ac.uk/dubai/careers", "extracted_text": "Dubai careers team"},
        {"title": "Research news", "url": "https://www.hw.ac.uk/news", "extracted_text": "News"},
    ])
    result = client.get('/api/services?campus=edinburgh')
    assert result.status_code == 200
    assert result.json() == [{"title": "Edinburgh wellbeing support", "url": "https://www.hw.ac.uk/support", "description": "Edinburgh wellbeing team"}]


def test_missing_corpus_reports_unavailable_instead_of_empty_events(tmp_path):
    app = create_app()
    app.dependency_overrides[get_campus_info] = lambda: CampusInfo(tmp_path / 'missing.json')
    assert TestClient(app).get('/api/events').status_code == 503


def test_map_opening_hours_are_cited_and_campus_specific(tmp_path):
    client = client_for(tmp_path, [
        {"title": "Edinburgh Library opening hours", "url": "https://www.hw.ac.uk/library/edinburgh", "category": "library", "extracted_text": "Edinburgh library: open 24/7. Learning Commons: 6am to midnight."},
        {"title": "Dubai library opening hours", "url": "https://www.hw.ac.uk/library/dubai", "extracted_text": "Dubai library hours"},
    ])
    result = client.get('/api/opening-hours?campus=edinburgh')
    assert result.status_code == 200
    assert result.json()[0]['url'] == 'https://www.hw.ac.uk/library/edinburgh'
    assert len(result.json()) == 1


def test_collected_libcal_events_survive_json_storage_and_are_served_by_campus(tmp_path):
    from agentic_rag.knowledge_collection import KnowledgeCollector, StudentEventsSource
    from agentic_rag.tests.test_knowledge_collection import FakeWeb, SCOPE, LIBCAL_EVENTS_API, _event_table

    # Shape and values observed in the university feed on 2 November 2026.
    feed = _event_table(
        'Tuesday Soup Lunch', 'https://hw.ac.libcal.com/event/4567341',
        '12:30 Tuesday, November 3, 2026', '14:00 Tuesday, November 3, 2026',
        'Chaplaincy, Edinburgh Campus',
    )
    corpus = KnowledgeCollector(fetch=FakeWeb({LIBCAL_EVENTS_API: feed}), scope=SCOPE, delay_s=0).collect(
        student_events=StudentEventsSource(api_url=LIBCAL_EVENTS_API)
    )
    client = client_for(tmp_path, corpus['pages'])
    assert client.get('/api/events?campus=edinburgh').json() == [{
        'title': 'Tuesday Soup Lunch', 'start': '2026-11-03T12:30:00',
        'end': '2026-11-03T14:00:00', 'location': 'Chaplaincy, Edinburgh Campus',
        'campus': 'edinburgh', 'url': 'https://hw.ac.libcal.com/event/4567341',
    }]
    assert client.get('/api/events?campus=dubai').json() == []
