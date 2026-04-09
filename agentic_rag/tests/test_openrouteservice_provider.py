from agentic_rag.maps.providers.openrouteservice_provider import OpenRouteServiceProvider


def test_parse_geojson_route_returns_canonical_schema():
    ors_payload = {
        "features": [
            {
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-3.32, 55.91], [-3.31, 55.905]],
                },
                "properties": {
                    "summary": {"distance": 1200.5, "duration": 900.2},
                    "segments": [
                        {
                            "steps": [
                                {
                                    "instruction": "Head east",
                                    "distance": 300.0,
                                    "duration": 240.0,
                                },
                                {
                                    "instruction": "Turn right",
                                    "distance": 900.5,
                                    "duration": 660.2,
                                },
                            ]
                        }
                    ],
                },
            }
        ]
    }

    parsed = OpenRouteServiceProvider.parse_geojson_route(ors_payload)

    assert parsed.provider == "openrouteservice"
    assert parsed.distance_m == 1200.5
    assert parsed.duration_s == 900.2
    assert parsed.geometry.type == "LineString"
    assert parsed.geometry.coordinates[0] == [-3.32, 55.91]
    assert len(parsed.steps) == 2
    assert parsed.steps[1].instruction == "Turn right"
