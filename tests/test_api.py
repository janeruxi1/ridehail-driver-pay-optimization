"""The quote engine and the HTTP API, on a small hand-written model artifact."""
import json
from datetime import datetime

import pytest

from ridepay.quote import QuoteModel

ARTIFACT = {
    "model_version": "ridepay-test",
    "zone_cluster": {"61": "Core", "17": "Core", "132": "Airports"},
    "zone_arrivals_per_pickup": {"61": 1.0, "17": 1.6, "132": 0.8},
    "segments": ["Core|am_peak", "Core|night", "Airports|am_peak"],
    "alpha": [1.0, 0.5, 1.5], "beta": [4.0, 5.0, 2.0],
    "gamma": {"short": 1.2, "deadhead": 0.7, "tight": 0.35, "night": 0.3},
    "difficulty_edges": [0.1, 0.3, 0.5, 0.8],
    "premium": {"Core|am_peak|q1": 0.0, "Core|am_peak|q2": 0.05, "Core|am_peak|q3": 0.1, "Core|am_peak|q4": 0.15,
                "Core|am_peak|q5": 0.9, "Core|night|q5": 0.2},
}
TRIP = dict(pickup_zone=61, dropoff_zone=17, request_time=datetime(2025, 6, 10, 8, 15), trip_miles=2.4,
            trip_minutes=14, rider_fare=18.5)


@pytest.fixture
def model_path(tmp_path):
    p = tmp_path / "quote_model.json"
    p.write_text(json.dumps(ARTIFACT))
    return p


def test_quote_respects_floor_and_cap(model_path):
    m = QuoteModel.load(model_path)
    q = m.quote(**TRIP)
    assert q.recommended_pay >= q.floor_pay
    assert q.recommended_pay <= round(q.floor_pay * 1.30, 2) + 0.01
    # a cell whose artifact premium is 90% is still capped at 30% (C1)
    hard = m.quote(**{**TRIP, "trip_miles": 0.5, "dropoff_zone": 265, "prev_hour_wait_min": 9.0})
    assert hard.difficulty_quintile == 5 and hard.premium_pct == 30.0


def test_quote_is_deterministic_and_versioned(model_path):
    m = QuoteModel.load(model_path)
    assert m.quote(**TRIP) == m.quote(**TRIP)
    assert m.quote(**TRIP).model_version == "ridepay-test"


def test_unknown_segment_falls_back_to_floor(model_path):
    q = QuoteModel.load(model_path).quote(**{**TRIP, "pickup_zone": 5})
    assert q.rule.startswith("default") and q.recommended_pay == q.floor_pay and q.accept_probability is None


def test_api_endpoints(model_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ridepay import api

    monkeypatch.setenv("RIDEPAY_MODEL", str(model_path))
    api.get_model.cache_clear()
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok", "model_version": "ridepay-test"}
    body = {**TRIP, "request_time": TRIP["request_time"].isoformat()}
    r = client.post("/quote", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["model_version"] == "ridepay-test" and out["recommended_pay"] >= out["floor_pay"]
    assert client.post("/quote", json={**body, "pickup_zone": 999}).status_code == 422     # validation
    assert client.post("/quote", json={**body, "trip_miles": -1}).status_code == 422
    api.get_model.cache_clear()
