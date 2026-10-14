"""Helpers behind the Streamlit demo."""
import json
from datetime import datetime

import pytest

from ridepay import demo
from ridepay.quote import QuoteModel
from test_api import ARTIFACT, TRIP


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "quote_model.json"
    p.write_text(json.dumps(ARTIFACT))
    return QuoteModel.load(p)


def test_offer_curve_is_monotone_in_acceptance_and_capped(model):
    c = demo.offer_curve(model, **TRIP)
    assert c["premium_pct"].max() <= 30 + 1e-9 and c["premium_pct"].min() == 0
    assert c["accept_probability"].is_monotonic_increasing
    assert (c["pay"].diff().dropna() > 0).all()


def test_offer_curve_matches_the_quote(model):
    q = model.quote(**TRIP)
    c = demo.offer_curve(model, **TRIP).set_index("premium_pct")
    assert c.loc[q.premium_pct, "accept_probability"] == pytest.approx(q.accept_probability, abs=1e-4)


def test_offer_curve_empty_for_sparse_zone(model):
    assert demo.offer_curve(model, **{**TRIP, "pickup_zone": 5}).empty


def test_switchback_mde_and_power():
    mde4 = demo.switchback_mde(0.1, 0.0, 4)
    assert mde4 == pytest.approx(0.28)
    assert demo.switchback_mde(0.1, 0.0, 16) == pytest.approx(0.14)          # 4x the weeks halves the MDE
    assert demo.switchback_power(0.1, 0.0, 4, mde4) == pytest.approx(0.8, abs=0.01)
    assert demo.switchback_power(0.1, 0.0, 4, 0.0) == pytest.approx(0.05, abs=1e-3)


def test_quote_time_is_used_for_band(model):
    assert model.quote(**{**TRIP, "request_time": datetime(2025, 6, 10, 2, 0)}).segment == "Core|night"
