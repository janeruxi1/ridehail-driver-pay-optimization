"""Pay floor rates by date, and the forecast features' no-leakage guarantee."""
import numpy as np
import pandas as pd
import pytest

from ridepay import forecast
from ridepay.pay import floor_pay


def test_floor_uses_rates_in_force_on_the_date():
    # 5 miles, 20 minutes, inside NYC, after 1 March 2024: 1.36 * 5 + 0.583 * 20
    assert floor_pay([5], [20], ["2024-07-01"])[0] == pytest.approx(1.36 * 5 + 0.583 * 20)
    before, after = floor_pay([5, 5], [20, 20], ["2025-02-28", "2025-03-01"])
    assert after == pytest.approx(before * 1.0392)


def test_floor_out_of_town_rate_applies_to_drop_offs_outside_nyc():
    inside, outside = floor_pay([10, 10], [30, 30], ["2024-07-01"] * 2, do_zone=[100, 265])
    assert outside == pytest.approx(1.576 * 10 + 0.675 * 30)
    assert outside > inside


def make_panel(weeks=60, zones=3, seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-01-01", periods=weeks * 168, freq="h")
    rows = [{"zone": z, "hour": h, "trips": int(rng.poisson(5 + z))} for z in range(1, zones + 1) for h in hours]
    return forecast.make_panel(pd.DataFrame(rows))


def test_features_do_not_use_data_after_the_origin():
    p = make_panel()
    origin = p.hours[0] + pd.Timedelta(weeks=56)
    f1 = forecast.features_at_origin(p, origin, holidays_set=set())
    o = p.col(origin)
    p.trips[:, o:] = 999.0                     # change every value at or after the origin
    f2 = forecast.features_at_origin(p, origin, holidays_set=set())
    cols = [c for c in f1.columns if c != "y"]
    pd.testing.assert_frame_equal(f1[cols], f2[cols])
    assert (f2["y"] == 999).all()


def test_seasonal_naive_feature_is_same_hour_last_week():
    p = make_panel(weeks=10, zones=1)
    origin = p.hours[0] + pd.Timedelta(weeks=5, hours=10)     # not aligned to Monday midnight on purpose
    f = forecast.features_at_origin(p, origin, holidays_set=set(), horizon_h=200)
    o = p.col(origin)
    expected = np.r_[p.trips[0, o - 168:o], p.trips[0, o - 168:o]][:200]   # repeats last week
    np.testing.assert_allclose(f["last1"].to_numpy(), expected)


def test_wmape_and_bias():
    assert forecast.wmape([10, 10], [12, 8]) == pytest.approx(0.2)
    assert forecast.bias([10, 10], [12, 12]) == pytest.approx(0.2)
