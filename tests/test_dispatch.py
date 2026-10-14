"""Dispatch simulation on a toy city: matching rules and bookkeeping."""
import numpy as np
import pandas as pd

from ridepay import dispatch


def toy_city():
    zones = pd.DataFrame({"x_mi": [0.0, 1.0, 5.0], "y_mi": [0.0, 0.0, 0.0], "radius_mi": [0.01] * 3,
                          "in_area": [True, True, True]}, index=[1, 2, 3])
    n = 200
    rng = np.random.default_rng(0)
    req = pd.DataFrame({"t_sec": np.sort(rng.uniform(0, 4 * 3600, n)), "PULocationID": rng.choice([1, 2, 3], n),
                        "DOLocationID": rng.choice([1, 2, 3], n), "minutes": 8.0, "fare": 15.0, "floor": 10.0})
    return zones, req


def run(req, zones, accept, matching="batch", scale=1.0):
    cfg = dispatch.SimConfig(matching=matching, supply_scale=scale, seed=1)
    return dispatch.run(req, np.full(len(req), accept), np.full(len(req), 11.0), zones, np.full(24, 30.0),
                        np.full(24, 12.0), cfg)


def test_everyone_accepts_with_ample_supply():
    zones, req = toy_city()
    res = run(req, zones, accept=1.0, scale=3.0)
    assert res["completed"].mean() > 0.95
    assert np.all(res["wait_sec"][res["completed"]] >= 0)
    assert res["first_offer_accept"][res["completed"]].all()
    assert res["driver_pay_total"] == 11.0 * res["completed"].sum()


def test_low_acceptance_means_more_offers_and_cancellations():
    zones, req = toy_city()
    hi = run(req, zones, accept=0.95, scale=3.0)
    lo = run(req, zones, accept=0.3, scale=3.0)
    assert lo["offers"][lo["completed"]].mean() > hi["offers"][hi["completed"]].mean()
    assert lo["cancelled"].mean() > hi["cancelled"].mean()


def test_batch_matching_never_worse_on_total_pickup_time():
    eta = np.array([[1.0, 2.0], [1.5, 10.0]])
    g = dispatch._match(eta, "greedy")
    b = dispatch._match(eta, "batch")
    assert sum(eta[i, j] for i, j in b) <= sum(eta[i, j] for i, j in g)
    assert sum(eta[i, j] for i, j in b) == 3.5          # 0->1, 1->0
