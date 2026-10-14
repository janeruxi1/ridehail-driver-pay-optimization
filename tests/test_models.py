"""Estimators recover known effects; the offer model and optimizer behave as specified."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from ridepay import accept, policy, simulate


def confounded(n=20000, theta=0.3, seed=0):
    """Y = theta*T + g(X) + e, with T depending on X: naive OLS is biased, DML is not."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (n, 2))
    t = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + rng.normal(0, 0.5, n)
    y = theta * t - 1.0 * (np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2) + rng.normal(0, 0.5, n)
    return y, t, pd.DataFrame(x, columns=["x0", "x1"])


def test_naive_is_biased_and_dml_recovers_the_effect():
    y, t, X = confounded()
    naive = accept.naive_slope(y, t)
    dml, _ = accept.dml_plr(y, t, X, lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3), n_folds=3)
    assert naive.ci[1] < 0.3                       # badly biased downward
    assert dml.ci[0] < 0.3 < dml.ci[1]


def test_iv_recovers_effect_from_random_shock():
    rng = np.random.default_rng(1)
    n = 20000
    conf = rng.normal(0, 1, n)
    s = rng.uniform(-0.15, 0.15, n)
    t = 0.5 * conf + s
    y = 0.4 * t - 0.8 * conf + rng.normal(0, 0.3, n)
    est = accept.iv_wald(y, t, s)
    assert est.ci[0] < 0.4 < est.ci[1]


def test_true_weighted_average():
    assert accept.true_weighted([1, 3], [1, 1]) == 2
    assert accept.true_weighted([1, 3], [3, 1]) == 1.5


def test_offer_model_limits():
    o = policy.outcomes(np.array([1 - 1e-9]), fare=20.0, pay=15.0, minutes=10.0)
    assert o["complete"][0] == pytest.approx(1)
    assert o["extra_wait_sec"][0] == pytest.approx(0, abs=1e-3)
    assert o["margin"][0] == pytest.approx(5.0)
    low = policy.outcomes(np.array([0.2]), 20.0, 15.0, 10.0)
    high = policy.outcomes(np.array([0.8]), 20.0, 15.0, 10.0)
    assert high["complete"][0] > low["complete"][0]
    assert high["extra_wait_sec"][0] < low["extra_wait_sec"][0]


def test_acceptance_curve_increases_with_pay_and_calibration_hits_target():
    p = simulate.acceptance_prob(0.5, 4.0, np.zeros(3), np.log([1.0, 1.1, 1.3]))
    assert np.all(np.diff(p) > 0)
    rng = np.random.default_rng(0)
    trips = pd.DataFrame({"cluster_name": ["A"] * 500 + ["B"] * 500, "band": ["midday"] * 1000,
                          "offer_sq": 10 * (1 + rng.uniform(0, 0.3, 1000)), "floor": 10.0})
    feats = pd.DataFrame({k: rng.uniform(0, 1, 1000) for k in simulate.DIFFICULTY})
    truth = simulate.make_truth(trips, feats, {"A": 4.0, "B": 6.0})
    for (cl, band), target in truth.target.items():
        idx = (trips["cluster_name"] == cl).to_numpy()
        got = simulate.acceptance_prob(truth.a[(cl, band)], truth.beta[(cl, band)],
                                       simulate.observed_difficulty(feats[idx]),
                                       np.log(trips.loc[idx, "offer_sq"] / trips.loc[idx, "floor"])).mean()
        assert got == pytest.approx(target, abs=1e-3)
    assert truth.target[("B", "midday")] < truth.target[("A", "midday")]     # longer waits, lower acceptance


def test_pay_test_never_goes_below_the_floor():
    rng = np.random.default_rng(0)
    n = 4000
    trips = pd.DataFrame({"cluster_name": "A", "band": "midday", "month": "2024-09", "floor": 10.0,
                          "offer_sq": 10 * (1 + rng.uniform(0, 0.2, n))})
    feats = pd.DataFrame({k: np.zeros(n) for k in simulate.DIFFICULTY})
    truth = simulate.make_truth(trips, feats, {"A": 4.5})
    sim = simulate.simulate(trips, feats, truth, ["2024-09"])
    assert (sim["offer"] >= sim["floor"] - 1e-9).all()
    assert 0.05 < sim["in_test"].mean() < 0.11


def small_table():
    """Two cells; cell 0 responds strongly to pay, cell 1 barely."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"cell": np.repeat(["A|midday|q1", "A|midday|q5"], n), "cluster_name": "A", "band": "midday",
                       "floor": 10.0, "fare": 16.0, "minutes": 15.0})
    df["offer_sq"] = df["floor"] * 1.1
    sens = np.where(df["cell"].str.endswith("q1"), 8.0, 0.5)

    def fn(d, lr, _):
        return 1 / (1 + np.exp(-(0.3 + sens * lr)))

    del rng
    return policy.cell_table(df, fn, [None])


def test_optimizer_respects_guardrails_and_beats_today():
    t = small_table()
    choice, _ = policy.optimize(t)
    checks = policy.guardrails(t, choice)
    assert all(checks.values())
    assert policy.summarize(t, choice)["margin_per_request"] >= policy.summarize(t, "sq")["margin_per_request"] - 1e-9
    prem = policy.PREMIUMS[choice]
    assert prem.max() <= 0.30 + 1e-9 and prem.min() >= 0
    assert prem[0] > prem[1]          # the pay-sensitive cell gets the bigger premium
