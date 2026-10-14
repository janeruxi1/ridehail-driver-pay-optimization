"""The simulated layer: drivers accepting or declining real trips (phase 5), with a planted truth.

Everything about each trip is real (where, when, miles, minutes, rider fare, and the pay the platform
actually offered). The TLC records don't say whether the *first* driver offered the trip accepted it,
so acceptance is simulated, from a model whose parameters we choose. Knowing the true parameters is
the point: it lets us check whether an estimation method recovers them.

Acceptance of an offer, for trip i in segment g (zone cluster x hour band):

    logit P(accept) = a_g - difficulty_i + beta_g * log(offer_i / floor_i)

    difficulty_i = 1.2 * short_i + 0.7 * deadhead_i + 0.35 * tight_i + 0.3 * night_i + u_i

* short: how far under 2 miles the trip is (drivers dislike short trips: little pay for the pickup)
* deadhead: log(drop-offs per pickup) of the destination zone (drivers end up stranded there)
* tight: how long waits were in the pickup zone in the hour before (a busy, short-supplied area)
* night: request between midnight and 6am
* u: what the analyst can't see (weather on the street, a closed road), N(0, 0.5)

**Why the history misleads:** the real premium the platform paid is highest on short trips and at
night, exactly the trips drivers like least. So high pay goes with low acceptance in the data.

**The pay test:** during four months, 8% of offers were moved by a random factor s ~ U(-15%, +15%),
never below the floor. That randomization is unrelated to difficulty, so it identifies the effect.

``hidden_confounding`` > 0 makes u correlated with the real premium (operations reacting to something
the analyst can't observe), the case where adjusting for observed features is not enough.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from . import config

DIFFICULTY = {"short": 1.2, "deadhead": 0.7, "tight": 0.35, "night": 0.3}
U_SD = 0.5
TEST_SHARE = 0.08
TIGHT_CENTER_MIN, TIGHT_SCALE_MIN = 4.4, 1.5   # typical zone-hour mean wait and its spread (phase 1); fixed so
                                               # the feature means the same thing in every month
TEST_RANGE = 0.15


@dataclass
class Truth:
    """Planted parameters for every segment g = (cluster_name, band)."""
    a: dict = field(default_factory=dict)          # intercepts, calibrated to target acceptance
    beta: dict = field(default_factory=dict)       # pay sensitivity (log-odds per unit log pay)
    target: dict = field(default_factory=dict)     # first-offer acceptance at status-quo pay

    def to_frame(self) -> pd.DataFrame:
        idx = pd.MultiIndex.from_tuples(list(self.beta), names=["cluster_name", "band"])
        return pd.DataFrame({"a": [self.a[k] for k in self.beta], "beta": [self.beta[k] for k in self.beta],
                             "target_acceptance": [self.target[k] for k in self.beta]}, index=idx)


def difficulty_features(trips: pd.DataFrame, zone_info: pd.DataFrame) -> pd.DataFrame:
    """Observed difficulty features from real trip and zone data.

    ``trips`` needs trip_miles, DOLocationID, band and prev_wait_min (pickup zone, previous hour);
    ``zone_info`` is indexed by zone with arrivals_per_pickup.
    """
    out = pd.DataFrame(index=trips.index)
    out["short"] = np.clip((2.0 - trips["trip_miles"]) / 2.0, 0, 1)
    app = trips["DOLocationID"].map(zone_info["arrivals_per_pickup"])
    app = app.where(trips["DOLocationID"] != config.OUTSIDE_NYC_ZONE, np.e)      # outside NYC: drive back empty
    out["deadhead"] = np.clip(np.log(app.fillna(1.0)), -0.5, 1.5)
    w = trips["prev_wait_min"]
    out["tight"] = np.clip((w - TIGHT_CENTER_MIN) / TIGHT_SCALE_MIN, -2, 3).fillna(0)
    out["night"] = (trips["band"] == "night").astype(float)
    return out


def observed_difficulty(feats: pd.DataFrame) -> np.ndarray:
    return sum(coef * feats[name].to_numpy() for name, coef in DIFFICULTY.items())


def make_truth(trips: pd.DataFrame, feats: pd.DataFrame, cluster_wait: dict, seed: int = 11) -> Truth:
    """Plant segment parameters and calibrate intercepts so status-quo acceptance hits a target.

    Target first-offer acceptance falls with the cluster's real mean wait (scarcer drivers, pickier
    acceptance) and is lower at night. Pay sensitivity beta is around 4 (a 10% higher offer raises
    the log-odds by about 0.38), higher where waits are long and at night.
    """
    rng = np.random.default_rng(seed)
    truth = Truth()
    waits = pd.Series(cluster_wait)
    zw = (waits - waits.mean()) / (waits.std() if waits.std() > 0 else 1)
    log_ratio = np.log(trips["offer_sq"] / trips["floor"]).to_numpy()
    d_obs = observed_difficulty(feats)
    band_shift = {"night": 0.25, "am_peak": 0.05, "midday": -0.1, "pm_peak": 0.05, "evening": 0.0}
    for (cl, band), idx in trips.groupby(["cluster_name", "band"], observed=True).indices.items():
        target = float(np.clip(0.75 - 0.05 * (cluster_wait.get(cl, 4.5) - 4.5) - (0.05 if band == "night" else 0),
                               0.5, 0.85))
        beta = float(4.0 * np.exp(0.2 * zw.get(cl, 0.0) + band_shift[band] + rng.normal(0, 0.1)))
        d_g, lr_g = d_obs[idx], log_ratio[idx]

        def gap(a, beta=beta, d_g=d_g, lr_g=lr_g, target=target):
            return acceptance_prob(a, beta, d_g, lr_g).mean() - target

        a = brentq(gap, -10, 15)
        truth.a[(cl, band)], truth.beta[(cl, band)], truth.target[(cl, band)] = float(a), beta, target
    return truth


def simulate(trips: pd.DataFrame, feats: pd.DataFrame, truth: Truth, test_months: list[str],
             seed: int = 5, hidden_confounding: float = 0.0) -> pd.DataFrame:
    """Simulate the pay test and one accept/decline decision per trip (the first offer).

    Returns trips with: in_test, s (random shock, 0 outside the test), offer, log_ratio, p_true, accept,
    beta_true, tau_true (true marginal effect of log pay on acceptance probability at this offer) and u.
    """
    rng = np.random.default_rng(seed)
    out = trips.copy()
    n = len(out)
    in_window = out["month"].isin(test_months).to_numpy()
    out["in_test"] = in_window & (rng.random(n) < TEST_SHARE)
    out["s"] = np.where(out["in_test"], rng.uniform(-TEST_RANGE, TEST_RANGE, n), 0.0)
    out["offer"] = np.maximum(out["floor"], out["offer_sq"] * (1 + out["s"]))   # never below the floor
    out["log_ratio"] = np.log(out["offer"] / out["floor"])
    u = rng.normal(0, U_SD, n)
    if hidden_confounding:
        prem = np.log(out["offer_sq"] / out["floor"]).to_numpy()
        z = (prem - prem.mean()) / prem.std()
        u = np.sqrt(1 - hidden_confounding**2) * u + hidden_confounding * U_SD * z   # still sd 0.5
    seg = list(zip(out["cluster_name"], out["band"]))
    a = np.array([truth.a[k] for k in seg])
    beta = np.array([truth.beta[k] for k in seg])
    eta = a - observed_difficulty(feats) - u + beta * out["log_ratio"].to_numpy()
    p = expit(eta)
    out["p_true"] = p
    out["accept"] = (rng.random(n) < p).astype(int)
    out["beta_true"] = beta
    out["tau_true"] = beta * p * (1 - p)
    out["u"] = u                          # kept so the true effect of any pay change can be computed
    return out


def p_at(sim: pd.DataFrame, feats: pd.DataFrame, truth: Truth, log_ratio) -> np.ndarray:
    """True acceptance probability of each simulated trip at a different log pay ratio, holding its
    unobserved difficulty u fixed: the counterfactual a causal estimate is about."""
    seg = list(zip(sim["cluster_name"], sim["band"]))
    a = np.array([truth.a[k] for k in seg])
    beta = np.array([truth.beta[k] for k in seg])
    return expit(a - observed_difficulty(feats) - sim["u"].to_numpy() + beta * np.asarray(log_ratio))


def acceptance_prob(a, beta, d_obs, log_ratio, u_sd: float = U_SD, n_quad: int = 20) -> np.ndarray:
    """P(accept) averaged over the unobserved u (Gauss-Hermite quadrature): the population curve."""
    x, w = np.polynomial.hermite_e.hermegauss(n_quad)
    w = w / w.sum()
    eta = np.asarray(a - d_obs + beta * log_ratio, float)
    return (expit(eta[..., None] - u_sd * x) * w).sum(axis=-1)
