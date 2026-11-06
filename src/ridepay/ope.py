"""Off-policy evaluation: what a pay policy would do, estimated from the randomized pay test alone.

In the pay test a share of offers is moved by a random factor ``1 + s``, ``s ~ U(-R, R)``, and never below
the floor. For a trip whose status-quo offer is ``offer_sq``, the realized log shift is
``x = log(offer / offer_sq)``. A new policy asks for a target shift ``x*``. Kernel-weighted inverse
propensity scoring (self-normalized) estimates the mean outcome at ``x*`` from the test trips whose
realized shift landed near it, weighting each by ``K_h(x - x*) / f(x)``.

This works only where the test could have produced ``x*``: inside the randomized range, away from its
edges and from the floor (where shifts pile up). Those trips are the *supported* ones; the share of
the policy's changes that fall outside is itself a finding about how the test was designed.
"""
from __future__ import annotations

import numpy as np


def shift_density(x: np.ndarray, r: float) -> np.ndarray:
    """Density of ``x = log(1 + s)`` for ``s ~ U(-r, r)``: exp(x) / (2r) on [log(1-r), log(1+r)]."""
    x = np.asarray(x, float)
    inside = (x >= np.log1p(-r)) & (x <= np.log1p(r))
    return np.where(inside, np.exp(x) / (2 * r), 0.0)


def supported(x_target: np.ndarray, floor_shift: np.ndarray, r: float, h: float) -> np.ndarray:
    """True where a kernel of half-width ``h`` around ``x_target`` sits inside the continuous part of the
    test's shift distribution: above the floor's pile-up (``floor_shift`` = log(floor / offer_sq)) and
    inside the randomized range."""
    lo = np.maximum(np.log1p(-r), floor_shift)
    return (x_target >= lo + h) & (x_target <= np.log1p(r) - h)


def kernel_ipw(y: np.ndarray, x_obs: np.ndarray, x_target: np.ndarray, floor_shift: np.ndarray, r: float,
               h: float) -> tuple[float, np.ndarray]:
    """Self-normalized kernel IPW estimate of E[y] at each trip's target shift, over supported test trips.

    Arguments are for the pay-test trips only. Returns the estimate and the per-trip weights (0 for
    trips that are unsupported or whose realized shift sits on the floor)."""
    y, x_obs, x_target = (np.asarray(a, float) for a in (y, x_obs, x_target))
    ok = supported(x_target, floor_shift, r, h) & (x_obs > np.asarray(floor_shift) + 1e-12)
    u = (x_obs - x_target) / h
    kern = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2) / h, 0.0)                  # Epanechnikov
    dens = shift_density(x_obs, r)
    w = np.where(ok & (dens > 0), kern / np.maximum(dens, 1e-12), 0.0)
    if w.sum() == 0:
        return float("nan"), w
    return float((w * y).sum() / w.sum()), w


def bootstrap_ci(fn, n: int, reps: int = 200, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of ``fn(index_array)`` over ``reps`` resamples of ``n`` rows."""
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(reps)])
    vals = vals[np.isfinite(vals)]
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
