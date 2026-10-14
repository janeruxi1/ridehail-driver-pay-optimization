"""Estimating how acceptance responds to pay (phase 5): naive, double machine learning, the randomized
pay test, and a Bayesian hierarchical model by segment.

Notation: Y = accepted (0/1), T = log(offer / floor), X = trip and context features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import KFold


@dataclass
class Estimate:
    name: str
    theta: float
    se: float

    @property
    def ci(self) -> tuple[float, float]:
        return self.theta - 1.96 * self.se, self.theta + 1.96 * self.se


def naive_slope(y, t) -> Estimate:
    """OLS of acceptance on log pay ratio, nothing else: what a dashboard correlation would say."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    tc = t - t.mean()
    theta = (tc * (y - y.mean())).sum() / (tc**2).sum()
    resid = y - y.mean() - theta * tc
    se = np.sqrt((tc**2 * resid**2).sum()) / (tc**2).sum()         # heteroskedasticity-robust
    return Estimate("Naive (no controls)", float(theta), float(se))


def dml_plr(y, t, X: pd.DataFrame, learner_factory, n_folds: int = 5, seed: int = 0,
            categorical=None) -> tuple[Estimate, np.ndarray]:
    """Double machine learning for the partially linear model Y = theta * T + g(X) + e.

    Cross-fitting: for each fold, nuisance models E[Y|X] and E[T|X] are fitted on the other folds and
    predicted on this one, so no trip's residual uses a model trained on that trip (overfitting would
    otherwise leak into theta). theta is the slope of Y-residuals on T-residuals; the standard error
    comes from the influence function of that orthogonal moment.
    Returns the estimate and the T-residuals (needed to compute the true value of this estimand).
    """
    y, t = np.asarray(y, float), np.asarray(t, float)
    ry, rt = np.zeros_like(y), np.zeros_like(t)
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        for target, out in ((y, ry), (t, rt)):
            model = learner_factory()
            kw = {"categorical_feature": categorical} if categorical else {}
            model.fit(X.iloc[train], target[train], **kw)
            out[test] = target[test] - model.predict(X.iloc[test])
    theta = (rt * ry).sum() / (rt**2).sum()
    psi = (ry - theta * rt) * rt
    se = np.sqrt((psi**2).mean() / (rt**2).mean() ** 2 / len(y))
    return Estimate("Double ML", float(theta), float(se)), rt


def iv_wald(y, t, s) -> Estimate:
    """Instrumental-variable estimate from the randomized pay test: cov(Y, s) / cov(T, s).

    s is the random shock; it moves T and affects Y only through T, and it is independent of trip
    difficulty by construction. No controls are needed for consistency.
    """
    y, t, s = (np.asarray(v, float) for v in (y, t, s))
    sc = s - s.mean()
    theta = (sc * (y - y.mean())).sum() / (sc * (t - t.mean())).sum()
    resid = (y - y.mean()) - theta * (t - t.mean())
    se = np.sqrt((sc**2 * resid**2).sum()) / abs((sc * (t - t.mean())).sum())
    return Estimate("Randomized pay test (IV)", float(theta), float(se))


def true_weighted(tau, weights) -> float:
    """The true value of an estimand that averages marginal effects tau with the given weights."""
    w = np.asarray(weights, float)
    return float((np.asarray(tau) * w).sum() / w.sum())


def true_dml(rt, p_actual, p_at_predicted) -> float:
    """True value of the DML (partially linear) estimand.

    DML regresses Y-residuals on T-residuals rt = T - E[T|X]. With a non-linear truth, that slope is an
    average of *secants*: for each trip, the change in acceptance between its predicted pay E[T|X] and
    its actual pay T, divided by rt, weighted by rt^2. ``p_at_predicted`` is the true acceptance at
    T - rt, holding the trip's unobserved difficulty fixed.
    """
    rt = np.asarray(rt, float)
    return float((rt * (np.asarray(p_actual) - np.asarray(p_at_predicted))).sum() / (rt**2).sum())


def true_iv(t, s, p_actual) -> float:
    """True value of the Wald IV estimand: cov(P_true, s) / cov(T, s)."""
    t, s, p = (np.asarray(v, float) for v in (t, s, p_actual))
    sc = s - s.mean()
    return float((sc * (p - p.mean())).sum() / (sc * (t - t.mean())).sum())


# ---------------------------------------------------------------- Bayesian hierarchical model

# "night" is not a model feature: it is constant within the night hour band, so the segment intercepts
# (cluster x band) already absorb it, and adding it would make the model unidentified.
FEATURES = ["short", "deadhead", "tight"]


def fit_hierarchical(df: pd.DataFrame, segments: list, draws: int = 1000, tune: int = 1000, chains: int = 2,
                     seed: int = 0):
    """Logistic model with segment-specific intercept and pay slope, hour bands pooled within cluster.

        logit P = alpha_g + beta_g * T - gamma . difficulty features
        beta_g  = mu_beta[cluster(g)] + sigma_beta * z_g         (non-centred)
        alpha_g = mu_alpha[cluster(g)] + sigma_alpha * w_g

    Each cluster has its own mean (weakly informative prior, no pooling across clusters: with only a
    handful of clusters, a city-level spread would be barely identified and would slow the sampler to a
    crawl). Hour bands within a cluster are partially pooled toward that cluster's mean, so a small
    segment (airports at night) borrows from its own cluster. ``df`` needs seg (index into
    ``segments``), log_ratio, accept and FEATURES.
    """
    import pymc as pm

    clusters = sorted({c for c, _ in segments})
    seg_cluster = np.array([clusters.index(c) for c, _ in segments])
    g = df["seg"].to_numpy()
    coords = {"seg": [f"{a}|{b}" for a, b in segments], "cluster": clusters, "feat": FEATURES}
    with pm.Model(coords=coords) as model:
        mu_b = pm.Normal("mu_beta", 3.0, 2.0, dims="cluster")
        sd_b = pm.HalfNormal("sigma_beta", 1.0)
        beta = pm.Deterministic("beta", mu_b[seg_cluster] + sd_b * pm.Normal("z_beta", 0, 1, dims="seg"), dims="seg")
        mu_a = pm.Normal("mu_alpha", 1.0, 2.0, dims="cluster")
        sd_a = pm.HalfNormal("sigma_alpha", 1.0)
        alpha = pm.Deterministic("alpha", mu_a[seg_cluster] + sd_a * pm.Normal("z_alpha", 0, 1, dims="seg"), dims="seg")
        gamma = pm.Normal("gamma", 0, 2, dims="feat")
        eta = alpha[g] + beta[g] * df["log_ratio"].to_numpy() - pm.math.dot(df[FEATURES].to_numpy(), gamma)
        pm.Bernoulli("accept", logit_p=eta, observed=df["accept"].to_numpy())
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 2), random_seed=seed,
                          progressbar=False, target_accept=0.9)
    return model, idata


def posterior_draws(idata, n: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    """Thin the posterior to n draws of alpha (n, G), beta (n, G), gamma (n, F) for downstream use."""
    post = _posterior(idata).stack(sample=("chain", "draw"))
    rng = np.random.default_rng(seed)
    pick = rng.choice(post.sizes["sample"], size=min(n, post.sizes["sample"]), replace=False)
    return {k: post[k].isel(sample=pick).transpose("sample", ...).to_numpy() for k in ("alpha", "beta", "gamma")}


def _posterior(idata):
    """The posterior as an xarray Dataset (newer ArviZ returns a DataTree node)."""
    post = idata.posterior
    return post.to_dataset() if hasattr(post, "to_dataset") else post


def predict_accept(alpha, beta, gamma, seg_idx, log_ratio, feats: np.ndarray) -> np.ndarray:
    """P(accept) from (point or drawn) parameters; vectorised over trips."""
    return expit(alpha[seg_idx] + beta[seg_idx] * log_ratio - feats @ gamma)
