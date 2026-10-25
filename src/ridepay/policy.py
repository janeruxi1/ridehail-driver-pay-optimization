"""Choosing pay (phase 6): expected outcomes of an offer, and a constrained optimizer.

**Offer model.** A trip is offered to drivers one at a time at the same pay (constraint C2: the
offer depends on the trip, not on who receives it). Each driver accepts with probability p; each
decline adds ``REOFFER_SEC`` to the rider's wait and the rider gives up with probability
``CANCEL_PER_DECLINE``. With r = (1 - p)(1 - q):

    P(trip completes)            = p / (1 - r)
    E[declines | completes]       = r / (1 - r)
    expected margin per request  = P(complete) * (fare - pay) - P(cancel) * CANCEL_COST

**Policy.** Pay = floor x (1 + premium), one premium per cell = (zone cluster, hour band,
difficulty quintile). Premium options run from 0 to 30% (constraint C1: never below the floor, never
above 1.3 x the floor).

**Optimizer.** Pick one option per cell (binary variables) to maximize expected margin subject to:
* C3: expected extra wait in each segment (cluster x band) no worse than today;
* C4: total first-offer acceptance no lower than today;
* C5: driver earnings per engaged hour in each cluster at least 95% of today's.
All are linear in the choice variables, so this is a mixed-integer linear program. To make it robust
to estimation error, the constraints can be imposed under several posterior draws at once (the
scenario approach): the policy must satisfy the guardrails in every sampled version of the world.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .pay import PAY_CAP_MULTIPLE

REOFFER_SEC = 20.0
CANCEL_PER_DECLINE = 0.06
CANCEL_COST = 5.0
EARNINGS_FLOOR = 0.95
PREMIUMS = np.round(np.arange(0.0, PAY_CAP_MULTIPLE - 1 + 1e-9, 0.025), 3)


def outcomes(p, fare, pay, minutes, pickup_min=4.0) -> dict[str, np.ndarray]:
    """Expected per-request outcomes for first-offer acceptance p (arrays broadcast)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = CANCEL_PER_DECLINE
    r = (1 - p) * (1 - q)
    complete = p / (1 - r)
    declines = r / (1 - r)
    return {
        "accept_first": p,
        "complete": complete,
        "extra_wait_sec": REOFFER_SEC * declines * complete,        # summed over requests -> per completed trip later
        "margin": complete * (fare - pay) - (1 - complete) * CANCEL_COST,
        "driver_pay": complete * pay,
        "engaged_hours": complete * (minutes + pickup_min) / 60,
    }


@dataclass
class CellTable:
    """Sums of expected outcomes for each cell and premium option, under one or more parameter draws."""
    cells: pd.DataFrame          # one row per cell: cell id, cluster, band, n
    sums: dict                   # name -> array (n_draws, n_cells, n_options)
    status_quo: dict             # name -> array (n_draws, n_cells): today's offers


def cell_table(df: pd.DataFrame, accept_fn, draws: list) -> CellTable:
    """Aggregate expected outcomes by cell for every premium option and for today's offers.

    ``df`` needs: cell, cluster_name, band, floor, offer_sq, fare, minutes.
    ``accept_fn(df, log_ratio, draw)`` returns P(accept) per trip under parameter draw ``draw``.
    """
    cells = df.groupby("cell").agg(cluster_name=("cluster_name", "first"), band=("band", "first"),
                                   n=("cell", "size")).reset_index()
    codes = df["cell"].map(dict(zip(cells["cell"], range(len(cells))))).to_numpy()
    names = ["accept_first", "complete", "extra_wait_sec", "margin", "driver_pay", "engaged_hours"]
    D, C, K = len(draws), len(cells), len(PREMIUMS)
    sums = {k: np.zeros((D, C, K)) for k in names}
    sq = {k: np.zeros((D, C)) for k in names}
    fare, floor, minutes = df["fare"].to_numpy(), df["floor"].to_numpy(), df["minutes"].to_numpy()
    for d, draw in enumerate(draws):
        for j, prem in enumerate(PREMIUMS):
            pay = floor * (1 + prem)
            o = outcomes(accept_fn(df, np.full(len(df), np.log1p(prem)), draw), fare, pay, minutes)
            for k in names:
                sums[k][d, :, j] = np.bincount(codes, weights=o[k], minlength=C)
        lr = np.log(df["offer_sq"].to_numpy() / floor)
        o = outcomes(accept_fn(df, lr, draw), fare, df["offer_sq"].to_numpy(), minutes)
        for k in names:
            sq[k][d] = np.bincount(codes, weights=o[k], minlength=C)
    return CellTable(cells, sums, sq)


def optimize(t: CellTable, constraint_draws=None, objective_draws=None, time_limit: float = 120,
             relax_if_infeasible: bool = True):
    """Solve the MILP. Objective: mean margin over ``objective_draws``; constraints in each of
    ``constraint_draws`` (indices into the table's draws). Returns chosen premium index per cell.

    Today's offers are not on the premium grid, so no grid policy is guaranteed to meet every
    guardrail. If none does, the policy that breaks the guardrails least (summed relative shortfall)
    is returned instead, with ``res.relaxed = True`` and the shortfall in ``res.violation``; the
    guardrails are always checked separately when the policy is scored."""
    D, C, K = t.sums["margin"].shape
    constraint_draws = list(range(D)) if constraint_draws is None else list(constraint_draws)
    objective_draws = list(range(D)) if objective_draws is None else list(objective_draws)
    nvar = C * K
    c = -t.sums["margin"][objective_draws].mean(axis=0).ravel()
    A, lo, hi = [], [], []
    # one option per cell
    for i in range(C):
        row = np.zeros(nvar)
        row[i * K:(i + 1) * K] = 1
        A.append(row)
        lo.append(1)
        hi.append(1)
    seg = t.cells.groupby(["cluster_name", "band"]).indices
    clus = t.cells.groupby("cluster_name").indices
    for d in constraint_draws:
        S, Q = t.sums, t.status_quo
        # C3: extra wait per segment <= today
        for idx in seg.values():
            row = np.zeros((C, K))
            row[idx] = S["extra_wait_sec"][d, idx]
            A.append(row.ravel())
            lo.append(-np.inf)
            hi.append(Q["extra_wait_sec"][d, idx].sum() + 1e-6)
        # C4: total first-offer acceptance >= today
        A.append(S["accept_first"][d].ravel())
        lo.append(Q["accept_first"][d].sum() - 1e-6)
        hi.append(np.inf)
        # C5: pay - 0.95 * today's earnings rate * engaged hours >= 0, per cluster
        for idx in clus.values():
            rate = Q["driver_pay"][d, idx].sum() / Q["engaged_hours"][d, idx].sum()
            row = np.zeros((C, K))
            row[idx] = S["driver_pay"][d, idx] - EARNINGS_FLOOR * rate * S["engaged_hours"][d, idx]
            A.append(row.ravel())
            lo.append(-1e-6)
            hi.append(np.inf)
    A, lo, hi = np.array(A), np.array(lo), np.array(hi)
    options = {"time_limit": time_limit, "disp": False}
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nvar), bounds=Bounds(0, 1),
               options=options)
    res.relaxed, res.violation = False, 0.0
    if res.x is None:
        if not relax_if_infeasible or res.status != 2:          # 2: infeasible; anything else is a real failure
            raise RuntimeError(f"optimizer failed: {res.message}")
        res = _least_violation(c, A, lo, hi, n_hard=C, options=options)
    choice = res.x[:nvar].reshape(C, K).argmax(axis=1)
    return choice, res


def _least_violation(c, A, lo, hi, n_hard: int, options: dict):
    """Elastic version of the MILP: one slack per guardrail row (the first ``n_hard`` rows, one option
    per cell, stay hard). Stage 1 minimizes the summed relative shortfall; stage 2 maximizes margin
    among policies with that shortfall."""
    m, nvar = A.shape[0] - n_hard, A.shape[1]
    upper = np.isfinite(hi[n_hard:])                               # rows of the form a·x <= hi
    bound = np.where(upper, hi[n_hard:], lo[n_hard:])
    scale = np.maximum(np.abs(bound), 1.0)
    S = np.zeros((A.shape[0], m))
    S[n_hard + np.arange(m), np.arange(m)] = np.where(upper, -1.0, 1.0)
    A2 = np.hstack([A, S])
    integrality = np.r_[np.ones(nvar), np.zeros(m)]
    bounds = Bounds(np.zeros(nvar + m), np.r_[np.ones(nvar), np.full(m, np.inf)])
    w = np.r_[np.zeros(nvar), 1 / scale]
    stage1 = milp(w, constraints=LinearConstraint(A2, lo, hi), integrality=integrality, bounds=bounds,
                  options=options)
    if stage1.x is None:
        raise RuntimeError(f"optimizer failed: {stage1.message}")
    cap = stage1.fun * (1 + 1e-6) + 1e-9
    stage2 = milp(np.r_[c, np.zeros(m)], constraints=[LinearConstraint(A2, lo, hi), LinearConstraint(w, -np.inf, cap)],
                  integrality=integrality, bounds=bounds, options=options)
    res = stage2 if stage2.x is not None else stage1
    res.relaxed, res.violation = True, float(stage1.fun)
    res.message = f"no policy meets every guardrail; least total shortfall {stage1.fun:.4f} ({res.message})"
    return res


def summarize(t: CellTable, choice, draw: int = 0) -> dict:
    """Totals for a choice (or 'sq' for today) under one draw, as guardrail-ready metrics."""
    C = len(t.cells)
    if isinstance(choice, str):
        tot = {k: v[draw] for k, v in t.status_quo.items()}
    else:
        tot = {k: v[draw, np.arange(C), choice] for k, v in t.sums.items()}
    n = t.cells["n"].sum()
    return {"margin_per_request": tot["margin"].sum() / n, "first_offer_acceptance": tot["accept_first"].sum() / n,
            "completion": tot["complete"].sum() / n, "extra_wait_sec": tot["extra_wait_sec"].sum() / tot["complete"].sum(),
            "pay_per_trip": tot["driver_pay"].sum() / tot["complete"].sum(),
            "earnings_per_hour": tot["driver_pay"].sum() / tot["engaged_hours"].sum()}


def guardrails(t: CellTable, choice, draw: int = 0) -> dict:
    """Check C3, C4, C5 for a choice under one draw. True = satisfied."""
    C = len(t.cells)
    S = {k: v[draw, np.arange(C), choice] for k, v in t.sums.items()}
    Q = {k: v[draw] for k, v in t.status_quo.items()}
    seg = t.cells.groupby(["cluster_name", "band"]).indices
    clus = t.cells.groupby("cluster_name").indices
    c3 = all(S["extra_wait_sec"][i].sum() <= Q["extra_wait_sec"][i].sum() * (1 + 1e-6) + 1e-6 for i in seg.values())
    c4 = S["accept_first"].sum() >= Q["accept_first"].sum() * (1 - 1e-6)
    c5 = all(S["driver_pay"][i].sum() / S["engaged_hours"][i].sum()
             >= EARNINGS_FLOOR * Q["driver_pay"][i].sum() / Q["engaged_hours"][i].sum() * (1 - 1e-6) for i in clus.values())
    return {"C3_wait": bool(c3), "C4_acceptance": bool(c4), "C5_earnings": bool(c5)}
