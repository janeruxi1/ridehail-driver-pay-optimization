"""A discrete-event dispatch simulation that replays real requests (phase 6).

Requests are real: every Brooklyn pickup on a given day, with its real time, pickup and drop-off
zone, distance, duration and rider fare. Drivers are simulated: they come online, get offered trips,
accept or decline (from the planted acceptance model), drive to the pickup and the drop-off, and
go offline. Locations are a random point inside the real zone (zone centroid plus a radius set by
the zone's area); drive time to a pickup is road distance (straight line x 1.3) at the hour's real
average speed.

Every ``batch_sec`` seconds, open requests are matched to idle drivers:
* ``greedy``: oldest request first, each to its nearest idle driver (a common simple rule);
* ``batch``: one assignment problem over all open requests and idle drivers at once, minimizing
  total pickup time (Hungarian algorithm, scipy ``linear_sum_assignment``).
A request is never offered twice to a driver who declined it.

A matched driver accepts with the acceptance probability of the offer; on a decline the request
waits for the next batch and the rider may give up (the same per-decline probability as the policy
model) or after ``patience_sec`` unserved. No driver is sent more than ``MAX_PICKUP_MIN`` away. A driver
idle for ``reposition_min`` moves to where requests are currently coming from (drivers follow demand
heat maps); without this, local shortages persist and waits get an unrealistically long tail.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .policy import CANCEL_PER_DECLINE

ROAD_FACTOR = 1.3
MAX_PICKUP_MIN = 12.0          # no driver is sent more than 12 minutes away
PICKUP_OVERHEAD_MIN = 1.0      # accepting, setting off, the rider coming out: real waits are rarely under a minute


@dataclass
class SimConfig:
    batch_sec: int = 30
    matching: str = "batch"            # or "greedy"
    patience_sec: int = 20 * 60        # real waits reach 16 minutes at the 99th percentile
    supply_scale: float = 1.0          # multiplies the driver arrival rate
    mean_shift_hours: float = 4.0
    reposition_min: float = 5.0        # an idle driver with no trip for this long moves toward demand
    seed: int = 0


def _jitter(rng, x, y, radius):
    r = radius * np.sqrt(rng.random(len(x)))
    a = rng.uniform(0, 2 * np.pi, len(x))
    return x + r * np.cos(a), y + r * np.sin(a)


def run(requests: pd.DataFrame, accept_prob: np.ndarray, offer: np.ndarray, zones: pd.DataFrame,
        arrival_per_hour: np.ndarray, speed_mph: np.ndarray, cfg: SimConfig) -> dict:
    """Simulate one day.

    requests: sorted by request time, with t_sec (seconds since the day's start), PULocationID,
      DOLocationID, minutes, fare, floor; ``accept_prob`` and ``offer`` per request (the policy).
    zones: indexed by zone with x_mi, y_mi, radius_mi, in_area (bool: drop-offs outside the area take
      the driver out of the simulation).
    arrival_per_hour: drivers coming online per hour of day (24 values) before ``supply_scale``.
    """
    rng = np.random.default_rng(cfg.seed)
    n = len(requests)
    t_req = requests["t_sec"].to_numpy()
    zx, zy = zones["x_mi"], zones["y_mi"]
    zr = zones["radius_mi"]
    px, py = _jitter(rng, requests["PULocationID"].map(zx).to_numpy(), requests["PULocationID"].map(zy).to_numpy(),
                     requests["PULocationID"].map(zr).to_numpy())
    dx, dy = _jitter(rng, requests["DOLocationID"].map(zx).fillna(0).to_numpy(),
                     requests["DOLocationID"].map(zy).fillna(0).to_numpy(), requests["DOLocationID"].map(zr).fillna(0).to_numpy())
    stays = requests["DOLocationID"].map(zones["in_area"].astype(bool)).eq(True).to_numpy()
    trip_sec = requests["minutes"].to_numpy() * 60
    u_accept = rng.random((n, 8))           # pre-drawn decisions for up to 8 offers per request
    # drivers appear at the location of a randomly chosen request of the day (demand-weighted)
    cap = 20000
    d_x, d_y = np.zeros(cap), np.zeros(cap)
    d_free_at = np.full(cap, np.inf)        # time the driver becomes idle (inf = not yet online / offline)
    d_online_from = np.zeros(cap)
    d_offline_at = np.zeros(cap)
    d_active = np.zeros(cap, bool)
    d_pay = np.zeros(cap)
    nd = 0
    # request state
    status = np.zeros(n, np.int8)           # 0 waiting/not yet, 1 matched, 2 cancelled
    offers_made = np.zeros(n, np.int16)
    declined_by: dict[int, set] = {}
    wait = np.full(n, np.nan)
    pickup_min = np.full(n, np.nan)
    next_req = 0
    open_req: list[int] = []
    horizon = int(t_req.max()) + 3600
    for t in range(0, horizon, cfg.batch_sec):
        hour = min(int(t // 3600), 23)
        # drivers coming online in this batch (Poisson), at a random recent request location
        rate = arrival_per_hour[hour] * cfg.supply_scale
        # at t = 0 start from a steady-state fleet (arrival rate x mean shift length), not an empty city
        k = rng.poisson(rate * cfg.mean_shift_hours if t == 0 else rate * cfg.batch_sec / 3600)
        if k:
            if nd + k > cap:                    # grow the driver arrays
                grow = max(cap, k)
                d_x, d_y = np.r_[d_x, np.zeros(grow)], np.r_[d_y, np.zeros(grow)]
                d_free_at = np.r_[d_free_at, np.full(grow, np.inf)]
                d_online_from, d_offline_at = np.r_[d_online_from, np.zeros(grow)], np.r_[d_offline_at, np.zeros(grow)]
                d_active, d_pay = np.r_[d_active, np.zeros(grow, bool)], np.r_[d_pay, np.zeros(grow)]
                cap += grow
            src = rng.integers(0, n, k)
            ids = np.arange(nd, nd + k)
            d_x[ids], d_y[ids] = px[src], py[src]
            d_free_at[ids] = t
            d_online_from[ids] = t
            d_offline_at[ids] = t + rng.exponential(cfg.mean_shift_hours * 3600, k)
            d_active[ids] = True
            nd += k
        # requests arriving in this batch
        while next_req < n and t_req[next_req] < t + cfg.batch_sec:
            open_req.append(next_req)
            next_req += 1
        # idle drivers with no trip for a while move to where requests are coming from (as drivers do
        # with a demand heat map); simplification: the move is instant and counted as idle time
        idle_long = np.flatnonzero(d_active[:nd] & (d_free_at[:nd] <= t - cfg.reposition_min * 60))
        if len(idle_long):
            lo_i = np.searchsorted(t_req, t - 1800)
            hi_i = max(np.searchsorted(t_req, t + 1800), lo_i + 1)
            src = rng.integers(lo_i, min(hi_i, n), len(idle_long)) if lo_i < n else rng.integers(0, n, len(idle_long))
            d_x[idle_long], d_y[idle_long] = px[src], py[src]
            d_free_at[idle_long] = t              # restart the idle clock after a move
        # drivers going offline when idle past their shift end
        act = np.flatnonzero(d_active[:nd])
        off = act[(d_offline_at[act] <= t) & (d_free_at[act] <= t)]
        d_active[off] = False
        # patience
        still = []
        for r in open_req:
            if t_req[r] <= t and t - t_req[r] > cfg.patience_sec:
                status[r] = 2
            else:
                still.append(r)
        open_req = still
        ready = [r for r in open_req if t_req[r] <= t + cfg.batch_sec]
        idle = np.flatnonzero(d_active[:nd] & (d_free_at[:nd] <= t + cfg.batch_sec))
        if not ready or not len(idle):
            continue
        R = np.array(ready)
        dist = np.hypot(px[R][:, None] - d_x[idle][None, :], py[R][:, None] - d_y[idle][None, :]) * ROAD_FACTOR
        eta_min = dist / speed_mph[hour] * 60 + PICKUP_OVERHEAD_MIN
        for i, r in enumerate(R):
            if r in declined_by:
                eta_min[i, np.isin(idle, list(declined_by[r]))] = np.inf
        eta_min[eta_min > MAX_PICKUP_MIN] = np.inf
        pairs = _match(eta_min, cfg.matching)
        for i, j in pairs:
            r, d = int(R[i]), int(idle[j])
            k_off = min(offers_made[r], 7)
            offers_made[r] += 1
            if u_accept[r, k_off] < accept_prob[r]:
                start = max(t, t_req[r], d_free_at[d])
                pm = eta_min[i, j]
                wait[r] = start + pm * 60 - t_req[r]
                pickup_min[r] = pm
                status[r] = 1
                d_free_at[d] = start + pm * 60 + trip_sec[r]
                d_pay[d] += offer[r]
                if stays[r]:
                    d_x[d], d_y[d] = dx[r], dy[r]
                else:                                   # left the area: out of this simulation
                    d_offline_at[d] = d_free_at[d]
                open_req.remove(r)
            else:
                declined_by.setdefault(r, set()).add(d)
                if rng.random() < CANCEL_PER_DECLINE:
                    status[r] = 2
                    open_req.remove(r)
    for r in open_req:
        status[r] = 2
    online_h = np.clip(np.minimum(d_offline_at[:nd], horizon) - d_online_from[:nd], 0, None).sum() / 3600
    done = status == 1
    first = np.zeros(n, bool)
    first[done] = offers_made[done] == 1
    return {"completed": done, "cancelled": status == 2, "wait_sec": wait, "pickup_min": pickup_min,
            "offers": offers_made, "first_offer_accept": first, "driver_pay_total": float(d_pay[:nd].sum()),
            "driver_online_hours": float(online_h), "drivers": int(nd)}


def _match(eta: np.ndarray, how: str) -> list[tuple[int, int]]:
    finite = np.isfinite(eta)
    if not finite.any():
        return []
    if how == "greedy":
        pairs, taken = [], set()
        for i in range(eta.shape[0]):                  # rows are in request-time order
            row = np.where(finite[i], eta[i], np.inf)
            if taken:
                row[list(taken)] = np.inf
            j = int(np.argmin(row))
            if np.isfinite(row[j]):
                pairs.append((i, j))
                taken.add(j)
        return pairs
    big = 1e6
    cost = np.where(finite, eta, big)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < big]


def summarize(requests: pd.DataFrame, res: dict, margin_fare: np.ndarray, offer: np.ndarray,
              cancel_cost: float) -> dict:
    done = res["completed"]
    w = res["wait_sec"][done] / 60
    n = len(requests)
    return {
        "requests": n, "completion": float(done.mean()), "first_offer_acceptance": float(res["first_offer_accept"].sum() / max(1, (res["offers"] > 0).sum())),
        "wait_mean_min": float(w.mean()), "wait_p50_min": float(np.median(w)), "wait_p90_min": float(np.percentile(w, 90)),
        "pickup_mean_min": float(np.nanmean(res["pickup_min"][done])),
        "margin_per_request": float(((margin_fare - offer) * done).sum() / n - cancel_cost * res["cancelled"].mean()),
        "pay_per_trip": float(offer[done].mean()),
        "earnings_per_online_hour": res["driver_pay_total"] / max(res["driver_online_hours"], 1e-9),
        "drivers": res["drivers"],
    }
