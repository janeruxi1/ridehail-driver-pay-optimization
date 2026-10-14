"""Demand forecast: trips by zone and hour, four weeks ahead (phase 2).

Direct multi-horizon setup. At each forecast origin (a Monday 00:00) we build features from data
*before* the origin only, and one row per zone x future hour in the next four weeks. Using lag-1-week
features for a target three weeks out would leak (that week isn't observed yet at the origin), so
every feature is either known in advance (calendar, holidays) or summarises history up to the origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON_H = 4 * 7 * 24
WEEK_H = 7 * 24


@dataclass
class Panel:
    """Dense zone x hour matrix of trips (zeros where a zone had no trips that hour)."""
    trips: np.ndarray            # shape (n_zones, n_hours)
    zones: np.ndarray            # zone ids, row order
    hours: pd.DatetimeIndex      # column order, hourly, contiguous

    def col(self, ts) -> int:
        return int((pd.Timestamp(ts) - self.hours[0]) / pd.Timedelta(hours=1))


def make_panel(zone_hour: pd.DataFrame, zones=None) -> Panel:
    """From long (zone, hour, trips) rows to a dense matrix over a contiguous hourly range."""
    df = zone_hour[["zone", "hour", "trips"]]
    if zones is not None:
        df = df[df["zone"].isin(zones)]
    hours = pd.date_range(df["hour"].min(), df["hour"].max(), freq="h")
    zone_ids = np.sort(df["zone"].unique())
    zi = np.searchsorted(zone_ids, df["zone"].to_numpy())
    hi = ((df["hour"] - hours[0]) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    m = np.zeros((len(zone_ids), len(hours)), dtype=np.float32)
    np.add.at(m, (zi, hi), df["trips"].to_numpy(dtype=np.float32))
    return Panel(m, zone_ids, hours)


def holiday_dates(years) -> set:
    import holidays

    days = set()
    for y in years:
        days |= set(holidays.US(state="NY", years=y).keys())
    return {pd.Timestamp(d) for d in days}


def features_at_origin(p: Panel, origin, holidays_set: set, zone_cluster: dict | None = None,
                       horizon_h: int = HORIZON_H) -> pd.DataFrame:
    """One row per zone x hour in [origin, origin + horizon), features from data before origin."""
    o = p.col(origin)
    if o < 4 * WEEK_H:
        raise ValueError("need four weeks of history before the origin")
    hist = p.trips[:, o - 4 * WEEK_H:o].reshape(len(p.zones), 4, WEEK_H)   # zone, week, hour of week
    prof4 = hist.mean(axis=1)                    # same hour of week, mean of last 4 weeks
    last1 = hist[:, -1, :]                       # same hour of week, last week (seasonal naive)
    level = hist[:, -1, :].sum(axis=1) / np.maximum(hist.sum(axis=(1, 2)) / 4, 1e-6)   # last week vs 4-week mean
    zone_total4 = hist.sum(axis=(1, 2)) / 4
    n_h = min(horizon_h, p.trips.shape[1] - o)
    h = np.arange(n_h)
    t_index = p.hours[o] + pd.to_timedelta(h, unit="h")
    how_target = (t_index.dayofweek * 24 + t_index.hour).to_numpy()
    how_origin0 = p.hours[o].dayofweek * 24 + p.hours[o].hour
    # hour-of-week column in `hist` for each target hour (history window starts at origin - 4 weeks)
    k = (how_target - how_origin0) % WEEK_H
    # year-ago: same zone, 364 days (52 weeks) before the target hour, if available and before origin
    ya_cols = o + h - 52 * WEEK_H
    has_ya = ya_cols >= 0
    ya = np.full((len(p.zones), n_h), np.nan, dtype=np.float32)
    ya[:, has_ya] = p.trips[:, ya_cols[has_ya]]
    # year-ago profile level, to turn the year-ago value into a ratio against its own baseline
    dates = t_index.normalize()
    is_hol = np.array([d in holidays_set for d in dates])
    hol_next = np.array([(d + pd.Timedelta(days=1)) in holidays_set for d in dates])
    hol_prev = np.array([(d - pd.Timedelta(days=1)) in holidays_set for d in dates])
    nz = len(p.zones)
    out = pd.DataFrame({
        "zone": np.repeat(p.zones, n_h),
        "hour": np.tile(t_index.to_numpy(), nz),
        "horizon_h": np.tile(h, nz),
        "how": np.tile(how_target, nz),
        "hod": np.tile(t_index.hour.to_numpy(), nz),
        "dow": np.tile(t_index.dayofweek.to_numpy(), nz),
        "month": np.tile(t_index.month.to_numpy(), nz),
        "holiday": np.tile(is_hol, nz).astype(np.int8),
        "day_before_holiday": np.tile(hol_next, nz).astype(np.int8),
        "day_after_holiday": np.tile(hol_prev, nz).astype(np.int8),
        "prof4": prof4[:, k].ravel(),
        "last1": last1[:, k].ravel(),
        "level_ratio": np.repeat(level, n_h),
        "zone_week_trips": np.repeat(zone_total4, n_h),
        "year_ago": ya.ravel(),
        "y": p.trips[:, o:o + n_h].ravel(),
    })
    if zone_cluster is not None:
        out["cluster"] = out["zone"].map(zone_cluster).fillna(-1).astype(int)
    out["origin"] = pd.Timestamp(p.hours[o])
    return out


def wmape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum())


def bias(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float((yhat - y).sum() / y.sum())
