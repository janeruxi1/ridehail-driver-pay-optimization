"""The pay floor: New York's minimum pay standard for high-volume app drivers, applied per trip.

The TLC sets a per-mile and a per-minute rate (they already include the TLC's utilization factor)
and raises them every 1 March with inflation. Rates used here, for non-wheelchair-accessible trips:

* from 1 March 2024: $1.360 per mile, $0.583 per minute; trips ending outside NYC $1.576 and $0.675
  (TLC industry notice 24-02);
* before that: the same rates divided by the 3.49% increase in that notice;
* from 1 March 2025: the 2024 rates raised 3.92% (TLC, February 2025 presentation).

Simplification, stated in the README: the real standard is enforced on a driver's weekly earnings,
not trip by trip. Phase 1 checks this formula against the data: the median trip is paid within
1% of it, so it is the platform's base pay in practice. Pay above it is the "premium".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import OUTSIDE_NYC_ZONE

# (start date, per mile, per minute, per mile out of town, per minute out of town)
RATES = [
    ("2023-03-01", 1.360 / 1.0349, 0.583 / 1.0349, 1.576 / 1.0349, 0.675 / 1.0349),
    ("2024-03-01", 1.360, 0.583, 1.576, 0.675),
    ("2025-03-01", 1.360 * 1.0392, 0.583 * 1.0392, 1.576 * 1.0392, 0.675 * 1.0392),
]
PAY_CAP_MULTIPLE = 1.30   # constraint C1: the policy may pay at most 1.30 x the floor


def rates_on(dates) -> np.ndarray:
    """Array of shape (n, 4) with the rates in force on each date."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy()
    starts = pd.to_datetime([r[0] for r in RATES]).to_numpy()
    idx = np.searchsorted(starts, d, side="right") - 1
    idx = np.clip(idx, 0, len(RATES) - 1)
    table = np.array([r[1:] for r in RATES])
    return table[idx]


def floor_pay(miles, minutes, dates, do_zone=None) -> np.ndarray:
    """Minimum pay for each trip given miles, minutes, request date and drop-off zone."""
    miles = np.asarray(miles, dtype=float)
    minutes = np.asarray(minutes, dtype=float)
    r = rates_on(dates)
    out_of_town = np.zeros(len(miles), bool) if do_zone is None else np.asarray(do_zone) == OUTSIDE_NYC_ZONE
    per_mile = np.where(out_of_town, r[:, 2], r[:, 0])
    per_min = np.where(out_of_town, r[:, 3], r[:, 1])
    return per_mile * miles + per_min * minutes
