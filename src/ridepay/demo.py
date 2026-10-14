"""Small pure functions behind the Streamlit demo (kept here so they are unit-tested)."""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from .pay import PAY_CAP_MULTIPLE, floor_pay
from .policy import PREMIUMS, outcomes
from .quote import QuoteModel


def offer_curve(model: QuoteModel, pickup_zone: int, dropoff_zone: int, request_time: datetime, trip_miles: float,
                trip_minutes: float, rider_fare: float, prev_hour_wait_min: float | None = None) -> pd.DataFrame:
    """Predicted acceptance, completion and expected margin for every premium the policy may choose.

    Empty if the pickup zone has no segment (sparse zones are always paid the floor).
    """
    from .config import BAND_OF_HOUR
    from .simulate import TIGHT_CENTER_MIN

    seg = f"{model.cluster_of.get(pickup_zone, 'Sparse')}|{BAND_OF_HOUR[request_time.hour]}"
    if seg not in model.seg_index:
        return pd.DataFrame()
    wait = TIGHT_CENTER_MIN if prev_hour_wait_min is None else prev_hour_wait_min
    f = model.features(pickup_zone, dropoff_zone, trip_miles, request_time.hour, wait)
    d_hat = sum(coef * f[k] for k, coef in model.gamma.items())
    i = model.seg_index[seg]
    floor = float(floor_pay([trip_miles], [trip_minutes], [request_time], [dropoff_zone])[0])
    prem = PREMIUMS[PREMIUMS <= PAY_CAP_MULTIPLE - 1 + 1e-9]
    p = 1 / (1 + np.exp(-(model.alpha[i] + model.beta[i] * np.log1p(prem) - d_hat)))
    pay = floor * (1 + prem)
    o = outcomes(p, rider_fare, pay, trip_minutes)
    return pd.DataFrame({"premium_pct": 100 * prem, "pay": pay, "accept_probability": p,
                         "completion_probability": o["complete"], "expected_margin": o["margin"]})


def switchback_mde(sd_4wk: float, bias_share: float, weeks: float) -> float:
    """Minimum detectable effect (80% power, 5% two-sided) for a test of `weeks` weeks, scaling the
    4-week randomization standard deviation by sqrt(4 / weeks)."""
    sd = sd_4wk * math.sqrt(4.0 / weeks)
    return 2.8 * sd / max(1e-9, 1 + bias_share)


def switchback_power(sd_4wk: float, bias_share: float, weeks: float, effect: float) -> float:
    from scipy.stats import norm

    sd = sd_4wk * math.sqrt(4.0 / weeks)
    z = effect * (1 + bias_share) / sd
    return float(norm.sf(1.96 - z) + norm.cdf(-1.96 - z))
