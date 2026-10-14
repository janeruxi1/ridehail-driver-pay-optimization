"""The quote engine behind the API: trip in, recommended driver pay out.

Pure Python + numpy, no data files beyond one JSON model artifact (``models/quote_model.json``),
so it runs in a small container. The artifact is exported by ``scripts/export_model.py`` from the
phase 1 clusters, the phase 5 acceptance model and the phase 6 policy.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import config
from .pay import PAY_CAP_MULTIPLE, floor_pay
from .policy import CANCEL_COST, outcomes
from .simulate import DIFFICULTY, TIGHT_CENTER_MIN, TIGHT_SCALE_MIN

DEFAULT_MODEL = config.ROOT / "models" / "quote_model.json"


@dataclass
class Quote:
    model_version: str
    floor_pay: float
    recommended_pay: float
    premium_pct: float
    segment: str
    difficulty_quintile: int
    accept_probability: float
    completion_probability: float
    expected_margin: float
    rule: str                      # "policy" or "default (no premium)" for zones without a segment

    def to_dict(self) -> dict:
        return asdict(self)


class QuoteModel:
    def __init__(self, artifact: dict):
        self.a = artifact
        self.version = artifact["model_version"]
        self.cluster_of = {int(k): v for k, v in artifact["zone_cluster"].items()}
        self.deadhead_of = {int(k): float(v) for k, v in artifact["zone_arrivals_per_pickup"].items()}
        self.seg_index = {s: i for i, s in enumerate(artifact["segments"])}
        self.alpha, self.beta = artifact["alpha"], artifact["beta"]
        self.gamma = artifact["gamma"]                    # dict feature -> coefficient
        self.edges = artifact["difficulty_edges"]
        self.premium = artifact["premium"]                # cell -> premium

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> "QuoteModel":
        return cls(json.loads(Path(path).read_text()))

    def features(self, pickup_zone: int, dropoff_zone: int, miles: float, hour: int, prev_wait_min: float) -> dict:
        app = math.e if dropoff_zone == config.OUTSIDE_NYC_ZONE else self.deadhead_of.get(dropoff_zone, 1.0)
        return {
            "short": min(max((2.0 - miles) / 2.0, 0.0), 1.0),
            "deadhead": min(max(math.log(max(app, 1e-6)), -0.5), 1.5),
            "tight": min(max((prev_wait_min - TIGHT_CENTER_MIN) / TIGHT_SCALE_MIN, -2.0), 3.0),
            "night": 1.0 if config.BAND_OF_HOUR[hour] == "night" else 0.0,
        }

    def quote(self, pickup_zone: int, dropoff_zone: int, request_time: datetime, trip_miles: float,
              trip_minutes: float, rider_fare: float, prev_hour_wait_min: float | None = None) -> Quote:
        floor = float(floor_pay([trip_miles], [trip_minutes], [request_time], [dropoff_zone])[0])
        band = config.BAND_OF_HOUR[request_time.hour]
        cluster = self.cluster_of.get(pickup_zone, "Sparse")
        seg = f"{cluster}|{band}"
        wait = TIGHT_CENTER_MIN if prev_hour_wait_min is None else prev_hour_wait_min
        f = self.features(pickup_zone, dropoff_zone, trip_miles, request_time.hour, wait)
        d_hat = sum(coef * f[k] for k, coef in self.gamma.items())
        q = 1 + sum(d_hat > e for e in self.edges)
        if seg not in self.seg_index:
            premium, rule, p = 0.0, "default (no premium)", float("nan")
        else:
            premium = float(self.premium.get(f"{seg}|q{q}", 0.0))
            rule = "policy"
            i = self.seg_index[seg]
            eta = self.alpha[i] + self.beta[i] * math.log1p(premium) - d_hat
            p = 1 / (1 + math.exp(-eta))
        premium = min(max(premium, 0.0), PAY_CAP_MULTIPLE - 1)          # C1, whatever the artifact says
        pay = round(floor * (1 + premium), 2)
        if math.isnan(p):
            comp, margin = float("nan"), float("nan")
        else:
            o = outcomes(p, rider_fare, pay, trip_minutes)
            comp, margin = float(o["complete"]), float(o["margin"])
        return Quote(self.version, round(floor, 2), pay, round(100 * premium, 1), seg, int(q),
                     round(p, 4) if not math.isnan(p) else None, round(comp, 4) if not math.isnan(comp) else None,
                     round(margin, 2) if not math.isnan(margin) else None, rule)


__all__ = ["Quote", "QuoteModel", "DIFFICULTY", "CANCEL_COST"]
