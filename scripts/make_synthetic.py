"""Make a small synthetic city in exactly the schema of the built tables, so CI can run every notebook
without downloading 8 GB of TLC files. The numbers are invented (only the zone map is real); notebooks
refuse to write their results to ``reports/`` when they run on it.

    python scripts/make_synthetic.py        # writes data/synthetic/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridepay import config  # noqa: E402
from ridepay.data import SYNTHETIC  # noqa: E402

RNG = np.random.default_rng(2025)
ZONES = pd.read_csv(config.DATA / "zones" / "zone_centroids.csv").set_index("zone")
FEE_START = pd.Timestamp(config.CBD_FEE_START)


def pick_zones() -> pd.DataFrame:
    """About 70 zones: airports, lower Manhattan (fee zone), upper Manhattan, Brooklyn, Queens, Bronx."""
    z = ZONES.copy()
    man = z[z["borough"] == "Manhattan"].sort_values("y_mi")
    chosen = list(config.AIRPORT_ZONES)
    chosen += man.index[:14].tolist() + man.index[-8:].tolist()
    for b, k in [("Brooklyn", 22), ("Queens", 12), ("Bronx", 8), ("Staten Island", 3)]:
        chosen += z[z["borough"] == b].sample(k, random_state=1).index.tolist()
    out = z.loc[sorted(set(chosen))].copy()
    out["fee_zone"] = out.index.isin(man.index[:14])
    base = {"Manhattan": 60, "Brooklyn": 30, "Queens": 22, "Bronx": 18, "Staten Island": 4, "EWR": 6}
    out["level"] = out["borough"].map(base) * np.exp(RNG.normal(0, 0.5, len(out)))
    out.loc[out.index.isin(config.AIRPORT_ZONES), "level"] = 80
    out["wait_base"] = np.where(out["borough"] == "Manhattan", 3.8, 4.8) + RNG.normal(0, 0.6, len(out)).clip(-1, 2)
    out["miles_base"] = np.where(out.index.isin(config.AIRPORT_ZONES), 14, np.where(out["borough"] == "Manhattan", 3.2, 4.5))
    return out


HOUR_SHAPE = np.array([.5, .35, .25, .2, .2, .3, .6, 1.0, 1.2, 1.0, .9, .9, 1.0, 1.0, 1.05, 1.1, 1.2, 1.3,
                       1.35, 1.25, 1.1, 1.0, .9, .7])


def zone_hour_month(month: str, zones: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(month + "-01")
    hours = pd.date_range(start, start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1), freq="h")
    season = 1 + 0.08 * np.sin(2 * np.pi * (hours.dayofyear.to_numpy() / 365.0))
    weekend = np.where(hours.dayofweek.to_numpy() >= 5, 1.15, 1.0)
    rows = []
    for zid, z in zones.iterrows():
        mu = z["level"] * HOUR_SHAPE[hours.hour] * season * weekend
        if z["fee_zone"]:
            mu = mu * np.where(hours >= FEE_START, 0.93, 1.0)
        trips = RNG.poisson(mu)
        keep = trips > 0
        n = trips[keep]
        wait_mean = (z["wait_base"] + 1.5 * (hours.hour.to_numpy() < 5) - 0.3 * (hours >= FEE_START) * z["fee_zone"])[keep]
        wait_n = (n * 0.99).astype(int)
        miles = n * z["miles_base"] * np.exp(RNG.normal(0, 0.1, keep.sum()))
        mph = np.where(z["fee_zone"] & (hours[keep] >= FEE_START), 10.0, 9.0) * np.exp(RNG.normal(0, 0.05, keep.sum()))
        trip_sec = (miles / mph * 3600).astype(int)
        pay = 1.36 * miles + 0.583 * trip_sec / 60 * 1.07
        rows.append(pd.DataFrame({
            "zone": zid, "hour": hours[keep], "trips": n, "wait_n": wait_n,
            "wait_p50_sec": (wait_mean * 55).astype(int), "wait_p90_sec": (wait_mean * 100).astype(int),
            "wait_sum_sec": (wait_n * wait_mean * 60).astype(int), "miles": miles, "trip_sec": trip_sec,
            "fare": pay / 0.76, "driver_pay": pay, "tips": n * 1.1,
            "trips_cbd_fee": (n * np.where(hours[keep] >= FEE_START, 1.0 if z["fee_zone"] else 0.02, 0)).astype(int),
            "trips_airport": (n * (0.9 if zid in config.AIRPORT_ZONES else 0.03)).astype(int),
            "trips_shared_request": (n * 0.05).astype(int), "trips_uber": (n * 0.72).astype(int),
            "trips_lyft": (n * 0.28).astype(int),
        }))
    out = pd.concat(rows, ignore_index=True)
    for c in ["trips", "wait_n", "wait_p50_sec", "wait_p90_sec", "wait_sum_sec", "trip_sec", "trips_cbd_fee",
              "trips_airport", "trips_shared_request", "trips_uber", "trips_lyft"]:
        out[c] = out[c].astype("int64")
    out["zone"] = out["zone"].astype("int32")
    return out


def od_hour_month(zh: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    ids = zones.index.to_numpy()
    w = zones["level"].to_numpy() / zones["level"].sum()
    g = zh.assign(hour_of_day=zh["hour"].dt.hour, weekend=zh["hour"].dt.dayofweek >= 5).groupby(
        ["zone", "hour_of_day", "weekend"])[["trips", "driver_pay", "miles"]].sum().reset_index()
    rows = []
    for _, r in g.iterrows():
        k = RNG.multinomial(int(r["trips"]), w)
        nz = k > 0
        rows.append(pd.DataFrame({"pu_zone": int(r["zone"]), "do_zone": ids[nz], "hour_of_day": int(r["hour_of_day"]),
                                  "weekend": bool(r["weekend"]), "trips": k[nz].astype("int64"),
                                  "driver_pay": r["driver_pay"] * k[nz] / r["trips"], "miles": r["miles"] * k[nz] / r["trips"]}))
    out = pd.concat(rows, ignore_index=True)
    out["pu_zone"] = out["pu_zone"].astype("int32")
    out["do_zone"] = out["do_zone"].astype("int32")
    out["hour_of_day"] = out["hour_of_day"].astype("int32")
    return out


def trips_for(zh: pd.DataFrame, zones: pd.DataFrame, n: int) -> pd.DataFrame:
    p = zh["trips"].to_numpy() / zh["trips"].sum()
    pick = zh.iloc[RNG.choice(len(zh), n, p=p)]
    ids = zones.index.to_numpy()
    w = zones["level"].to_numpy() / zones["level"].sum()
    req = pick["hour"].to_numpy() + pd.to_timedelta(RNG.integers(0, 3600, n), unit="s")
    pu = pick["zone"].to_numpy()
    do = RNG.choice(ids, n, p=w)
    do = np.where(RNG.random(n) < 0.03, config.OUTSIDE_NYC_ZONE, do)
    miles = np.clip(RNG.gamma(2.0, zones.loc[pu, "miles_base"].to_numpy() / 2.0), 0.3, 60)
    trip_time = (miles / 9.0 * 3600 * np.exp(RNG.normal(0, 0.2, n))).astype(int).clip(120, 4 * 3600)
    wait = (RNG.gamma(3.0, zones.loc[pu, "wait_base"].to_numpy() * 20) + (pd.DatetimeIndex(req).hour < 5) * 90).astype(int)
    floor = 1.36 * miles + 0.583 * trip_time / 60
    prem = np.where(RNG.random(n) < 0.45, np.exp(RNG.normal(-1.5, 0.8, n)) * (1 + (miles < 2)), 0.0)
    pay = floor * (1 + prem)
    pickup = req + pd.to_timedelta(wait, unit="s")
    return pd.DataFrame({
        "platform": np.where(RNG.random(n) < 0.72, "Uber", "Lyft"), "request_datetime": req,
        "pickup_datetime": pickup, "dropoff_datetime": pickup + pd.to_timedelta(trip_time, unit="s"),
        "PULocationID": pu.astype("int32"), "DOLocationID": do.astype("int32"), "trip_miles": miles,
        "trip_time": trip_time.astype("int64"), "base_passenger_fare": pay / 0.75 + 2, "tolls": 0.0, "tips": 0.0,
        "driver_pay": pay, "wait_sec": wait.astype("int64"), "wait_valid": True,
        "cbd_fee": (pd.DatetimeIndex(req) >= FEE_START) & zones.loc[pu, "fee_zone"].to_numpy(),
        "is_airport": np.isin(pu, config.AIRPORT_ZONES), "shared_request": RNG.random(n) < 0.05,
    })


def main() -> None:
    if SYNTHETIC.exists():
        shutil.rmtree(SYNTHETIC)
    interim, sample = SYNTHETIC / "interim", SYNTHETIC / "sample"
    zones = pick_zones()
    for month in config.MONTHS:
        zh = zone_hour_month(month, zones)
        for name, df in [("zone_hour", zh), ("od_hour", od_hour_month(zh, zones))]:
            d = interim / name / f"month={month}"
            d.mkdir(parents=True)
            df.to_parquet(d / "part-0.parquet", index=False)
        d = sample / f"month={month}"
        d.mkdir(parents=True)
        tr = trips_for(zh, zones, 4000)
        tr.to_parquet(d / "part-0.parquet", index=False)
        q = interim / "quality"
        q.mkdir(parents=True, exist_ok=True)
        rows = int(zh["trips"].sum() * 1.001)
        report = {"month": month, "file_rows": rows, "rows": rows, "kept": int(zh["trips"].sum()),
                  "kept_with_valid_wait": int(zh["wait_n"].sum())}
        report |= {f"fail_{r}": int(rows * 1e-4) for r in config.RULES}
        (q / f"{month}.json").write_text(json.dumps(report))
        print(month, len(zh), "zone-hours")
    bk = zones[zones["borough"] == "Brooklyn"]
    zh_june = zone_hour_month("2025-06", bk)
    parts = []
    for day in ["2025-06-10", "2025-06-14"]:
        z = zh_june[zh_june["hour"].dt.date == pd.Timestamp(day).date()]
        parts.append(trips_for(z, bk, 2500))
    req = pd.concat(parts).sort_values("request_datetime").reset_index(drop=True)
    req.drop(columns=["cbd_fee"]).to_parquet(interim / "sim_requests.parquet", index=False)
    print(f"synthetic city written to {SYNTHETIC}")


if __name__ == "__main__":
    main()
