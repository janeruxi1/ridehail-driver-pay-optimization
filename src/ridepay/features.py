"""The trip frame shared by phases 5 and 6: sampled real trips with their pay floor, the pay actually
offered, segment, and the live state of the pickup zone in the hour before the request."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data
from .pay import floor_pay


def zone_info(con=None) -> pd.DataFrame:
    """Per zone: drop-offs per pickup (where drivers get stranded) and cluster."""
    con = con or data.connect()
    info = con.sql("""
        WITH a AS (SELECT do_zone AS zone, SUM(trips) AS dropoffs FROM od_hour GROUP BY 1),
             d AS (SELECT pu_zone AS zone, SUM(trips) AS pickups FROM od_hour GROUP BY 1)
        SELECT zone, dropoffs / pickups AS arrivals_per_pickup FROM a JOIN d USING (zone) ORDER BY zone
    """).df().set_index("zone")
    clusters = data.load_table("zone_clusters").set_index("zone")
    return clusters[["cluster", "cluster_name", "borough"]].join(info)


def trip_frame(months: list[str] | None = None, con=None) -> pd.DataFrame:
    """Sampled trips (valid or not, every trip has a pay offer) with segment and context features."""
    con = con or data.connect()
    where = f"WHERE t.month IN ({', '.join(repr(m) for m in months)})" if months else ""
    df = con.sql(f"""
        WITH zone_prev AS (
            SELECT zone, hour + INTERVAL 1 HOUR AS hour, wait_sum_sec / NULLIF(wait_n, 0) / 60 AS prev_wait_min
            FROM zone_hour)
        SELECT t.month, t.request_datetime, t.PULocationID, t.DOLocationID, t.trip_miles, t.trip_time,
               t.base_passenger_fare AS fare, t.driver_pay, t.wait_sec, t.wait_valid, t.is_airport, t.shared_request,
               hour(t.request_datetime) AS hod, isodow(t.request_datetime) - 1 AS dow, zp.prev_wait_min
        FROM trips t
        LEFT JOIN zone_prev zp ON zp.zone = t.PULocationID AND zp.hour = date_trunc('hour', t.request_datetime)
        {where}
        ORDER BY ALL
    """).df()
    info = zone_info(con)
    df["cluster_name"] = df["PULocationID"].map(info["cluster_name"]).fillna("Sparse")
    df = df[df["cluster_name"] != "Sparse"].copy()              # too few trips to price; served by the default rule
    df["band"] = df["hod"].map(config.BAND_OF_HOUR)
    df["prev_wait_min"] = df["prev_wait_min"].fillna(df["prev_wait_min"].median())
    df["floor"] = floor_pay(df["trip_miles"], df["trip_time"] / 60, df["request_datetime"], df["DOLocationID"])
    df["offer_sq"] = np.maximum(df["driver_pay"], df["floor"])  # today's offer: real pay, never below the floor
    df["minutes"] = df["trip_time"] / 60
    return df.reset_index(drop=True), info
