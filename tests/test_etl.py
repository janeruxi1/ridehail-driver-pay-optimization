"""The cleaning rules and aggregations, on a tiny hand-made month with one known defect per row."""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from ridepay import config, etl

T0 = datetime(2025, 1, 10, 8, 0, 0)


def trip(**kw):
    base = dict(hvfhs_license_num="HV0003", request_datetime=T0, pickup_datetime=T0 + timedelta(minutes=5),
                dropoff_datetime=T0 + timedelta(minutes=25), PULocationID=161, DOLocationID=236, trip_miles=3.0,
                trip_time=1200, base_passenger_fare=25.0, tolls=0.0, tips=2.0, driver_pay=18.0,
                shared_request_flag="N", shared_match_flag="N", airport_fee=0.0, congestion_surcharge=2.75,
                cbd_congestion_fee=1.5)
    base.update(kw)
    return base


ROWS = {
    "good": trip(),
    "good_lyft_other_zone": trip(hvfhs_license_num="HV0005", PULocationID=7, cbd_congestion_fee=0.0),
    "good_outside_nyc": trip(PULocationID=7, DOLocationID=265, cbd_congestion_fee=0.0),
    "request_in_month": trip(request_datetime=datetime(2024, 12, 31, 23, 59),
                             pickup_datetime=datetime(2025, 1, 1, 0, 3)),
    "valid_zones": trip(PULocationID=264),
    "pickup_after_request": trip(pickup_datetime=T0 - timedelta(minutes=1)),
    "wait_under_2h": trip(pickup_datetime=T0 + timedelta(hours=3), dropoff_datetime=T0 + timedelta(hours=3, minutes=20)),
    "dropoff_after_pickup": trip(dropoff_datetime=T0 + timedelta(minutes=5)),
    "miles_positive": trip(trip_miles=0.0),
    "duration_ok": trip(trip_time=30),
    "fare_positive": trip(base_passenger_fare=-4.0),
    "pay_positive": trip(driver_pay=None),
}


@pytest.fixture(scope="module")
def flagged(spark, tmp_path_factory):
    path = tmp_path_factory.mktemp("tlc") / "month.parquet"
    df = pd.DataFrame(ROWS.values())
    df.insert(0, "case", list(ROWS))
    df.to_parquet(path, index=False, coerce_timestamps="us")
    raw = etl.standardize(spark.read.parquet(str(path)), extra=("case",))
    return etl.flag_rules(etl.add_derived(raw), "2025-01").toPandas().set_index("case")


def test_each_bad_row_fails_exactly_its_rule(flagged):
    for case in ROWS:
        failed = {r for r in config.RULES if not flagged.loc[case, f"ok_{r}"]}
        row = flagged.loc[case]
        if case.startswith("good"):
            assert failed == set() and row["keep"] and row["wait_valid"], case
        elif case in config.WAIT_RULES:   # trip kept, wait treated as missing
            assert failed == {case} and row["keep"] and not row["wait_valid"], (case, failed)
        else:
            assert case in failed and not row["keep"], (case, failed)


def test_trips_ending_outside_nyc_are_kept(flagged):
    assert flagged.loc["good_outside_nyc", "keep"]


def test_derived_columns(flagged):
    g = flagged.loc["good"]
    assert g["wait_sec"] == 300 and g["platform"] == "Uber" and bool(g["cbd_fee"])
    assert flagged.loc["good_lyft_other_zone", "platform"] == "Lyft"
    assert not flagged.loc["good_lyft_other_zone", "cbd_fee"]


def test_quality_report_and_zone_hour(spark, flagged):
    sdf = spark.createDataFrame(flagged.reset_index().drop(columns=["case"]))
    rep = etl.quality_report(sdf)
    assert rep["rows"] == len(ROWS) and rep["kept"] == 5 and rep["kept_with_valid_wait"] == 3
    zh = etl.zone_hour(sdf.filter("keep")).toPandas().set_index("zone")
    assert set(zh.index) == {161, 7}
    z = zh.loc[161]   # the good trip plus the two kept trips whose wait is invalid
    assert z["trips"] == 3 and z["wait_n"] == 1 and z["wait_p50_sec"] == 300 and z["wait_sum_sec"] == 300
    assert z["driver_pay"] == 54.0 and z["trips_cbd_fee"] == 3


def test_missing_cbd_column_defaults_to_zero(spark, tmp_path):
    df = pd.DataFrame([trip()]).drop(columns=["cbd_congestion_fee"])
    df.to_parquet(tmp_path / "old.parquet", index=False, coerce_timestamps="us")
    out = etl.add_derived(etl.read_month(spark, tmp_path / "old.parquet")).toPandas()
    assert out.loc[0, "cbd_congestion_fee"] == 0.0 and not out.loc[0, "cbd_fee"]
