"""Extract every trip request picked up in Brooklyn on two days of June 2025, at full resolution, for
the dispatch simulation (phase 6). The 0.5% sample is too thin to replay a marketplace; the full
record of one borough on two days is about 300,000 requests and a few MB.

    python scripts/extract_sim_requests.py            # downloads June 2025 again if it isn't on disk
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from ridepay import config, etl  # noqa: E402
from ridepay.download import download  # noqa: E402

MONTH = "2025-06"
DAYS = ["2025-06-10", "2025-06-14"]          # a Tuesday and a Saturday
BOROUGH = "Brooklyn"
OUT = config.INTERIM / "sim_requests.parquet"


def main(keep_raw: bool = False) -> None:
    raw = download(config.URL.format(month=MONTH), config.RAW / f"fhvhv_tripdata_{MONTH}.parquet")
    zones = pd.read_csv(config.DATA / "zones" / "zone_centroids.csv")
    pu_zones = zones.loc[zones["borough"] == BOROUGH, "zone"].tolist()
    spark = etl.spark_session("extract-sim", memory="4g")
    spark.sparkContext.setLogLevel("ERROR")
    df = etl.flag_rules(etl.add_derived(etl.read_month(spark, raw)), MONTH).filter("keep")
    day = F.to_date("request_datetime").cast("string")
    sel = (df.filter(F.col("PULocationID").isin(pu_zones) & day.isin(DAYS))
           .select("request_datetime", "pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID",
                   "trip_miles", "trip_time", "base_passenger_fare", "driver_pay", "tips", "wait_sec", "wait_valid",
                   "platform", "shared_request", "is_airport"))
    pdf = sel.toPandas().sort_values("request_datetime").reset_index(drop=True)
    spark.stop()
    pdf.to_parquet(OUT, index=False)
    print(f"{len(pdf):,} requests written to {OUT}")
    if not keep_raw:
        raw.unlink()


if __name__ == "__main__":
    main(keep_raw="--keep-raw" in sys.argv)
