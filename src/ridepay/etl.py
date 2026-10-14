"""PySpark pipeline: one monthly TLC file in, cleaned trips -> zone x hour table + a trip sample out.

Kept as small pure functions over DataFrames so each step is unit-tested on a tiny synthetic file.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config

COLUMNS = ["hvfhs_license_num", "request_datetime", "pickup_datetime", "dropoff_datetime", "PULocationID",
           "DOLocationID", "trip_miles", "trip_time", "base_passenger_fare", "tolls", "tips", "driver_pay",
           "shared_request_flag", "shared_match_flag", "airport_fee", "congestion_surcharge"]
OPTIONAL = {"cbd_congestion_fee": 0.0}   # only present from January 2025


def spark_session(app: str = "ridepay", memory: str = "4g") -> SparkSession:
    return (SparkSession.builder.appName(app).master("local[*]")
            .config("spark.driver.memory", memory)
            .config("spark.sql.session.timeZone", "UTC")   # TLC times are local NYC wall-clock; never shift them
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def standardize(df: DataFrame, extra: tuple[str, ...] = ()) -> DataFrame:
    """Select the columns used, add columns missing in older files, and fix timestamp types.

    The files store wall-clock New York times without a time zone (TIMESTAMP_NTZ in Spark). With the
    session time zone fixed to UTC, casting to TIMESTAMP keeps the same wall-clock values and lets us
    subtract and truncate them.
    """
    for col, default in OPTIONAL.items():
        if col not in df.columns:
            df = df.withColumn(col, F.lit(default).cast("double"))
    df = df.select(*extra, *COLUMNS, *OPTIONAL)
    for c in ("request_datetime", "pickup_datetime", "dropoff_datetime"):
        df = df.withColumn(c, F.col(c).cast("timestamp"))
    return df


def read_month(spark: SparkSession, path: str) -> DataFrame:
    return standardize(spark.read.parquet(str(path)))


def add_derived(df: DataFrame) -> DataFrame:
    """Wait time, the request hour, platform name and per-trip ratios used everywhere downstream."""
    platform = F.coalesce(*[F.when(F.col("hvfhs_license_num") == k, F.lit(v)) for k, v in config.PLATFORMS.items()],
                          F.lit("Other"))
    return (df
            .withColumn("wait_sec", F.col("pickup_datetime").cast("long") - F.col("request_datetime").cast("long"))
            .withColumn("hour", F.date_trunc("hour", "request_datetime"))
            .withColumn("platform", platform)
            .withColumn("cbd_fee", F.coalesce(F.col("cbd_congestion_fee"), F.lit(0.0)) > 0)
            .withColumn("is_airport", F.coalesce(F.col("airport_fee"), F.lit(0.0)) > 0)
            .withColumn("shared_request", F.col("shared_request_flag") == "Y"))


def flag_rules(df: DataFrame, month: str) -> DataFrame:
    """One boolean column per rule (True = passes); `keep` = passes every trip rule; `wait_valid` =
    passes every wait rule (wait metrics use only those trips)."""
    start = F.to_timestamp(F.lit(f"{month}-01 00:00:00"))
    end = F.add_months(start, 1)
    trip_sec = F.col("dropoff_datetime").cast("long") - F.col("pickup_datetime").cast("long")
    checks = {
        "request_in_month": (F.col("request_datetime") >= start) & (F.col("request_datetime") < end),
        "valid_zones": F.col("PULocationID").between(1, 263)
        & (F.col("DOLocationID").between(1, 263) | (F.col("DOLocationID") == config.OUTSIDE_NYC_ZONE)),
        "pickup_after_request": F.col("wait_sec") >= 0,
        "wait_under_2h": F.col("wait_sec") < config.MAX_WAIT_SEC,
        "dropoff_after_pickup": trip_sec > 0,
        "miles_positive": (F.col("trip_miles") > 0) & (F.col("trip_miles") <= config.MAX_MILES),
        "duration_ok": F.col("trip_time").between(config.MIN_TRIP_SEC, config.MAX_TRIP_SEC),
        "fare_positive": (F.col("base_passenger_fare") > 0) & (F.col("base_passenger_fare") <= config.MAX_FARE),
        "pay_positive": (F.col("driver_pay") > 0) & (F.col("driver_pay") <= config.MAX_PAY),
    }
    assert set(checks) == set(config.RULES), "every rule in config.RULES needs a check"
    for name, cond in checks.items():
        df = df.withColumn(f"ok_{name}", F.coalesce(cond, F.lit(False)))   # a null in a checked field fails
    def all_of(names):
        out = None
        for name in names:
            out = F.col(f"ok_{name}") if out is None else out & F.col(f"ok_{name}")
        return out

    return df.withColumn("keep", all_of(config.TRIP_RULES)).withColumn("wait_valid", all_of(config.WAIT_RULES))


def quality_report(flagged: DataFrame) -> dict:
    """Rows failing each rule (a row can fail several), rows kept, and kept rows with a valid wait."""
    aggs = [F.count(F.lit(1)).alias("rows"), F.sum(F.col("keep").cast("int")).alias("kept"),
            F.sum((F.col("keep") & F.col("wait_valid")).cast("int")).alias("kept_with_valid_wait")]
    aggs += [F.sum((~F.col(f"ok_{r}")).cast("int")).alias(f"fail_{r}") for r in config.RULES]
    row = flagged.agg(*aggs).first().asDict()
    return {k: int(v or 0) for k, v in row.items()}


def zone_hour(clean: DataFrame) -> DataFrame:
    """Marketplace health by pickup zone and request hour. Sums are kept (not only means) so the
    table can be re-aggregated to any coarser level without bias."""
    return (clean.groupBy(F.col("PULocationID").alias("zone"), "hour")
            .agg(F.count(F.lit(1)).alias("trips"),
                 F.sum(F.col("wait_valid").cast("int")).alias("wait_n"),
                 F.percentile_approx(F.when(F.col("wait_valid"), F.col("wait_sec")), 0.5, 1000).alias("wait_p50_sec"),
                 F.percentile_approx(F.when(F.col("wait_valid"), F.col("wait_sec")), 0.9, 1000).alias("wait_p90_sec"),
                 F.sum(F.when(F.col("wait_valid"), F.col("wait_sec"))).alias("wait_sum_sec"),
                 F.sum("trip_miles").alias("miles"),
                 F.sum("trip_time").alias("trip_sec"),
                 F.sum("base_passenger_fare").alias("fare"),
                 F.sum("driver_pay").alias("driver_pay"),
                 F.sum("tips").alias("tips"),
                 F.sum(F.col("cbd_fee").cast("int")).alias("trips_cbd_fee"),
                 F.sum(F.col("is_airport").cast("int")).alias("trips_airport"),
                 F.sum(F.col("shared_request").cast("int")).alias("trips_shared_request"),
                 F.sum((F.col("platform") == "Uber").cast("int")).alias("trips_uber"),
                 F.sum((F.col("platform") == "Lyft").cast("int")).alias("trips_lyft")))


def od_hour(clean: DataFrame) -> DataFrame:
    """Trips by pickup zone, drop-off zone and hour of day x day type: where drivers end up after a trip."""
    return (clean.withColumn("hour_of_day", F.hour("request_datetime"))
            .withColumn("weekend", F.dayofweek("request_datetime").isin(1, 7))
            .groupBy(F.col("PULocationID").alias("pu_zone"), F.col("DOLocationID").alias("do_zone"),
                     "hour_of_day", "weekend")
            .agg(F.count(F.lit(1)).alias("trips"), F.sum("driver_pay").alias("driver_pay"),
                 F.sum("trip_miles").alias("miles")))


SAMPLE_COLUMNS = ["platform", "request_datetime", "pickup_datetime", "dropoff_datetime", "PULocationID",
                  "DOLocationID", "trip_miles", "trip_time", "base_passenger_fare", "tolls", "tips", "driver_pay",
                  "wait_sec", "wait_valid", "cbd_fee", "is_airport", "shared_request"]


def trip_sample(clean: DataFrame, fraction: float = config.SAMPLE_FRACTION,
                seed: int = config.SAMPLE_SEED) -> DataFrame:
    """A fixed random sample of clean trips for trip-level modelling (phase 3 onward)."""
    return clean.sample(withReplacement=False, fraction=fraction, seed=seed).select(*SAMPLE_COLUMNS)
