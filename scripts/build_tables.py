"""Build the project's tables from the NYC TLC High Volume FHV trip records.

For each month: download (if needed) -> clean with PySpark -> write the zone x hour table, the
origin-destination table, a 0.5% trip sample and a quality report -> delete the raw file unless
--keep-raw. Only the small outputs stay on disk (about 10-20 MB per month).

    python scripts/build_tables.py                    # all months in config.MONTHS
    python scripts/build_tables.py --months 2024-07   # one month
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridepay import config, etl  # noqa: E402
from ridepay.download import download, parquet_rows  # noqa: E402


def process_month(spark, month: str, keep_raw: bool) -> dict:
    out_dir = config.INTERIM
    done = out_dir / "quality" / f"{month}.json"
    if done.exists():
        print(month, "already built")
        return json.loads(done.read_text())
    t0 = time.time()
    raw = download(config.URL.format(month=month), config.RAW / f"fhvhv_tripdata_{month}.parquet")
    flagged = etl.flag_rules(etl.add_derived(etl.read_month(spark, raw)), month).cache()
    report = {"month": month, "file_rows": parquet_rows(raw), **etl.quality_report(flagged)}
    clean = flagged.filter("keep")
    etl.zone_hour(clean).write.mode("overwrite").parquet(str(out_dir / "zone_hour" / f"month={month}"))
    etl.od_hour(clean).write.mode("overwrite").parquet(str(out_dir / "od_hour" / f"month={month}"))
    etl.trip_sample(clean).write.mode("overwrite").parquet(str(config.SAMPLE / f"month={month}"))
    flagged.unpersist()
    report["seconds"] = round(time.time() - t0, 1)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(report, indent=2))
    if not keep_raw:
        raw.unlink()
    print(month, report["rows"], "rows,", report["kept"], "kept,", report["seconds"], "s")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=config.MONTHS)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--memory", default="5g")
    args = ap.parse_args()
    download(config.ZONE_LOOKUP_URL, config.DATA / "taxi_zone_lookup.csv")
    spark = etl.spark_session(memory=args.memory)
    spark.sparkContext.setLogLevel("ERROR")
    for m in args.months:
        process_month(spark, m, args.keep_raw)
    spark.stop()


if __name__ == "__main__":
    main()
