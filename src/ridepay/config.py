"""Paths, months and the data-cleaning rules, in one place so every script and notebook agrees."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"                    # monthly TLC files; deleted after processing
INTERIM = DATA / "interim"            # zone x hour tables and quality reports (small)
SAMPLE = DATA / "sample"              # a fixed 0.5% sample of clean trips, for trip-level models
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"

# Eighteen months: January 2024 to June 2025. The congestion fee started on 5 January 2025, so there
# are twelve months before it and six after; January-June 2024 gives the same calendar weeks one year
# earlier, which separates the fee's effect from winter and spring seasonality (phase 4).
MONTHS = [f"2024-{m:02d}" for m in range(1, 13)] + [f"2025-{m:02d}" for m in range(1, 7)]
CBD_FEE_START = "2025-01-05"
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_{month}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

PLATFORMS = {"HV0002": "Juno", "HV0003": "Uber", "HV0004": "Via", "HV0005": "Lyft"}

# Cleaning rules, in two groups. Each rule is counted separately in the quality report.
# TRIP_RULES: a trip failing any of these is dropped.
# WAIT_RULES: a trip failing these is kept (its fare, pay and distance are fine), but its wait time
# is treated as missing. About 1% of trips have a pickup logged minutes *before* the request, which
# makes the wait meaningless without making the trip wrong.
TRIP_RULES = {
    "request_in_month": "request time falls in the file's month",
    "valid_zones": "pickup in a real taxi zone (1-263); drop-off in one, or outside NYC (265)",
    "dropoff_after_pickup": "drop-off after pickup",
    "miles_positive": "trip miles above 0 and at most 100",
    "duration_ok": "trip time between 1 minute and 4 hours",
    "fare_positive": "rider fare above $0 and at most $500",
    "pay_positive": "driver pay above $0 and at most $500",
}
WAIT_RULES = {
    "pickup_after_request": "pickup is at or after the request",
    "wait_under_2h": "wait from request to pickup under 2 hours",
}
RULES = {**TRIP_RULES, **WAIT_RULES}
OUTSIDE_NYC_ZONE = 265
MAX_WAIT_SEC = 2 * 3600
MAX_MILES = 100.0
MIN_TRIP_SEC, MAX_TRIP_SEC = 60, 4 * 3600
MAX_FARE = 500.0
MAX_PAY = 500.0
SAMPLE_FRACTION = 0.005
SAMPLE_SEED = 20250105

# Hour bands used for segments, guardrails (C3) and pay policy. Keys are request hours 0-23.
HOUR_BANDS = {"night": range(0, 6), "am_peak": range(6, 10), "midday": range(10, 16),
              "pm_peak": range(16, 20), "evening": range(20, 24)}
BAND_OF_HOUR = {h: b for b, hours in HOUR_BANDS.items() for h in hours}

# Airport zones (Newark, JFK, LaGuardia) are their own segment: they run queue systems, so they are
# not clustered with ordinary zones.
AIRPORT_ZONES = (1, 132, 138)
