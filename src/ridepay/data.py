"""Loading the built tables. Notebooks read only these small tables, never the raw TLC files.

Two sources share one layout:

* ``real``: built from the TLC files by ``scripts/build_tables.py`` into ``data/interim`` and
  ``data/sample``;
* ``synthetic``: a small fake city in the same schema, made by ``scripts/make_synthetic.py`` into
  ``data/synthetic``. CI runs every notebook on it, so the code is tested without the 8 GB download.

``RIDEPAY_SYNTHETIC=1`` forces the synthetic source. Results from it never go to ``reports/``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from . import config

SYNTHETIC = config.DATA / "synthetic"


def synthetic_forced() -> bool:
    return os.environ.get("RIDEPAY_SYNTHETIC", "") in ("1", "true", "yes")


def real_data_available() -> bool:
    """True when the zone x hour table has been built and RIDEPAY_SYNTHETIC is not set."""
    return (config.INTERIM / "zone_hour").exists() and not synthetic_forced()


def source() -> str:
    return "real" if real_data_available() else "synthetic"


def interim_dir() -> Path:
    return config.INTERIM if source() == "real" else SYNTHETIC / "interim"


def sample_dir() -> Path:
    return config.SAMPLE if source() == "real" else SYNTHETIC / "sample"


def months() -> list[str]:
    """Months that have been fully built (a quality report is written last, so it marks completion)."""
    return sorted(p.stem for p in (interim_dir() / "quality").glob("*.json"))


def connect() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection with views over the built tables, for SQL in notebooks.

    Views: ``zone_hour``, ``od_hour``, ``trips`` (the 0.5% trip sample) and ``zones`` (taxi-zone
    names, borough and centroid in miles). Only completed months are included.
    """
    con = duckdb.connect()
    root = interim_dir().as_posix()
    done = months()
    month_list = ", ".join(f"'{m}'" for m in done) or "''"
    for view, folder in (("zone_hour", f"{root}/zone_hour"), ("od_hour", f"{root}/od_hour"),
                         ("trips", sample_dir().as_posix())):
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{folder}/*/*.parquet', "
                    f"hive_partitioning=true) WHERE month IN ({month_list})")
    cents = (config.DATA / "zones" / "zone_centroids.csv").as_posix()
    con.execute(f"CREATE VIEW zones AS SELECT * FROM read_csv_auto('{cents}')")
    return con


def quality_reports() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((interim_dir() / "quality").glob("*.json"))]
    return pd.DataFrame(rows).set_index("month")


def trip_sample(columns: list[str] | None = None, where: str | None = None) -> pd.DataFrame:
    cols = ", ".join(columns) if columns else "*"
    sql = f"SELECT {cols} FROM trips" + (f" WHERE {where}" if where else "")
    return connect().sql(sql).df()


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save a derived table (e.g. zone clusters) next to the built tables of the current source."""
    out = interim_dir() / "derived" / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(interim_dir() / "derived" / f"{name}.parquet")


def figure_path(name: str) -> Path | None:
    """Where to save a figure: reports/figures for real runs, nowhere for synthetic ones."""
    if source() != "real":
        return None
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    return config.FIGURES / f"{name}.png"


def savefig(fig, name: str) -> None:
    path = figure_path(name)
    if path is not None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
