"""Taxi-zone geometry: centroids (for distances between zones) and simplified outlines (for maps).

The TLC publishes zone outlines as a shapefile in New York State Plane coordinates (feet). We convert
to miles on that flat projection, which is accurate to well under 1% across the city, so distances are
plain Euclidean. The outputs are small and committed to the repo (`data/zones/`), so notebooks and CI
never need the shapefile.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ZONES_DIR = config.DATA / "zones"
FEET_PER_MILE = 5280.0


def build_zone_files(shp_path: str | Path, out_dir: Path = ZONES_DIR, tolerance_ft: float = 150.0) -> pd.DataFrame:
    """Read the TLC shapefile; write zone_centroids.csv and zone_outlines.json. Needs pyshp and shapely."""
    import shapefile
    from shapely.geometry import shape
    from shapely.ops import unary_union

    reader = shapefile.Reader(str(shp_path))
    parts: dict[int, list] = {}
    attrs: dict[int, tuple[str, str]] = {}
    for sr in reader.shapeRecords():
        rec = sr.record.as_dict()
        zid = int(rec["LocationID"])
        parts.setdefault(zid, []).append(shape(sr.shape.__geo_interface__))
        attrs[zid] = (rec["zone"], rec["borough"])
    rows, outlines = [], {}
    for zid in sorted(parts):
        geom = unary_union(parts[zid])                      # a few zones are split over several records
        c = geom.representative_point() if not geom.centroid.within(geom) else geom.centroid
        rows.append({"zone": zid, "zone_name": attrs[zid][0], "borough": attrs[zid][1],
                     "x_mi": c.x / FEET_PER_MILE, "y_mi": c.y / FEET_PER_MILE,
                     "area_sq_mi": geom.area / FEET_PER_MILE**2})
        simple = geom.simplify(tolerance_ft)
        polys = getattr(simple, "geoms", [simple])
        outlines[zid] = [[[round(x / FEET_PER_MILE, 3), round(y / FEET_PER_MILE, 3)] for x, y in p.exterior.coords]
                         for p in polys if p.area > 0]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "zone_centroids.csv", index=False, float_format="%.4f")
    (out_dir / "zone_outlines.json").write_text(json.dumps(outlines, separators=(",", ":")))
    return df


def centroids() -> pd.DataFrame:
    return pd.read_csv(ZONES_DIR / "zone_centroids.csv")


def outlines() -> dict[int, list]:
    return {int(k): v for k, v in json.loads((ZONES_DIR / "zone_outlines.json").read_text()).items()}


def distance_matrix(zone_ids, cents: pd.DataFrame | None = None) -> np.ndarray:
    """Straight-line miles between zone centroids, in the order of `zone_ids`."""
    cents = centroids() if cents is None else cents
    xy = cents.set_index("zone").loc[list(zone_ids), ["x_mi", "y_mi"]].to_numpy()
    return np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])


def plot_zones(ax, values: dict[int, float] | pd.Series, cmap="viridis", vmin=None, vmax=None,
               missing="#dddddd", categorical: bool = False, edge="white"):
    """Choropleth of taxi zones on a matplotlib axis. Returns the ScalarMappable (or None if categorical)."""
    import matplotlib as mpl
    from matplotlib.collections import PolyCollection

    shapes = outlines()
    values = dict(values)
    polys, colors = [], []
    vals = np.array([v for v in values.values() if v is not None and np.isfinite(v)], dtype=float)
    norm = mpl.colors.Normalize(vmin if vmin is not None else (vals.min() if len(vals) else 0),
                                vmax if vmax is not None else (vals.max() if len(vals) else 1))
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    for zid, rings in shapes.items():
        v = values.get(zid)
        if v is None or (not categorical and not np.isfinite(v)):
            col = missing
        else:
            col = cm(int(v) % cm.N) if categorical else cm(norm(v))
        for ring in rings:
            polys.append(ring)
            colors.append(col)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=edge, linewidths=0.2))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    if categorical:
        return None
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    return sm
