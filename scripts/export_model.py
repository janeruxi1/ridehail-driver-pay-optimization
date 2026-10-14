"""Export the quote model artifact (models/quote_model.json) from the phase 1, 5 and 6 outputs.

    python scripts/export_model.py

The artifact holds: zone -> cluster, zone drop-offs per pickup, the acceptance model's posterior means
by segment, the difficulty quintile edges and the policy's premium per cell. It is small, readable
and versioned with the code, so the API image needs no data files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridepay import accept, config, data, features  # noqa: E402

OUT = config.ROOT / "models" / "quote_model.json"


def main() -> None:
    if data.source() != "real":
        out = data.SYNTHETIC / "quote_model.json"
        print("synthetic source: writing the artifact to", out)
    else:
        out = OUT
    info = features.zone_info()
    post = np.load(data.interim_dir() / "derived" / "accept_posterior.npz")
    pol = data.load_table("pay_policy")
    gamma = post["gamma"].mean(axis=0)
    edges = data.load_table("difficulty_edges")["edge"].tolist()   # saved by phase 6 (fit window)
    artifact = {
        "model_version": "",
        "created": date.today().isoformat(),
        "data_months": [data.months()[0], data.months()[-1]],
        "description": "Driver pay quote: floor x (1 + premium per cell); acceptance model = hierarchical logit "
                       "(posterior means) fitted on simulated decisions over real trips. Scenario, not a real platform.",
        "zone_cluster": {int(z): c for z, c in info["cluster_name"].items() if c != "Sparse"},
        "zone_arrivals_per_pickup": {int(z): round(float(v), 4) for z, v in info["arrivals_per_pickup"].dropna().items()},
        "segments": [str(s) for s in post["segments"]],
        "alpha": [round(float(v), 5) for v in post["alpha"].mean(axis=0)],
        "beta": [round(float(v), 5) for v in post["beta"].mean(axis=0)],
        "gamma": {k: round(float(v), 5) for k, v in zip(accept.FEATURES, gamma)},
        "difficulty_edges": [round(e, 5) for e in edges],
        "premium": {r.cell: float(r.premium_robust) for r in pol.itertuples()},
    }
    body = json.dumps({k: v for k, v in artifact.items() if k not in ("model_version", "created")}, sort_keys=True)
    artifact["model_version"] = "ridepay-" + hashlib.sha256(body.encode()).hexdigest()[:10]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {out} ({artifact['model_version']}, {len(artifact['premium'])} cells)")


if __name__ == "__main__":
    main()
