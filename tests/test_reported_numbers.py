"""README, decision memo and model card quote numbers from reports/metrics; this checks them.

Each expected string is formatted from the committed records, so a re-run that moves a number and a
document that was not updated fail here instead of reaching a reader.
"""
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"
NAMES = ["01_marketplace_health", "02_demand_forecast", "03_long_wait", "04_congestion_fee", "05_pay_to_accept",
         "06_policy_simulation", "07_switchback"]

pytestmark = pytest.mark.skipif(not all((M / f"{n}.json").exists() for n in NAMES), reason="records not present")


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


def r(x, digits):
    """Round half up (as people do), not half to even."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def expected():
    m1, m2, m3, m4, m5, m6, m7 = (rec(n) for n in NAMES)
    H = m6["held_summary"]
    rob, today, flo, pnt = H["Optimized (robust)"], H["Today"], H["Floor only"], H["Optimized (point)"]
    eff = {(e["comparison"], e["kpi"]): e for e in m6["sim"]["effects"]}
    pol = "Robust policy − today's pay"
    acc, mar, p90 = (eff[(pol, k)] for k in ("first_offer_acceptance", "margin_per_request", "wait_p90_min"))
    batch = eff[("Batched − greedy matching", "wait_mean_min")]
    bor = {b["borough"]: b for b in m6["by_borough"]}
    est = {e["method"]: e for e in m5["estimates"]}
    naive, iv = est["Naive (no controls)"], est["Randomized pay test (IV)"]
    pp = lambda v: 100 * 0.0953101798 * v  # noqa: E731  (log 1.1: per 10% more pay)
    lw = m3["policy_cost"]
    model_cost = min(v for k, v in lw.items() if k.startswith("Model, threshold"))
    cut = 1 - model_cost / min(lw["Do nothing"], lw["Flag every request"])
    mph = [x for x in m4["yoy_did"] if x["outcome"] == "mph"][0]
    chosen = [d for d in m7["design"] if (d["block_h"], d["washout"]) == (m7["chosen"]["block_h"], m7["chosen"]["washout"])][0]
    val = m6["sim"]["validation"]
    sim_p90 = sum(v["sim p90"] for v in val) / len(val)
    real_p90 = sum(v["real p90 (not targeted)"] for v in val) / len(val)
    shared = {
        f"{r(m1['floor']['premium_share_of_pay_pct'], 1)}%",
        f"+{r(rob['margin_vs_today_pct'], 1)}%",
        f"{r(100 * acc['mean'], 1)} points",
        f"{r(-batch['mean'], 1)} min",
        f"{r(pp(naive['estimate']), 1)}",
        f"{r(pp(naive['truth_of_estimand']), 1)}",
    }
    readme = shared | {
        f"**{r(m1['floor']['median_ratio'], 3)}×**", f"{r(m1['floor']['at_floor_pct'], 0)}% of trips",
        f"correlation with hourly trips {r(m1['weekly']['corr_trips_wait'], 2)}",
        f"**{r(100 * m2['wmape_levels']['LightGBM']['zone × hour'], 1)}%**",
        f"{r(100 * m2['wmape_levels']['4-week mean']['zone × hour'], 1)}% (4-week mean)",
        f"{r(100 * m2['wmape_levels']['Seasonal naive (last week)']['zone × hour'], 1)}% (seasonal naive)",
        f"{r(100 * m2['poisson_noise_floor_zone_hour'], 1)}%",
        f"ROC AUC **{r(m3['metrics']['LightGBM']['ROC AUC'], 3)}**", f"cuts expected cost {r(100 * cut, 0)}%",
        f"**{r(mph['estimate'], 2)} mph faster ({r(mph['pct_effect'], 1)}%)**",
        f"\"earn\" +{r(flo['margin_vs_today_pct'], 0)}%",
        f"${r(-mar['mean'], 2)} of margin", f"predicts ${r(-bor['Brooklyn']['margin change'], 2)} for Brooklyn",
        f"p90 −{r(-p90['mean'], 2)} min, 95% interval −{r(-p90['lo'], 2)} to +{r(p90['hi'], 2)}",
        f"**{chosen['weeks_for_expected']} weeks**",
        f"{m1['quality']['rows']:,} trips", f"{r(m1['quality']['kept_pct'], 2)}%",
        f"p90 {r(sim_p90, 1)} vs {r(real_p90, 1)} min",
    }
    memo = shared | {
        f"{r(pp(iv['estimate']), 1)} points", f"truth there: {r(pp(iv['truth_of_estimand']), 1)}",
        f"${r(today['margin_per_request'], 2)}", f"**${r(rob['margin_per_request'], 2)} (+{r(rob['margin_vs_today_pct'], 1)}%)**",
        f"{r(100 * today['first_offer_acceptance'], 1)}% | {r(100 * rob['first_offer_acceptance'], 1)}%",
        f"{r(100 * today['completion'], 1)}% | {r(100 * rob['completion'], 1)}%",
        f"{r(today['extra_wait_sec'], 1)} | {r(rob['extra_wait_sec'], 1)}",
        f"${r(today['pay_per_trip'], 2)} | ${r(rob['pay_per_trip'], 2)}",
        f"${r(today['earnings_per_hour'], 2)} | ${r(rob['earnings_per_hour'], 2)} "
        f"(-{r(100 * (1 - rob['earnings_per_hour'] / today['earnings_per_hour']), 1)}%)",
        f"Manhattan (+{r(bor['Manhattan']['margin change'], 2)}", f"Brooklyn -{r(-bor['Brooklyn']['margin change'], 2)}",
        f"Bronx\n   -{r(-bor['Bronx']['margin change'], 2)}",
        f"margin -{r(-mar['mean'], 2)} $", f"p90 -{r(-p90['mean'], 2)} min, 95% interval",
        f"Looks like +{r(flo['margin_vs_today_pct'], 0)}% margin", f"{r(100 * flo['first_offer_acceptance'], 1)}%",
        f"by {r(100 * (1 - flo['earnings_per_hour'] / today['earnings_per_hour']), 0)}%",
        f"{r(pnt['margin_vs_today_pct'] - rob['margin_vs_today_pct'], 1)} percentage points",
        f"only {round(20 * m6['guardrails_in_draws']['point']['C3_wait'])} of the 20",
        f"${r(chosen['mde_4wk'], 3)} in 4 weeks", f"expected $+{r(m7['expected_effect'], 3)}",
        f"**{chosen['weeks_for_expected']} weeks**", f"{r(m7['wait_mde_min'], 2)} min", f"{m7['regions']} regions",
    }
    card = {
        f"+{r(rob['margin_vs_today_pct'], 1)}%", f"{r(100 * (1 - m6['guardrails_in_draws']['point']['C3_wait']), 0)}% of",
        f"+{r(100 * acc['mean'], 1)} points", f"−${r(-mar['mean'], 2)} per request",
        f"−{r(-p90['mean'], 2)} min, 95% interval −{r(-p90['lo'], 2)} to +{r(p90['hi'], 2)}",
        f"{round(m5['hierarchical']['coverage90'] * 20)} of 20 segments",
    }
    return readme, memo, card


@pytest.mark.parametrize("doc,index", [("README.md", 0), ("reports/decision_memo.md", 1), ("reports/model_card.md", 2)])
def test_document_numbers_match_records(doc, index):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = sorted(s for s in expected()[index] if s not in text)
    assert not missing, f"{doc} does not quote: {missing}"
