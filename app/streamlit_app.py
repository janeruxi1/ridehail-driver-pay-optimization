"""Interactive demo: driver pay in a ride-hailing marketplace (NYC TLC data, 2024–2025).

Runs only on small files committed to the repository (model artifact, metric records, zone map),
so it deploys without the raw trip data.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ridepay import demo, geo, plots  # noqa: E402
from ridepay.quote import QuoteModel  # noqa: E402

METRICS = ROOT / "reports" / "metrics"
SERIES = plots.SERIES
BAND_LABEL = {"night": "Night (0–6)", "am_peak": "AM peak (6–10)", "midday": "Midday (10–16)",
              "pm_peak": "PM peak (16–20)", "evening": "Evening (20–24)"}
REPO = "https://github.com/janeruxi1/ridehail-driver-pay-optimization"

st.set_page_config(page_title="Driver pay pricing", page_icon="🚕", layout="wide")


def money(x: float, digits: int = 2) -> str:
    """Dollar amount with the $ escaped, so Streamlit's Markdown doesn't read it as maths."""
    return f"\\${x:,.{digits}f}" if x >= 0 else f"−\\${-x:,.{digits}f}"


@st.cache_data
def load_metrics() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in METRICS.glob("*.json")}


@st.cache_resource
def load_model() -> QuoteModel:
    return QuoteModel.load(ROOT / "models" / "quote_model.json")


@st.cache_data
def load_zones() -> pd.DataFrame:
    return geo.centroids().set_index("zone")


M = load_metrics()
model = load_model()
zones = load_zones()
m1, m2, m3, m4, m5, m6, m7 = (M[k] for k in ["01_marketplace_health", "02_demand_forecast", "03_long_wait",
                                              "04_congestion_fee", "05_pay_to_accept", "06_policy_simulation",
                                              "07_switchback"])

st.title("Driver pay in a ride-hailing marketplace")
st.markdown(
    f"What should the platform offer a driver for each trip? Built on **{m1['quality']['rows'] / 1e6:,.0f} million real "
    "New York City trips** (TLC records, Jan 2024 – Jun 2025). Trips, fares, pay and waits are real; whether a driver "
    f"accepted an offer is simulated with a known truth, so the methods can be checked. [Code and notebooks]({REPO})"
)

tab_quote, tab_policy, tab_market, tab_test = st.tabs(
    ["Quote a trip", "Pay policy", "Marketplace", "Rollout test"])

# ------------------------------------------------------------------------------------------ quote
with tab_quote:
    st.subheader("Recommended pay for one trip")
    st.caption("Pay = New York's minimum for the trip's miles and minutes, plus a premium chosen per zone group, "
               "hour band and trip difficulty. This is what the quote service (`POST /quote`) returns.")
    named = zones[zones.index.isin(range(1, 264))].copy()
    named["label"] = named["zone_name"] + " (" + named["borough"] + ")"
    options = named.sort_values("label")
    c1, c2, c3 = st.columns(3)
    with c1:
        pu = st.selectbox("Pickup zone", options.index, format_func=lambda z: options.loc[z, "label"],
                          index=list(options.index).index(61))
        do_choices = list(options.index) + [265]
        do = st.selectbox("Drop-off zone", do_choices,
                          format_func=lambda z: "Outside NYC" if z == 265 else options.loc[z, "label"],
                          index=do_choices.index(17))
    with c2:
        day = st.date_input("Date", date(2025, 6, 14), min_value=date(2024, 1, 1), max_value=date(2026, 12, 31))
        hour = st.slider("Request hour", 0, 23, 2)
        wait = st.slider("Mean rider wait in the pickup zone last hour (min)", 1.0, 15.0, 4.4, 0.1,
                         help="Live supply signal: long recent waits mean drivers are scarce there.")
    with c3:
        miles = st.number_input("Trip miles", 0.2, 60.0, 0.8, 0.1)
        minutes = st.number_input("Trip minutes", 1.0, 180.0, 6.0, 1.0)
        fare = st.number_input("Rider fare ($)", 1.0, 300.0, 12.0, 0.5)

    when = datetime.combine(day, time(hour, 15))
    q = model.quote(int(pu), int(do), when, miles, minutes, fare, wait)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Minimum pay (floor)", f"${q.floor_pay:,.2f}")
    k2.metric("Recommended pay", f"${q.recommended_pay:,.2f}", f"+{q.premium_pct:.1f}% premium", delta_color="off")
    if q.accept_probability is None:
        k3.metric("First-offer acceptance", "n/a")
        k4.metric("Expected margin", "n/a")
        st.info("This pickup zone has too few trips to have a segment; it is paid the floor by default.")
    else:
        k3.metric("First-offer acceptance", f"{q.accept_probability:.0%}")
        k4.metric("Expected margin per request", f"${q.expected_margin:,.2f}")
        cl, band = q.segment.split("|")
        st.markdown(f"Segment: **{cl}**, **{BAND_LABEL[band]}**, difficulty quintile **{q.difficulty_quintile}** of 5 "
                    "(short trips, destinations where drivers get stranded and areas short of drivers are harder).")
        curve = demo.offer_curve(model, int(pu), int(do), when, miles, minutes, fare, wait)
        ca, cb = st.columns(2)
        for col, ycol, title, fmt, color in [
                (ca, "accept_probability", "First-offer acceptance", ".0%", SERIES[0]),
                (cb, "expected_margin", "Expected margin per request ($)", ".2f", SERIES[1])]:
            fig = go.Figure(go.Scatter(x=curve["premium_pct"], y=curve[ycol], mode="lines+markers",
                                       line=dict(color=color, width=2), marker=dict(size=5),
                                       hovertemplate="premium %{x:.1f}%<br>%{y:" + fmt + "}<extra></extra>"))
            fig.add_vline(x=q.premium_pct, line_color="#52514e", line_dash="dash",
                          annotation_text="recommended", annotation_position="top")
            fig.update_layout(title=dict(text=title, font=dict(size=14)), height=300,
                              margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Premium over the floor (%)",
                              yaxis=dict(tickformat=fmt))
            col.plotly_chart(fig, width="stretch")
        st.caption("The recommended premium isn't simply the margin-maximizing point for this one trip: it is set for "
                   "the whole cell so that rider wait, acceptance and driver earnings stay at least at today's level. "
                   "That is why the dashed line need not sit at the top of the margin curve.")

# ------------------------------------------------------------------------------------------ policy
with tab_policy:
    H = m6["held_summary"]
    rob, today, flo = H["Optimized (robust)"], H["Today"], H["Floor only"]
    st.subheader("The recommended policy, scored on months it never saw")
    st.caption(f"Fitted on {m6['fit_months'][0]} – {m6['fit_months'][-1]} trips, scored on "
               f"{m6['held_months'][0]} – {m6['held_months'][-1]} against the simulated truth.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Margin per request", f"${rob['margin_per_request']:.2f}",
              f"{rob['margin_vs_today_pct']:+.1f}% vs today")
    k2.metric("First-offer acceptance", f"{rob['first_offer_acceptance']:.1%}",
              f"{100 * (rob['first_offer_acceptance'] - today['first_offer_acceptance']):+.1f} pts")
    k3.metric("Driver pay per trip", f"${rob['pay_per_trip']:.2f}", f"{rob['pay_per_trip'] - today['pay_per_trip']:+.2f}",
              delta_color="off")
    k4.metric("Driver earnings per hour", f"${rob['earnings_per_hour']:.2f}",
              f"{100 * (rob['earnings_per_hour'] / today['earnings_per_hour'] - 1):+.1f}% (limit −5%)", delta_color="off")

    rows = []
    for name in ["Today", "Floor only", "Optimized (point)", "Optimized (robust)"]:
        r = H[name]
        rows.append({"Policy": {"Today": "Today's pay", "Floor only": "Pay the minimum only",
                                "Optimized (point)": "Optimized, best guess", "Optimized (robust)": "Optimized, robust"}[name],
                     "Margin / request": f"${r['margin_per_request']:.2f}",
                     "vs today": f"{r['margin_vs_today_pct']:+.1f}%",
                     "First-offer acceptance": f"{r['first_offer_acceptance']:.1%}",
                     "Pay / trip": f"${r['pay_per_trip']:.2f}",
                     "Earnings / hour": f"${r['earnings_per_hour']:.2f}",
                     "Wait guardrail": "—" if name == "Today" else ("✅" if r["C3_wait"] else "❌"),
                     "Acceptance guardrail": "—" if name == "Today" else ("✅" if r["C4_acceptance"] else "❌"),
                     "Earnings guardrail": "—" if name == "Today" else ("✅" if r["C5_earnings"] else "❌")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pt_draws = m6["guardrails_in_draws"]["point"]
    st.markdown(
        f"Paying only the minimum looks like **{flo['margin_vs_today_pct']:+.0f}%** margin but breaks every guardrail. "
        f"The best-guess optimum meets the wait guardrail in only **{pt_draws['C3_wait']:.0%}** of 20 plausible versions "
        "of the acceptance model (posterior draws). The robust policy is built to meet all guardrails in all 20 at once "
        f"and gives up {H['Optimized (point)']['margin_vs_today_pct'] - rob['margin_vs_today_pct']:.1f} points of margin "
        "for that insurance; held out and scored against the truth, both meet every guardrail.")

    st.markdown("#### Where the premium goes")
    pt = pd.DataFrame(m6["premium_table"])
    which = st.radio("Show", ["Robust policy", "Today", "Difference (policy − today)"], horizontal=True)
    val = {"Robust policy": "premium_robust", "Today": "premium_today"}.get(which)
    if val is None:
        pt["v"] = pt["premium_robust"] - pt["premium_today"]
    else:
        pt["v"] = pt[val]
    pt["row"] = pt["cluster_name"] + " · " + pt["band"].map(BAND_LABEL)
    order = [f"{c} · {BAND_LABEL[b]}" for c in ["Core", "Commuter residential", "Long waits", "Airports"]
             for b in BAND_LABEL]
    grid = pt.pivot_table(index="row", columns="q", values="v").reindex([o for o in order if o in set(pt["row"])])
    diverging = val is None
    fig = go.Figure(go.Heatmap(
        z=100 * grid.to_numpy(), x=[f"q{c}" for c in grid.columns], y=grid.index,
        colorscale="RdBu_r" if diverging else "Blues", zmid=0 if diverging else None,
        zmin=None if diverging else 0, zmax=None if diverging else 40,
        text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.0f}%")(100 * grid.to_numpy()),
        texttemplate="%{text}", hovertemplate="%{y}<br>difficulty %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="%")))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Trip difficulty quintile (q5 = hardest)",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("The policy removes premiums on easy trips (drivers accept anyway) and concentrates them on the hardest "
               "quintiles, where a little more pay buys much faster acceptance.")

    st.markdown("#### By borough")
    bb = pd.DataFrame(m6["by_borough"]).sort_values("trips", ascending=False)
    st.dataframe(pd.DataFrame({
        "Borough": bb["borough"], "Trips (held-out sample)": bb["trips"].map("{:,.0f}".format),
        "Margin change / request": bb["margin change"].map(lambda v: f"{v:+.2f}"),
        "Acceptance today": bb["acceptance today"].map("{:.1%}".format),
        "Acceptance policy": bb["acceptance policy"].map("{:.1%}".format),
        "Pay / trip today": bb["pay per trip today"].map("${:.2f}".format),
        "Pay / trip policy": bb["pay per trip policy"].map("${:.2f}".format)}), hide_index=True, width="stretch")
    st.caption("Premium moves from Manhattan, where drivers accept anyway, to the outer boroughs, where it buys service. "
               "A dispatch simulation replaying real Brooklyn requests shows the same trade.")

# ------------------------------------------------------------------------------------------ market
with tab_market:
    st.subheader("How the marketplace behaves (real data)")
    f = m1["floor"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trips analysed", f"{m1['quality']['rows'] / 1e6:,.0f}M", f"{m1['quality']['kept_pct']:.2f}% kept",
              delta_color="off")
    k2.metric("Median pay ÷ legal minimum", f"{f['median_ratio']:.3f}×")
    k3.metric("Trips paid within 1% of minimum", f"{f['at_floor_pct']:.0f}%")
    k4.metric("Premium share of driver pay", f"{f['premium_share_of_pay_pct']:.1f}%")

    left, right = st.columns([1, 1])
    with left:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Patch

        names = ["Core", "Commuter residential", "Long waits", "Airports"]
        colors = {n: c for n, c in zip(names, [SERIES[0], SERIES[1], SERIES[2], plots.GREY])}
        plots.use_style()
        fig_map, ax = plt.subplots(figsize=(6, 6))
        polys, cols = [], []
        for zid, rings in geo.outlines().items():
            c = colors.get(model.cluster_of.get(zid), "#dddddd")
            for ring in rings:
                polys.append(ring)
                cols.append(c)
        ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.2))
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()
        prof = {p["name"]: p for p in m1["clusters"]["profile"]}
        ax.legend(handles=[Patch(color=colors[n], label=f"{n}" + (f" ({prof[n]['trip_share_pct']:.0f}% of trips)" if n in prof else ""))
                           for n in names] + [Patch(color="#dddddd", label="Sparse")], loc="upper left", fontsize=8)
        ax.set_title("Zones that behave alike (k-means, stable under resampling)", fontsize=10)
        st.pyplot(fig_map, width="stretch")
    with right:
        hm = pd.DataFrame(m1["health_month"])
        metric = st.selectbox("Monthly metric", ["wait_min", "trips_per_day", "driver_share", "pay_per_mile", "pay_per_min"],
                              format_func={"wait_min": "Mean rider wait (min)", "trips_per_day": "Trips per day",
                                           "driver_share": "Driver pay ÷ rider fare", "pay_per_mile": "Driver pay per mile ($)",
                                           "pay_per_min": "Driver pay per minute ($)"}.get)
        months = pd.to_datetime(hm["month"] + "-01")
        fig = go.Figure(go.Scatter(x=months, y=hm[metric], mode="lines+markers", line=dict(color=SERIES[0], width=2)))
        fig.add_shape(type="line", x0="2025-01-05", x1="2025-01-05", y0=0, y1=1, yref="paper",
                      line=dict(dash="dot", color="#8a8985"))
        fig.add_annotation(x="2025-01-05", y=1, yref="paper", text="congestion fee starts", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color="#52514e"))
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        L = m2["wmape_levels"]["LightGBM"]
        st.markdown(
            f"- **Waits follow the supply–demand balance, not demand:** the longest mean wait is around 4am on Sundays "
            f"({m1['weekly']['peak_wait_min']:.1f} min).\n"
            f"- **4-week demand forecast:** {L['zone × hour']:.1%} weighted error at zone × hour (Poisson noise alone: "
            f"{m2['poisson_noise_floor_zone_hour']:.1%}); {L['cluster × hour']:.1%} at cluster × hour.\n"
            f"- **Long-wait warning (> 10 min):** ROC AUC {m3['metrics']['LightGBM']['ROC AUC']:.2f}, calibrated; a "
            "cost-based threshold set in advance.\n"
            f"- **Congestion fee:** traffic in the zone {[r for r in m4['yoy_did'] if r['outcome'] == 'mph'][0]['pct_effect']:.1f}% "
            "faster (robust to placebo dates); effects on waits and driver pay not measurable.")

    st.markdown("#### Why the history can't be read directly")
    est = {e["method"]: e for e in m5["estimates"]}
    pp = lambda v: v * np.log(1.1) * 100  # noqa: E731
    rows = [("History, no adjustment", est["Naive (no controls)"]),
            ("Double machine learning", est["Double ML"]),
            ("Randomized pay test", est["Randomized pay test (IV)"])]
    fig = go.Figure()
    for i, (lab, e) in enumerate(rows):
        fig.add_trace(go.Scatter(x=[pp(e["estimate"])], y=[lab], mode="markers", marker=dict(size=11, color=SERIES[0]),
                                 error_x=dict(type="data", symmetric=False, array=[pp(e["hi"] - e["estimate"])],
                                              arrayminus=[pp(e["estimate"] - e["lo"])]), name="estimate", showlegend=i == 0))
        fig.add_trace(go.Scatter(x=[pp(e["truth_of_estimand"])], y=[lab], mode="markers",
                                 marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="#0b0b0b")),
                                 name="true value", showlegend=i == 0))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Acceptance points per 10% more pay",
                      yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig, width="stretch")
    st.caption("Premiums go to the trips drivers like least, so the raw history credits pay with about half its effect. "
               "Double ML and the randomized test each recover the true value of what they estimate (they weight trips "
               "differently, so their numbers differ).")

# ------------------------------------------------------------------------------------------ test
with tab_test:
    st.subheader("Sizing the rollout test")
    st.markdown("Drivers are shared within an area, so trips can't be randomized one by one: the arms would compete for the "
                "same drivers. The test switches whole **regions** between policies in **2-hour blocks**, dropping the first "
                "hour after each switch (drivers are still positioned for the previous policy). The variance comes from "
                "real block-to-block variation in margin per trip.")
    designs = {(d["block_h"], d["washout"]): d for d in m7["design"]}
    labels = {f"{k[0]}-hour blocks" + (", first hour dropped" if k[1] else ""): k for k in designs}
    chosen = (m7["chosen"]["block_h"], m7["chosen"]["washout"])
    default_label = [lab for lab, k in labels.items() if k == chosen][0]
    c1, c2, c3 = st.columns(3)
    lab = c1.selectbox("Design", list(labels), index=list(labels).index(default_label))
    effect = c2.slider("Effect to detect ($ margin per trip)", 0.05, 0.50, round(float(m7["expected_effect"]), 2), 0.01)
    weeks = c3.slider("Weeks", 1, 16, 8)
    d = designs[labels[lab]]
    k1, k2, k3 = st.columns(3)
    k1.metric("Minimum detectable effect", f"${demo.switchback_mde(d['sd_randomization'], d['bias_share'], weeks):.3f}")
    k2.metric("Power for this effect", f"{demo.switchback_power(d['sd_randomization'], d['bias_share'], weeks, effect):.0%}")
    k3.metric("Carryover bias of the design", f"{d['bias_share']:+.0%}", delta_color="off")
    ws = np.arange(1, 17)
    fig = go.Figure()
    for i, (name, key) in enumerate(labels.items()):
        dd = designs[key]
        fig.add_trace(go.Scatter(x=ws, y=[demo.switchback_power(dd["sd_randomization"], dd["bias_share"], w, effect) for w in ws],
                                 name=name, line=dict(color=SERIES[i], width=3 if key == labels[lab] else 1.5)))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#8a8985")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Weeks", yaxis=dict(title="Power", tickformat=".0%"),
                      legend=dict(orientation="h", y=-0.3), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"The policy's expected effect is {money(m7['expected_effect'], 3)} margin per trip. Designs without a "
               "wash-out look more powerful but under-state the effect (carryover from the previous block dilutes it). "
               "Analysis uses randomization inference: in A/A simulations on real data, clustered standard errors flagged "
               "too many null results.")

st.divider()
st.caption("Scenario on public NYC Taxi & Limousine Commission trip records; the TLC data covers several app companies, "
           "treated here as one platform. Driver accept/decline decisions are simulated. Not affiliated with any company.")
