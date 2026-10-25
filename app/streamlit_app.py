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
GOOD, BAD, GREY = "#1baf7a", "#e34948", "#8a8985"
BAND_LABEL = {"night": "Night (0–6)", "am_peak": "AM peak (6–10)", "midday": "Midday (10–16)",
              "pm_peak": "PM peak (16–20)", "evening": "Evening (20–24)"}
GUARDS = {"C3_wait": ("Rider wait", "Expected extra wait from declined offers, in every zone group × hour band, "
                                    "no worse than today."),
          "C4_acceptance": ("First-offer acceptance", "Share of trips accepted by the first driver offered, "
                                                      "no lower than today."),
          "C5_earnings": ("Driver earnings", "Earnings per engaged hour in every zone group at least 95% of today's.")}
POLICY_LABEL = {"Today": "Today's pay", "Floor only": "Pay the minimum only",
                "Optimized (point)": "Optimized, best guess", "Optimized (robust)": "Optimized, robust (recommended)"}
REPO = "https://github.com/janeruxi1/ridehail-driver-pay-optimization"

st.set_page_config(page_title="Driver pay pricing", page_icon="🚕", layout="wide")


def money(x: float, digits: int = 2) -> str:
    """Dollar amount with the $ escaped, so Streamlit's Markdown doesn't read it as maths."""
    return f"\\${x:,.{digits}f}" if x >= 0 else f"−\\${-x:,.{digits}f}"


def usd(x: float, digits: int = 2) -> str:
    """Dollar amount for metrics and chart labels (ASCII minus, so st.metric reads the sign of a delta)."""
    return f"${x:,.{digits}f}" if x >= 0 else f"-${-x:,.{digits}f}"


def chart(fig: go.Figure, height: int = 320, title: str | None = None, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=60 if title else 10, b=10),
                      title=dict(text=title, font=dict(size=14), y=0.98, yanchor="top") if title else None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0), **layout)
    return fig


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
H = m6["held_summary"]
rob, today, flo, pnt = H["Optimized (robust)"], H["Today"], H["Floor only"], H["Optimized (point)"]
designs = {(d["block_h"], d["washout"]): d for d in m7["design"]}
chosen = (m7["chosen"]["block_h"], m7["chosen"]["washout"])
weeks_needed = int(designs[chosen]["weeks_for_expected"])
effects = {(e["comparison"], e["kpi"]): e for e in m6["sim"]["effects"]}

# ------------------------------------------------------------------------------------------ header
st.title("🚕 Driver pay in a ride-hailing marketplace")
st.markdown(
    f"**What should the platform offer a driver for each trip?** Pay too little and drivers decline, so riders wait "
    f"and give up; pay too much and the margin disappears. Built on **{m1['quality']['rows'] / 1e6:,.0f} million real "
    "New York City trips** (TLC records, Jan 2024 – Jun 2025). Trips, fares, pay and waits are real; whether a driver "
    f"accepted an offer is simulated with a known truth, so every method can be checked. [Code and notebooks]({REPO})")

met = sum(bool(rob[g]) for g in GUARDS)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Margin per request", usd(rob["margin_per_request"]), f"{rob['margin_vs_today_pct']:+.1f}% vs today",
          border=True, help="Expected fare minus driver pay, net of riders who give up, per trip request.")
k2.metric("Guardrails met", f"{met} of {len(GUARDS)}", border=True,
          help="Rider wait, first-offer acceptance and driver earnings, each owned by a different team.")
k3.metric("Driver earnings per hour", usd(rob["earnings_per_hour"]),
          f"{100 * (rob['earnings_per_hour'] / today['earnings_per_hour'] - 1):+.1f}% (limit −5%)", delta_color="off",
          border=True)
k4.metric("Rollout test needed", f"{weeks_needed} weeks", border=True,
          help="How long a randomized switchback test must run to detect the expected gain.")
st.caption(f"Recommended policy, scored on {m6['held_months'][0]} – {m6['held_months'][-1]} trips it never saw, "
           "against the simulated truth.")

with st.expander("How to use this demo", icon="🧭"):
    st.markdown(
        "1. **Recommendation**: what to pay, how it compares with today and with simpler options, and where the extra "
        "pay goes.\n"
        "2. **Quote a trip**: pick a trip (or an example) and see the pay the quote service would offer, and why.\n"
        "3. **Marketplace**: the real-data evidence behind it: how pay is set today, where waits are long, and why "
        "the trip history alone would mislead.\n"
        "4. **Rollout test**: how long a switchback test must run before the policy goes live.")

tab_rec, tab_quote, tab_market, tab_test = st.tabs(
    ["📋 Recommendation", "🧮 Quote a trip", "🗺️ Marketplace", "🧪 Rollout test"])

# ------------------------------------------------------------------------------------------ recommendation
with tab_rec:
    st.subheader("Pay a premium where it buys acceptance, and only there")
    st.markdown(
        "Every trip is paid New York's **minimum pay** for its miles and minutes plus a **premium** of 0–30%. The "
        "policy picks one premium per zone group × hour band × trip difficulty, to maximize margin while three "
        "guardrails hold:")
    cols = st.columns(3)
    for col, (g, (name, desc)) in zip(cols, GUARDS.items()):
        ok = bool(rob[g])
        col.markdown(f"{'✅' if ok else '❌'} **{name}**  \n{desc}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        names = ["Floor only", "Optimized (point)", "Optimized (robust)"]
        vals = [H[n]["margin_vs_today_pct"] for n in names]
        all_ok = [all(bool(H[n][g]) for g in GUARDS) for n in names]
        fig = go.Figure(go.Bar(
            y=[POLICY_LABEL[n] for n in names], x=vals, orientation="h",
            marker_color=[GOOD if ok else BAD for ok in all_ok], text=[f"{v:+.1f}%" for v in vals],
            textposition="outside", hovertemplate="%{y}: %{x:+.1f}% margin vs today<extra></extra>"))
        chart(fig, 260, "Margin vs today's pay, held-out months (green: all guardrails met)",
              xaxis=dict(ticksuffix="%", range=[0, 1.2 * max(vals)]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with right:
        broken = [GUARDS[g][0].lower() for g in GUARDS if not pnt[g]]
        held_note = (f"On the held-out months it also breaks the **{' and '.join(broken)}** guardrail."
                     if broken else "On the held-out months it happens to meet all three.")
        st.markdown(
            f"- **Paying only the minimum** looks like **{flo['margin_vs_today_pct']:+.0f}%** margin, but acceptance "
            f"falls to {flo['first_offer_acceptance']:.0%} and every guardrail breaks: drivers would leave.\n"
            f"- **The best-guess optimum** earns {pnt['margin_vs_today_pct'] - rob['margin_vs_today_pct']:.1f} points "
            f"more, but meets the wait guardrail in only {m6['guardrails_in_draws']['point']['C3_wait']:.0%} of 20 "
            f"plausible versions of the acceptance model. {held_note}\n"
            "- **The robust policy** must meet all guardrails in all 20 versions at once: the margin it gives up is "
            "insurance against the model being wrong.")

    with st.expander("Full comparison table"):
        rows = []
        for name in ["Today", "Floor only", "Optimized (point)", "Optimized (robust)"]:
            r = H[name]
            row = {"Policy": POLICY_LABEL[name], "Margin / request": f"${r['margin_per_request']:.2f}",
                   "vs today": f"{r['margin_vs_today_pct']:+.1f}%",
                   "First-offer acceptance": f"{r['first_offer_acceptance']:.1%}",
                   "Pay / trip": f"${r['pay_per_trip']:.2f}", "Earnings / hour": f"${r['earnings_per_hour']:.2f}"}
            for g, (gname, _) in GUARDS.items():
                row[gname] = "—" if name == "Today" else ("✅" if r[g] else "❌")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("#### Where the premium goes")
    st.caption("Each row is a zone group and time of day; each column a fifth of trips by difficulty (q5 = hardest: "
               "short trips, destinations where drivers get stranded, areas short of drivers). Cells show the average "
               "premium over the minimum pay.")
    pt = pd.DataFrame(m6["premium_table"])
    which = st.radio("Show", ["Recommended policy", "Today", "Change (policy − today)"], horizontal=True)
    val = {"Recommended policy": "premium_robust", "Today": "premium_today"}.get(which)
    pt["v"] = pt["premium_robust"] - pt["premium_today"] if val is None else pt[val]
    pt["row"] = pt["cluster_name"] + " · " + pt["band"].map(BAND_LABEL)
    order = [f"{c} · {BAND_LABEL[b]}" for c in ["Core", "Commuter residential", "Long waits", "Airports"]
             for b in BAND_LABEL]
    grid = pt.pivot_table(index="row", columns="q", values="v").reindex([o for o in order if o in set(pt["row"])])
    diverging = val is None
    fig = go.Figure(go.Heatmap(
        z=100 * grid.to_numpy(), x=[f"q{c}" for c in grid.columns], y=grid.index,
        colorscale="RdBu_r" if diverging else "Blues", zmid=0 if diverging else None,
        zmin=None if diverging else 0, zmax=None if diverging else 40,
        text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:+.0f}" if diverging else f"{v:.0f}%")(100 * grid.to_numpy()),
        texttemplate="%{text}", hovertemplate="%{y}<br>difficulty %{x}<br>%{z:.1f}%<extra></extra>",
        colorbar=dict(title="points" if diverging else "%")))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Trip difficulty (q1 = easiest fifth, q5 = hardest)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("The policy removes premiums on easy trips (drivers accept anyway) and concentrates them on the hardest "
               "fifth, where a little more pay buys much faster acceptance. Choose “Change” to see the shift.")

    st.markdown("#### Who gains and who pays")
    bb = pd.DataFrame(m6["by_borough"]).sort_values("trips", ascending=False)
    fig = go.Figure(go.Bar(x=bb["borough"], y=bb["margin change"],
                           marker_color=[SERIES[0] if v >= 0 else SERIES[1] for v in bb["margin change"]],
                           text=[f"{v:+.2f}" for v in bb["margin change"]], textposition="outside",
                           customdata=np.c_[100 * bb["acceptance today"], 100 * bb["acceptance policy"]],
                           hovertemplate="%{x}: %{y:+.2f} $ per request<br>acceptance %{customdata[0]:.1f}% → "
                                         "%{customdata[1]:.1f}%<extra></extra>"))
    lo_y, hi_y = float(bb["margin change"].min()), float(bb["margin change"].max())
    chart(fig, 300, "Change in margin per request, by pickup borough ($)",
          yaxis=dict(range=[1.35 * min(lo_y, 0), 1.35 * max(hi_y, 0)], zeroline=True))
    st.plotly_chart(fig, width="stretch")
    acc = effects[("Robust policy − today's pay", "first_offer_acceptance")]
    mar = effects[("Robust policy − today's pay", "margin_per_request")]
    p90 = effects[("Robust policy − today's pay", "wait_p90_min")]
    st.caption(
        "Margin moves from Manhattan, where drivers accept anyway, to the outer boroughs, where the extra pay buys "
        "faster acceptance. A dispatch simulation replaying the real Brooklyn requests of two days shows the same "
        f"trade: first-offer acceptance {100 * acc['mean']:+.2f} points, margin {mar['mean']:+.2f} $ per request; the "
        f"change in the 90th-percentile wait ({p90['mean']:+.2f} min, 95% interval {p90['lo']:+.2f} to {p90['hi']:+.2f}) "
        "is within simulation noise.")

# ------------------------------------------------------------------------------------------ quote
EXAMPLES = {
    "Late-night short hop in Brooklyn": dict(pu=61, do=17, day=date(2025, 6, 14), hour=2, wait=9.0, miles=0.8,
                                             minutes=6.0, fare=12.0),
    "Morning commute in Manhattan": dict(pu=161, do=236, day=date(2025, 6, 10), hour=8, wait=4.0, miles=2.4,
                                         minutes=14.0, fare=18.5),
    "Afternoon ride from JFK to Midtown": dict(pu=132, do=161, day=date(2025, 6, 12), hour=15, wait=5.0, miles=17.0,
                                               minutes=50.0, fare=72.0),
}


def apply_example() -> None:
    for k, v in EXAMPLES[st.session_state["example"]].items():
        st.session_state[f"q_{k}"] = v


if "q_pu" not in st.session_state:
    st.session_state["example"] = next(iter(EXAMPLES))
    apply_example()

with tab_quote:
    st.subheader("What would we offer the driver for this trip?")
    st.markdown("Start from an example, then change anything. The answer is what the quote service "
                "(`POST /quote`) returns: the legal minimum for the trip, plus the premium the policy sets for "
                "trips like it.")
    st.selectbox("Example trip", list(EXAMPLES), key="example", on_change=apply_example)
    named = zones[zones.index.isin(range(1, 264))].copy()
    named["label"] = named["zone_name"] + " (" + named["borough"] + ")"
    options = named.sort_values("label")
    c1, c2, c3 = st.columns(3)
    with c1:
        pu = st.selectbox("Pickup zone", list(options.index), format_func=lambda z: options.loc[z, "label"], key="q_pu")
        do_choices = list(options.index) + [265]
        do = st.selectbox("Drop-off zone", do_choices, key="q_do",
                          format_func=lambda z: "Outside NYC" if z == 265 else options.loc[z, "label"])
    with c2:
        day = st.date_input("Date", min_value=date(2024, 1, 1), max_value=date(2026, 12, 31), key="q_day")
        hour = st.slider("Request hour", 0, 23, key="q_hour")
        wait = st.slider("Mean rider wait in the pickup zone last hour (min)", 1.0, 15.0, step=0.1, key="q_wait",
                         help="A live supply signal: long recent waits mean drivers are scarce there right now.")
    with c3:
        miles = st.number_input("Trip miles", 0.2, 60.0, step=0.1, key="q_miles")
        minutes = st.number_input("Trip minutes", 1.0, 180.0, step=1.0, key="q_minutes")
        fare = st.number_input("Rider fare ($)", 1.0, 300.0, step=0.5, key="q_fare")

    when = datetime.combine(day, time(hour, 15))
    q = model.quote(int(pu), int(do), when, miles, minutes, fare, wait)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Legal minimum pay", f"${q.floor_pay:,.2f}", border=True,
              help="New York's minimum driver pay for this trip's miles and minutes.")
    k2.metric("Recommended pay", f"${q.recommended_pay:,.2f}", f"+{q.premium_pct:.1f}% premium", delta_color="off",
              border=True)
    if q.accept_probability is None:
        k3.metric("Chance the first driver accepts", "n/a", border=True)
        k4.metric("Expected margin per request", "n/a", border=True)
        st.info("This pickup zone has too few trips to have its own segment, so it is paid the legal minimum by default.",
                icon="ℹ️")
    else:
        k3.metric("Chance the first driver accepts", f"{q.accept_probability:.0%}", border=True,
                  help="Predicted by the acceptance model at the recommended pay.")
        k4.metric("Expected margin per request", f"${q.expected_margin:,.2f}", border=True,
                  help="Fare minus pay if the trip completes, less the cost of riders who give up after declines.")
        cl, band = q.segment.split("|")
        hardness = {1: "one of the easiest", 2: "an easier", 3: "a middling", 4: "a harder", 5: "one of the hardest"}
        st.markdown(f"**Why this premium:** a **{cl}** zone at **{BAND_LABEL[band]}**, and {hardness[q.difficulty_quintile]} "
                    f"trips to place (difficulty {q.difficulty_quintile} of 5). Short trips, destinations where drivers "
                    "get stranded and areas short of drivers are harder, so they get more of the premium budget.")
        curve = demo.offer_curve(model, int(pu), int(do), when, miles, minutes, fare, wait)
        ca, cb = st.columns(2)
        for col, ycol, title, fmt, color in [
                (ca, "accept_probability", "Chance the first driver accepts, by premium", ".0%", SERIES[0]),
                (cb, "expected_margin", "Expected margin per request, by premium ($)", ".2f", SERIES[1])]:
            fig = go.Figure(go.Scatter(x=curve["premium_pct"], y=curve[ycol], mode="lines+markers",
                                       line=dict(color=color, width=2), marker=dict(size=5),
                                       hovertemplate="premium %{x:.1f}%<br>%{y:" + fmt + "}<extra></extra>"))
            fig.add_vline(x=q.premium_pct, line_color="#52514e", line_dash="dash",
                          annotation_text="recommended", annotation_position="top")
            chart(fig, 300, title, xaxis_title="Premium over the legal minimum (%)", yaxis=dict(tickformat=fmt))
            col.plotly_chart(fig, width="stretch")
        st.caption("More pay always raises acceptance, but for a single trip it can lower margin. The premium is set "
                   "for all similar trips together so that waits, acceptance and driver earnings stay at least at "
                   "today's level, which is why the dashed line need not sit at the top of the margin curve.")

# ------------------------------------------------------------------------------------------ market
with tab_market:
    st.subheader("How the marketplace behaves today (real data)")
    f = m1["floor"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Trips analysed", f"{m1['quality']['rows'] / 1e6:,.0f}M", border=True,
              help=f"{m1['quality']['kept_pct']:.2f}% pass every cleaning rule.")
    k2.metric("Median pay ÷ legal minimum", f"{f['median_ratio']:.3f}×", border=True,
              help="Most trips are paid almost exactly the legal minimum.")
    k3.metric("Trips paid within 1% of the minimum", f"{f['at_floor_pct']:.0f}%", border=True)
    k4.metric("Premium share of driver pay", f"{f['premium_share_of_pay_pct']:.1f}%", border=True,
              help="Pay above the legal minimum: the budget the policy reallocates.")

    left, right = st.columns([1, 1], gap="large")
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
        ax.set_title("Zone groups that behave alike", fontsize=10)
        st.pyplot(fig_map, width="stretch")
        st.caption("Zones grouped by k-means on volume, daily shape, waits, trip economics and drop-off imbalance; "
                   "the groups stay the same when zones are resampled. These groups are the policy's segments.")
    with right:
        hm = pd.DataFrame(m1["health_month"])
        metric = st.selectbox("Monthly metric", ["wait_min", "trips_per_day", "driver_share", "pay_per_mile", "pay_per_min"],
                              format_func={"wait_min": "Mean rider wait (min)", "trips_per_day": "Trips per day",
                                           "driver_share": "Driver pay ÷ rider fare", "pay_per_mile": "Driver pay per mile ($)",
                                           "pay_per_min": "Driver pay per minute ($)"}.get)
        months = pd.to_datetime(hm["month"] + "-01")
        fig = go.Figure(go.Scatter(x=months, y=hm[metric], mode="lines+markers", line=dict(color=SERIES[0], width=2)))
        fig.add_shape(type="line", x0="2025-01-05", x1="2025-01-05", y0=0, y1=1, yref="paper",
                      line=dict(dash="dot", color=GREY))
        fig.add_annotation(x="2025-01-05", y=1, yref="paper", text="congestion fee starts", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color="#52514e"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        L = m2["wmape_levels"]["LightGBM"]
        st.markdown(
            "**What else the data shows**\n"
            f"- **Waits follow the balance of drivers and riders, not demand:** the longest mean wait is around 4am on "
            f"Sundays ({m1['weekly']['peak_wait_min']:.1f} min).\n"
            f"- **4-week demand forecast:** {L['zone × hour']:.1%} weighted error by zone and hour (random noise alone "
            f"would give {m2['poisson_noise_floor_zone_hour']:.1%}); {L['cluster × hour']:.1%} by zone group and hour.\n"
            f"- **Long-wait warning (over 10 min):** ROC AUC {m3['metrics']['LightGBM']['ROC AUC']:.3f}, calibrated, with "
            "a cost-based threshold set in advance.\n"
            f"- **2025 congestion fee:** traffic in the zone "
            f"{[r for r in m4['yoy_did'] if r['outcome'] == 'mph'][0]['pct_effect']:.1f}% faster (robust to placebo "
            "dates); effects on waits and driver pay are not measurable.")

    st.markdown("#### Why the trip history alone would mislead")
    st.markdown("Today's premiums go to the trips drivers like least, so in the history extra pay arrives together with "
                "extra difficulty. Read the chart as: dot = what each method estimates, black tick = the true value of "
                "what that method estimates (known here because acceptance is simulated).")
    est = {e["method"]: e for e in m5["estimates"]}
    pp = lambda v: v * np.log(1.1) * 100  # noqa: E731
    rows = [("History, no adjustment", est["Naive (no controls)"]),
            ("Double machine learning", est["Double ML"]),
            ("Randomized pay test", est["Randomized pay test (IV)"])]
    fig = go.Figure()
    for i, (lab, e) in enumerate(rows):
        fig.add_trace(go.Scatter(x=[pp(e["estimate"])], y=[lab], mode="markers", marker=dict(size=11, color=SERIES[0]),
                                 error_x=dict(type="data", symmetric=False, array=[pp(e["hi"] - e["estimate"])],
                                              arrayminus=[pp(e["estimate"] - e["lo"])]), name="estimate (95% CI)",
                                 showlegend=i == 0))
        fig.add_trace(go.Scatter(x=[pp(e["truth_of_estimand"])], y=[lab], mode="markers",
                                 marker=dict(symbol="line-ns", size=22, line=dict(width=3, color="#0b0b0b")),
                                 name="true value", showlegend=i == 0))
    chart(fig, 280, "Extra acceptance from 10% more pay (percentage points)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("The raw history credits pay with about half of its true effect. Double machine learning and the "
               "randomized pay test each land on the true value of what they estimate; they weight trips differently, "
               "so their numbers differ. Only the randomized test stays right when a confounder is hidden.")

# ------------------------------------------------------------------------------------------ test
with tab_test:
    st.subheader("How long must the rollout test run?")
    st.markdown("Drivers are shared within an area, so trips can't be randomized one by one: both arms would compete "
                "for the same drivers. Instead the test switches whole **regions** between today's pay and the new "
                "policy in **time blocks** (a switchback test), and drops the first hour after each switch while "
                "drivers reposition. The noise comes from real block-to-block variation in margin per trip.")
    labels = {f"{k[0]}-hour blocks" + (", first hour dropped" if k[1] else ""): k for k in designs}
    default_label = [lab for lab, k in labels.items() if k == chosen][0]
    c1, c2, c3 = st.columns(3)
    lab = c1.selectbox("Design", list(labels), index=list(labels).index(default_label))
    effect = c2.slider("Gain to detect ($ margin per trip)", 0.05, 0.50, round(float(m7["expected_effect"]), 2), 0.01,
                       help=f"The policy's expected gain is {usd(m7['expected_effect'], 3)} per trip.")
    weeks = c3.slider("Weeks", 1, 16, int(min(16, max(1, weeks_needed))))
    d = designs[labels[lab]]
    mde = demo.switchback_mde(d["sd_randomization"], d["bias_share"], weeks)
    power = demo.switchback_power(d["sd_randomization"], d["bias_share"], weeks, effect)
    k1, k2, k3 = st.columns(3)
    k1.metric("Smallest gain this test can detect", f"${mde:.3f}", border=True,
              help="Minimum detectable effect at 80% power and 5% significance.")
    k2.metric("Chance of detecting the chosen gain", f"{power:.0%}", border=True,
              help="Statistical power: the probability the test shows the gain if it is real.")
    k3.metric("Carryover bias of the design", f"{d['bias_share']:+.0%}", border=True,
              help="How much switching between policies distorts the measured gain.")
    box = st.success if power >= 0.8 else st.warning
    box(f"{lab}, {weeks} weeks: {power:.0%} chance of detecting a gain of {money(effect)} per trip "
        + ("(the usual bar is 80%)." if power >= 0.8 else "— below the usual 80%: run longer or accept a larger miss."),
        icon="✅" if power >= 0.8 else "⚠️")
    ws = np.arange(1, 17)
    fig = go.Figure()
    for i, (name, key) in enumerate(labels.items()):
        dd = designs[key]
        fig.add_trace(go.Scatter(x=ws, y=[demo.switchback_power(dd["sd_randomization"], dd["bias_share"], w, effect) for w in ws],
                                 name=name, line=dict(color=SERIES[i], width=3 if key == labels[lab] else 1.5)))
    fig.add_hline(y=0.8, line_dash="dash", line_color=GREY, annotation_text="80% power", annotation_position="bottom right")
    chart(fig, 340, "Chance of detecting the gain, by test length", xaxis_title="Weeks",
          yaxis=dict(title="Power", tickformat=".0%"), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Designs without a wash-out hour look more powerful, but they under-state the gain: drivers still "
               "positioned for the previous policy dilute it. The analysis uses randomization inference, because in A/A "
               "simulations on real data clustered standard errors flagged too many null results.")

st.divider()
st.caption("Scenario on public NYC Taxi & Limousine Commission trip records; the TLC data covers several app companies, "
           "treated here as one platform. Driver accept/decline decisions are simulated. Not affiliated with any company.")
