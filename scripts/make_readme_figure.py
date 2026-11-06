"""Draw the README's summary figure from the recorded results (no data needed).

    python scripts/make_readme_figure.py   ->  reports/figures/00_readme_summary.png
"""
import json
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


m5, m6, m7 = rec("05_pay_to_accept"), rec("06_policy_simulation"), rec("07_switchback")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})

ax = axes[0]
labels = {"Naive (no controls)": "History, no controls", "Double ML": "Double ML",
          "Randomized pay test (IV)": "Randomized pay test"}
est = [e for e in m5["estimates"] if e["method"] in labels][::-1]
for i, e in enumerate(est):
    k = e["per_10pct_pay_pp"] / e["estimate"]
    c = BLUE if e["covers_truth"] else ORANGE
    ax.plot([k * e["lo"], k * e["hi"]], [i, i], color=c, lw=2.5, solid_capstyle="round")
    ax.plot(k * e["estimate"], i, "o", ms=9, color=c, mec="white", mew=2, label="estimate" if i == 0 else None)
    ax.plot(k * e["truth_of_estimand"], i, "|", ms=18, mew=2.5, color=INK, label="planted truth" if i == 0 else None)
    ax.text(k * max(e["hi"], e["truth_of_estimand"]) + 0.25, i, f"{e['per_10pct_pay_pp']:+.1f} pp", va="center",
            fontsize=10.5)
ax.set_yticks(range(len(est)), [labels[e["method"]] for e in est])
ax.set_xlim(0, 8.5)
ax.set_xlabel("first-offer acceptance gain from 10% more pay (points)")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
ax.set_title("1 · The history understates pay's effect;\ncausal methods recover the planted truth")

ax = axes[1]
h = m6["held_summary"]
pols = [("Floor only", "Pay the\nminimum"), ("Optimized (point)", "Best-guess\noptimum"),
        ("Optimized (robust)", "Robust\n(recommended)")]
ticks = []
for i, (k, lab) in enumerate(pols):
    v = h[k]
    broken = sum(not v[g] for g in ("C3_wait", "C4_acceptance", "C5_earnings"))
    ax.bar(i, v["margin_vs_today_pct"], color=BLUE if k == "Optimized (robust)" else GREY, width=0.55)
    ax.text(i, v["margin_vs_today_pct"] + 0.8, f"+{v['margin_vs_today_pct']:.1f}%", ha="center", fontsize=11)
    miss = [n for g, n in (("C3_wait", "wait"), ("C4_acceptance", "acceptance"), ("C5_earnings", "earnings")) if not v[g]]
    ticks.append(lab + ("\nmeets all 3" if not miss else "\nbreaks all 3" if len(miss) == 3 else "\nmisses " + ", ".join(miss)))
ax.set_xticks(range(len(pols)), ticks)
ax.set_ylim(0, 34)
ax.set_ylabel("margin per request vs today (%)")
ax.grid(axis="x", visible=False)
rob_ok = all(h["Optimized (robust)"][g] for g in ("C3_wait", "C4_acceptance", "C5_earnings"))
slips = m6.get("held_wait_slips", {}).get("Optimized (robust)")
ax.set_title(f"2 · The robust policy adds {h['Optimized (robust)']['margin_vs_today_pct']:.1f}% margin;\n"
             + ("every guardrail met held out" if rob_ok else
                f"held out, wait slips in {len(slips)} of 20 segments" if slips else "held out, it misses on wait"))

ax = axes[2]
ds = m7["design"]
names = [f"{d['block_h']}-hour blocks" + (", washout" if d["washout"] else "") for d in ds]
chosen = [d["block_h"] == m7["chosen"]["block_h"] and d["washout"] == m7["chosen"]["washout"] for d in ds]
for i, (d, n, c) in enumerate(zip(ds, names, chosen)):
    ax.barh(i, d["weeks_for_expected"], color=BLUE if c else GREY, height=0.55)
    ax.text(d["weeks_for_expected"] + 0.15, i, f"{d['weeks_for_expected']} weeks · bias {d['bias_share']:+.0%}".replace("-", "−"),
            va="center", fontsize=10)
ax.set_yticks(range(len(ds)), names)
ax.invert_yaxis()
ax.set_xlim(0, 10)
ax.set_xlabel("weeks to detect the expected margin effect")
ax.grid(axis="y", visible=False)
ax.set_title("3 · Roll out with a 4-week switchback:\n2-hour blocks, first hour washed out")

save(fig, ROOT / "reports" / "figures" / "00_readme_summary.png")


# ---- hero: why the robust policy (guardrails across the posterior draws) ----------------------
gd = m6["guardrails_in_draws"]
names = [("C3_wait", "Rider wait no worse\nthan today, every segment"),
         ("C4_acceptance", "First-offer acceptance\nno lower than today"),
         ("C5_earnings", "Driver earnings per hour\n≥ 95% of today")]
fig, ax = plt.subplots(figsize=(11, 4.4))
yy = list(range(len(names)))[::-1]
for y, (k, lab) in zip(yy, names):
    for dy, pol_key, c, txt in ((0.17, "robust", BLUE, "robust"), (-0.17, "point", ORANGE, "best guess")):
        v = 100 * gd[pol_key][k]
        ax.barh(y + dy, v, height=0.3, color=c)
        ax.text(v + 1, y + dy, f"{v:.0f}%", va="center", fontsize=10.5)
ax.set_yticks(yy, [lab for _, lab in names])
ax.set_xlim(0, 112)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("share of plausible driver responses (posterior draws) in which the guardrail holds")
ax.grid(axis="y", visible=False)
ax.legend(handles=[matplotlib.patches.Patch(color=BLUE, label="robust policy: "
                                            f"+{h['Optimized (robust)']['margin_vs_today_pct']:.1f}% margin"),
                   matplotlib.patches.Patch(color=ORANGE, label="best-guess optimum: "
                                            f"+{h['Optimized (point)']['margin_vs_today_pct']:.1f}% margin")],
          frameon=False, fontsize=10.5, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2)
ax.set_title(f"The best-guess optimum keeps the wait guardrail in only {100 * gd['point']['C3_wait']:.0f}% of plausible "
             "driver responses;\nthe robust policy keeps every guardrail in all of them, for "
             f"{h['Optimized (point)']['margin_vs_today_pct'] - h['Optimized (robust)']['margin_vs_today_pct']:.1f} "
             "points of held-out margin")
save(fig, ROOT / "reports" / "figures" / "00_readme_hero.png")

# ---- who gains and who pays, by borough ---------------------------------------------------------
bb = sorted(m6["by_borough"], key=lambda r: r["trips"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, (col_a, col_b, lab, fmt) in zip(axes, (("pay per trip policy", "pay per trip today",
                                                 "change in driver pay per trip ($)", "{:+.2f}"),
                                                ("acceptance policy", "acceptance today",
                                                 "change in first-offer acceptance (points)", "{:+.1f}"))):
    vals = [(r[col_a] - r[col_b]) * (100 if "acceptance" in col_a else 1) for r in bb]
    ax.barh([r["borough"] for r in bb], vals, color=[BLUE if v >= 0 else GREY for v in vals], height=0.55)
    span = max(abs(v) for v in vals)
    for i, v in enumerate(vals):
        ax.text(v + (0.04 if v >= 0 else -0.04) * span * 2.5, i, fmt.format(v).replace("-", "−"), va="center",
                ha="left" if v >= 0 else "right", fontsize=10)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlim(-1.5 * span, 1.5 * span)
    ax.set_xlabel(lab)
    ax.grid(axis="y", visible=False)
axes[0].set_title("Drivers are paid more in the outer boroughs\nand less in Manhattan, where they accept anyway")
axes[1].set_title("…which buys faster acceptance there")
save(fig, ROOT / "reports" / "figures" / "00_readme_boroughs.png")
