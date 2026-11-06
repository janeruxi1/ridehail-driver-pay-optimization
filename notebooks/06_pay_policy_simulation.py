# %% [markdown]
# # 06 · Pay policy, matching and a dispatch simulation
#
# **Question.** For each trip, what premium above the floor should we offer, to maximize expected
# margin while keeping waits, acceptance and driver earnings at least where they are today? And does
# the policy still hold up when we replay a real day of requests with drivers moving around the city?
#
# **Part A: the policy.** Using the acceptance curves from phase 5 (the hierarchical model, with its
# uncertainty), choose a premium for each cell = zone cluster × hour band × difficulty quintile, by
# mixed-integer linear programming under constraints C1–C5. The policy is **fitted on January–March
# 2025 trips and evaluated on April–June 2025**, scored against the *planted truth*, which the
# optimizer never saw.
#
# **Part B: the simulation.** Replay every real Brooklyn request on two days of June 2025 (a Tuesday
# and a Saturday) through a dispatch simulator: simulated drivers, real requests. Compare today's pay
# with the policy, and greedy matching with batched optimal matching.


# %% [markdown]
# | | |
# |---|---|
# | **Answer in short** | A robust pay policy, built to meet the guardrails across many plausible versions of the pay-response model, raises margin per request by 3.7% on held-out months with acceptance, waits and driver earnings all within their guardrails. Paying only the floor looks cheapest but breaks the guardrails. |
# | **Next** | [Notebook 07](07_switchback_design.ipynb) designs the experiment to test it live. |

# %%
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from ridepay import accept, config, data, dispatch, features, plots, policy, simulate
from ridepay.records import show_verdict, write_metrics

SOURCE = data.source()
FAST = SOURCE != "real"          # CI smoke run on the synthetic city: fewer draws, seeds and calibration steps
con = data.connect()
ALL = data.months()
FIT = [m for m in ALL if "2025-01" <= m <= "2025-03"] or ALL[-6:-3]
HELD = [m for m in ALL if "2025-04" <= m <= "2025-06"] or ALL[-3:]
post = np.load(data.interim_dir() / "derived" / "accept_posterior.npz")
segments = [tuple(s.split("|")) for s in post["segments"]]
seg_index = {s: i for i, s in enumerate(segments)}
truth_tab = data.load_table("accept_truth").set_index(["cluster_name", "band"])
print(f"fit on {FIT}, evaluate on {HELD}; {len(segments)} segments; {post['beta'].shape[0]} posterior draws")


def prepare(months):
    df, info = features.trip_frame(months, con)
    f = simulate.difficulty_features(df, info)
    df = df.join(f)
    df = df[[(c, b) in seg_index for c, b in zip(df["cluster_name"], df["band"])]].copy()
    df["seg"] = [seg_index[(c, b)] for c, b in zip(df["cluster_name"], df["band"])]
    return df, info


fit_df, info = prepare(FIT)
held_df, _ = prepare(HELD)
gamma_mean = post["gamma"].mean(axis=0)
fit_df["d_hat"] = fit_df[accept.FEATURES].to_numpy() @ gamma_mean
edges = np.quantile(fit_df["d_hat"], [0.2, 0.4, 0.6, 0.8])           # quintile edges from the fit window only
for d in (fit_df, held_df):
    d["d_hat"] = d[accept.FEATURES].to_numpy() @ gamma_mean
    d["q"] = np.searchsorted(edges, d["d_hat"]) + 1
    d["cell"] = d["cluster_name"] + "|" + d["band"] + "|q" + d["q"].astype(str)
print(f"{len(fit_df):,} fit trips, {len(held_df):,} held-out trips, {fit_df['cell'].nunique()} cells")

# %% [markdown]
# ## Parameter sets
#
# * **posterior mean**: the single best guess;
# * **20 posterior draws**: plausible versions of the world, for the robust policy;
# * **the truth**: the planted model (integrated over the unobserved term), used *only* to score
#   policies on the held-out months.

# %%
D = 6 if FAST else 20
rng = np.random.default_rng(0)
pick = rng.choice(post["beta"].shape[0], D, replace=False)
mean_draw = {"alpha": post["alpha"].mean(0), "beta": post["beta"].mean(0), "gamma": gamma_mean}
draws = [mean_draw] + [{"alpha": post["alpha"][k], "beta": post["beta"][k], "gamma": post["gamma"][k]} for k in pick]


def accept_model(df, log_ratio, draw):
    return accept.predict_accept(draw["alpha"], draw["beta"], draw["gamma"], df["seg"].to_numpy(), log_ratio,
                                 df[accept.FEATURES].to_numpy())


def accept_truth(df, log_ratio, _draw=None):
    idx = list(zip(df["cluster_name"], df["band"]))
    a = truth_tab.loc[idx, "a"].to_numpy()
    b = truth_tab.loc[idx, "beta"].to_numpy()
    return simulate.acceptance_prob(a, b, simulate.observed_difficulty(df), log_ratio)


t0 = time.time()
fit_table = policy.cell_table(fit_df, accept_model, draws)
print(f"cell table: {len(fit_table.cells)} cells × {len(policy.PREMIUMS)} premium options × {len(draws)} draws "
      f"({time.time() - t0:.0f}s)")

# %% [markdown]
# ## Optimize
#
# Four policies to compare:
# 1. **Today**: the real pay offered (floor + the platform's actual premiums).
# 2. **Floor only**: no premiums at all, the naive finance answer.
# 3. **Optimized (point)**: MILP with the posterior mean, constraints imposed once.
# 4. **Optimized (robust)**: same objective averaged over draws; the guardrails must hold in all 20
#    posterior draws (scenario approach).

# %%
t0 = time.time()
choice_point, res_point = policy.optimize(fit_table, constraint_draws=[0], objective_draws=[0])
choice_robust, res_robust = policy.optimize(fit_table, constraint_draws=range(1, D + 1), objective_draws=range(1, D + 1))
print(f"solved in {time.time() - t0:.0f}s; point: {res_point.message}; robust: {res_robust.message}")
floor_only = np.zeros(len(fit_table.cells), int)
fit_summary = pd.DataFrame({name: policy.summarize(fit_table, ch, draw=0) for name, ch in
                            [("Today", "sq"), ("Floor only", floor_only), ("Optimized (point)", choice_point),
                             ("Optimized (robust)", choice_robust)]}).T
print("On the fit months, under the posterior mean (what the optimizer believes):")
fit_summary.round(4)

# %% [markdown]
# ## Score on held-out months against the truth

# %%
held_table = policy.cell_table(held_df, accept_truth, [None])
cells_fit = fit_table.cells["cell"].tolist()
# map each held-out cell to the fitted choice (cells seen in the fit window; unseen cells get no premium)


def choice_for(table, fitted_choice):
    m = dict(zip(cells_fit, fitted_choice))
    return np.array([m.get(c, 0) for c in table.cells["cell"]])


policies = {"Today": "sq", "Floor only": np.zeros(len(held_table.cells), int),
            "Optimized (point)": choice_for(held_table, choice_point),
            "Optimized (robust)": choice_for(held_table, choice_robust)}
held_summary = pd.DataFrame({k: policy.summarize(held_table, v) for k, v in policies.items()}).T
checks = pd.DataFrame({k: policy.guardrails(held_table, v) for k, v in policies.items() if not isinstance(v, str)}).T
held_summary = held_summary.join(checks)
held_summary["margin_vs_today_pct"] = 100 * (held_summary["margin_per_request"] / held_summary.loc["Today", "margin_per_request"] - 1)
print("Held-out months (April–June 2025), scored with the planted truth:")
held_summary.round(4)

# %% [markdown]
# **Reading.** Floor-only saves the most pay per trip but breaks the guardrails: acceptance falls,
# riders wait and some give up. The optimized policies raise margin and acceptance together, because they
# **move** premium: away from cells where drivers barely respond to it (or accept anyway) toward cells where
# a little more pay buys much faster acceptance. The wait guardrail (C3) is checked in each of the 20
# segments, with no tolerance, so the next cell shows where it slips and by how much.

# %%
def wait_by_segment(choice):
    """Extra wait per completed trip by segment on the held-out months: policy vs today."""
    C = len(held_table.cells)
    S = {k: v[0, np.arange(C), choice] for k, v in held_table.sums.items()}
    Q = {k: v[0] for k, v in held_table.status_quo.items()}
    rows = []
    for (cl, band), i in held_table.cells.groupby(["cluster_name", "band"]).indices.items():
        today_s, pol_s = Q["extra_wait_sec"][i].sum(), S["extra_wait_sec"][i].sum()
        rows.append({"cluster": cl, "band": band, "completed": float(Q["complete"][i].sum()),
                     "wait_today_sec": today_s / Q["complete"][i].sum(), "wait_policy_sec": pol_s / S["complete"][i].sum(),
                     "total_change_pct": 100 * (pol_s / today_s - 1)})
    return pd.DataFrame(rows).sort_values("total_change_pct", ascending=False).reset_index(drop=True)


wait_seg = {name: wait_by_segment(policies[name]) for name in ["Optimized (point)", "Optimized (robust)"]}
wait_slips = {name: t.loc[t["total_change_pct"] > 0] for name, t in wait_seg.items()}
for name, t in wait_slips.items():
    print(f"{name}: wait above today in {len(t)} of {len(wait_seg[name])} segments")
wait_slips["Optimized (robust)"].round(2)

# %% [markdown]
# **Reading.** Robust here means robust to the acceptance model's *uncertainty*: the guardrails hold in
# every one of the 20 posterior draws. It does not protect against the model being *wrong* in the same
# direction in every draw. Any segment that slips against the truth is worth checking against the
# hierarchical model's accuracy there in notebook 05 (the airports have the fewest trips).

# %% [markdown]
# **Where does the gain come from?** The same held-out comparison by pickup borough, trip by trip.
# The policy is citywide; it need not help everywhere.

# %%
robust_premium = dict(zip(cells_fit, policy.PREMIUMS[choice_robust]))
held_df["borough"] = held_df["PULocationID"].map(info["borough"])


def trip_outcomes(df, offer):
    p = accept_truth(df, np.log(offer / df["floor"].to_numpy()))
    return pd.DataFrame(policy.outcomes(p, df["fare"].to_numpy(), offer, df["minutes"].to_numpy()), index=df.index)


o_today = trip_outcomes(held_df, held_df["offer_sq"].to_numpy())
o_pol = trip_outcomes(held_df, (held_df["floor"] * (1 + held_df["cell"].map(robust_premium).fillna(0.0))).to_numpy())
by_borough = pd.DataFrame({
    "trips": held_df.groupby("borough").size(),
    "margin today": o_today.groupby(held_df["borough"])["margin"].mean(),
    "margin policy": o_pol.groupby(held_df["borough"])["margin"].mean(),
    "pay per trip today": o_today["driver_pay"].groupby(held_df["borough"]).sum() / o_today["complete"].groupby(held_df["borough"]).sum(),
    "pay per trip policy": o_pol["driver_pay"].groupby(held_df["borough"]).sum() / o_pol["complete"].groupby(held_df["borough"]).sum(),
    "acceptance today": o_today.groupby(held_df["borough"])["accept_first"].mean(),
    "acceptance policy": o_pol.groupby(held_df["borough"])["accept_first"].mean(),
})
by_borough["margin change"] = by_borough["margin policy"] - by_borough["margin today"]
by_borough.sort_values("trips", ascending=False).round(3)

# %%
cells_view = fit_table.cells.assign(premium_point=policy.PREMIUMS[choice_point],
                                    premium_robust=policy.PREMIUMS[choice_robust])
sq_ratio = fit_df.groupby("cell").apply(lambda g: (g["offer_sq"].sum() / g["floor"].sum()) - 1, include_groups=False)
cells_view["premium_today"] = cells_view["cell"].map(sq_ratio)
cells_view["q"] = cells_view["cell"].str[-1].astype(int)
heat = cells_view.pivot_table(index=["cluster_name", "band"], columns="q", values="premium_robust")
heat_today = cells_view.pivot_table(index=["cluster_name", "band"], columns="q", values="premium_today")
fig, axes = plots.subplots(1, 2, figsize=(11, 6), sharey=True)
vmax = max(heat.max().max(), heat_today.max().max(), 0.3)
for ax, h, title in [(axes[0], heat_today, "Today: average premium"), (axes[1], heat, "Robust policy: premium")]:
    im = ax.imshow(h.to_numpy(), cmap=plots.SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(h.shape[1]), [f"q{c}" for c in h.columns])
    ax.set_yticks(range(h.shape[0]), [f"{c[:14]} · {b}" for c, b in h.index], fontsize=7)
    ax.set_xlabel("difficulty quintile (q5 = hardest)")
    ax.set_title(title)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(h.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6,
                    color="white" if v > vmax * 0.6 else plots.INK)
fig.colorbar(im, ax=axes, shrink=0.6, label="premium over the floor")
data.savefig(fig, "06_premium_map")


# %% [markdown]
# **How to read the figure.** Rows are zone clusters and times of day; columns are trip difficulty, from easiest (q1) to hardest (q5). The colour is the premium above the floor: today's average on the left, the robust policy's on the right. Outside the airports, the policy pays nothing above the floor on the easiest trips and concentrates the premium on the hardest ones (q4–q5), where it changes acceptance most.

# %% [markdown]
# ## How stable is the recommendation?
#
# Re-solve the point problem under 10 individual posterior draws. If the recommended premium for a
# cell jumps around between plausible worlds, the data doesn't support a confident cell-level choice.

# %%
alt = []
for d in range(1, min(11, D + 1)):
    ch, _ = policy.optimize(fit_table, constraint_draws=[d], objective_draws=[d], time_limit=60)
    alt.append(policy.PREMIUMS[ch])
alt = np.array(alt)
spread = np.abs(alt - policy.PREMIUMS[choice_robust][None, :])
stability = {"same_as_robust_share": float((spread < 1e-9).mean()), "within_5pts_share": float((spread <= 0.05 + 1e-9).mean()),
             "mean_abs_diff_pts": float(100 * spread.mean())}
robust_in_draws = pd.DataFrame([policy.guardrails(fit_table, choice_robust, d) for d in range(1, D + 1)]).mean()
point_in_draws = pd.DataFrame([policy.guardrails(fit_table, choice_point, d) for d in range(1, D + 1)]).mean()
print("stability:", {k: round(v, 3) for k, v in stability.items()})
pd.DataFrame({"point policy": point_in_draws, "robust policy": robust_in_draws}).rename_axis("share of posterior draws where the guardrail holds")

# %% [markdown]
# ## The price of safety: how sure do we need to be?
#
# The robust policy demands every guardrail in all 20 posterior draws; the point policy demands them only
# under the best guess. Between the two, a **chance-constrained** policy demands them in at least a given
# share of the draws, and the solver picks which draws may fail. Each policy is fitted on the fit months
# and scored on the held-out months against the truth, so the curve shows what each level of caution
# costs and whether it buys guardrails that actually hold.

# %%
from ridepay import ope  # noqa: E402

LEVELS = [0.0, 0.5, 0.8, 0.95, 1.0]
today_margin = held_summary.loc["Today", "margin_per_request"]
frontier = []
for share in LEVELS:
    ch, res_c, _ = policy.optimize_chance(fit_table, range(1, D + 1), share, time_limit=60 if FAST else 300)
    if ch is None:
        print(f"no policy keeps every guardrail in ≥{share:.0%} of draws: {res_c.message}")
        continue
    hc = choice_for(held_table, ch)
    s_, g_ = policy.summarize(held_table, hc), policy.guardrails(held_table, hc)
    in_draws = pd.DataFrame([policy.guardrails(fit_table, ch, d) for d in range(1, D + 1)]).all(axis=1).mean()
    frontier.append({"min_share": share, "draws_with_all_guardrails": float(in_draws),
                     "margin_vs_today_pct": float(100 * (s_["margin_per_request"] / today_margin - 1)),
                     "held_out_all_guardrails": bool(all(g_.values())), **g_})
frontier = pd.DataFrame(frontier)
frontier.round(3)

# %%
fig, ax = plots.subplots(figsize=(8, 4))
shown = frontier[frontier["min_share"] > 0]   # no guardrails at all (≥0%) is far off the scale; noted in the title
ax.plot(100 * shown["draws_with_all_guardrails"], shown["margin_vs_today_pct"], color=plots.SERIES[0], lw=2)
for k, r in enumerate(shown.itertuples()):
    ax.plot(100 * r.draws_with_all_guardrails, r.margin_vs_today_pct, "o", ms=9,
            color=plots.SERIES[0] if r.held_out_all_guardrails else plots.SERIES[1])
    ax.annotate(f"≥{r.min_share:.0%}: {r.margin_vs_today_pct:+.1f}%", (100 * r.draws_with_all_guardrails, r.margin_vs_today_pct),
                (-8, 10 if k % 2 == 0 else -16), textcoords="offset points", fontsize=8, ha="right")
free = frontier.loc[frontier["min_share"] == 0, "margin_vs_today_pct"]
ax.set(xlabel="share of posterior draws in which every guardrail holds (fit months)",
       ylabel="margin per request vs today, held out (%)",
       title="The price of safety: margin given up for guardrails that hold in more plausible worlds"
       + (f"\n(with no guardrails at all: {free.iloc[0]:+.1f}%, breaking them)" if len(free) else ""))
ax.margins(x=0.15, y=0.3)
data.savefig(fig, "06b_price_of_safety")

# %% [markdown]
# **How to read the figure.** Each point is a policy that demands the guardrails in at least the labelled
# share of posterior draws. Left to right, more caution; down, less margin. Blue points keep every guardrail
# on the held-out months against the truth, orange points break at least one.
#
# Which points clear the held-out check depends on where the model happens to be wrong, which a platform
# cannot see before launch. So the curve is read for what caution costs in margin, not to pick the point
# that passed: a policy that keeps the guardrails in 80% of plausible worlds breaks them in the other 20%.

# %% [markdown]
# ## Could we have known without the truth? Off-policy evaluation
#
# A real platform never sees the planted truth. What it has is the randomized pay test. Suppose the test
# kept running in the held-out months (8% of offers moved by ±15%, never below the floor). Kernel-weighted
# inverse propensity scoring estimates the robust policy's first-offer acceptance from those test trips
# alone, and is checked here against the truth on the same trips. It can only judge trips where the
# policy's change falls inside what the test randomized; the share it cannot judge is itself a finding.

# %%
R, H = simulate.TEST_RANGE, 0.02
rng_t = np.random.default_rng(11)
n_h = len(held_df)
in_test = rng_t.random(n_h) < simulate.TEST_SHARE
shock = np.where(in_test, rng_t.uniform(-R, R, n_h), 0.0)
floor_h, sq_h = held_df["floor"].to_numpy(), held_df["offer_sq"].to_numpy()
offer_obs = np.maximum(floor_h, sq_h * (1 + shock))
accept_obs = (rng_t.random(n_h) < accept_truth(held_df, np.log(offer_obs / floor_h))).astype(float)
pol_offer = floor_h * (1 + held_df["cell"].map(robust_premium).fillna(0.0).to_numpy())
x_target, x_obs = np.log(pol_offer / sq_h), np.log(offer_obs / sq_h)
floor_shift = np.log(floor_h / sq_h)
sup = ope.supported(x_target, floor_shift, R, H)
changed = np.abs(x_target) > 0.005


def ope_change(idx):
    t_, c_ = idx[in_test[idx]], idx[~in_test[idx] & sup[idx]]
    est, _ = ope.kernel_ipw(accept_obs[t_], x_obs[t_], x_target[t_], floor_shift[t_], R, H)
    return est - accept_obs[c_].mean()


est_change = ope_change(np.arange(n_h))
est_ci = ope.bootstrap_ci(ope_change, n_h, reps=50 if FAST else 300)
hs = held_df[sup]
true_change = float(accept_truth(hs, np.log(pol_offer[sup] / floor_h[sup])).mean()
                    - accept_truth(hs, np.log(sq_h[sup] / floor_h[sup])).mean())
model_change = float(accept_model(hs, np.log(pol_offer[sup] / floor_h[sup]), mean_draw).mean()
                     - accept_model(hs, np.log(sq_h[sup] / floor_h[sup]), mean_draw).mean())
ope_check = {"bandwidth": H, "test_share": simulate.TEST_SHARE, "test_range": R, "test_trips": int(in_test.sum()),
             "share_supported": float(sup.mean()), "share_changed": float(changed.mean()),
             "share_of_changes_supported": float(sup[changed].mean()) if changed.any() else float("nan"),
             "share_of_changes_beyond_range": float((np.abs(x_target[changed]) > np.log1p(R)).mean()) if changed.any() else float("nan"),
             "acceptance_change_ope": float(est_change), "acceptance_change_ope_ci": list(est_ci),
             "acceptance_change_truth": true_change, "acceptance_change_model": model_change,
             "truth_inside_ci": bool(est_ci[0] <= true_change <= est_ci[1])}
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ope_check.items()})

# %%
_ = write_metrics("06b_policy_checks", {
    "added": "2026-11-06, after the results above; see PROJECT_PLAN.md",
    "frontier": frontier.to_dict(orient="records"), "ope": ope_check}, SOURCE)
show_verdict(
    "Checks: the price of safety and off-policy evaluation",
    [
        "**The price of safety**: " + "; ".join(
            f"guardrails in ≥{r.min_share:.0%} of draws → {r.margin_vs_today_pct:+.1f}% margin held out"
            + ("" if r.held_out_all_guardrails else " (breaks a guardrail held out)") for r in frontier.itertuples()) + ".",
        f"**Off-policy evaluation from the pay test alone** puts the robust policy's change in first-offer acceptance at "
        f"{100 * est_change:+.1f} points (95% interval {100 * est_ci[0]:+.1f} to {100 * est_ci[1]:+.1f}) on the trips it can "
        f"judge; the truth there is {100 * true_change:+.1f} and the pay model says {100 * model_change:+.1f}"
        + (", so the test alone would have confirmed it." if ope_check["truth_inside_ci"] else ", which misses the truth.")
        + f" It can judge only {ope_check['share_of_changes_supported']:.0%} of the trips whose offer the policy changes; "
        f"{ope_check['share_of_changes_beyond_range']:.0%} of the changes lie beyond the test's ±{R:.0%}. To evaluate pay "
        "policies like this one offline, the randomized slice must reach the premiums they use.",
    ],
    "the dispatch simulation below, then phase 7.",
)

# %% [markdown]
# ## Part B: dispatch simulation on real Brooklyn requests
#
# **Calibration first.** The simulator's one free input that matters is how many drivers come online
# each hour; the data has no driver records. It is calibrated so that, under today's pay, the
# simulated mean wait in **each hour** matches the real mean wait of these requests, with batched
# matching (a few rounds of
# multiplicative updates: hours with too-long simulated waits get more drivers). Driver supply that
# simply follows demand was tried first and could not reproduce the real pattern: real waits are
# longest overnight, when drivers are scarcer *relative to* demand.
#
# **Validation on what was not calibrated**: the 90th-percentile wait and the median wait by hour are
# not targeted, so how close they come is a genuine check.

# %%
req_all = pd.read_parquet(data.interim_dir() / "sim_requests.parquet")
cents = con.sql("SELECT * FROM zones ORDER BY zone").df().set_index("zone")
cents["radius_mi"] = np.sqrt(cents["area_sq_mi"] / np.pi)
cents["in_area"] = cents["borough"] == "Brooklyn"
prev = con.sql("""SELECT zone, hour + INTERVAL 1 HOUR AS hour, wait_sum_sec / NULLIF(wait_n, 0) / 60 AS prev_wait_min
                  FROM zone_hour ORDER BY zone, hour""").df()
req_all["hour_key"] = req_all["request_datetime"].dt.floor("h")
req_all = req_all.merge(prev, left_on=["PULocationID", "hour_key"], right_on=["zone", "hour"], how="left")
req_all["prev_wait_min"] = req_all["prev_wait_min"].fillna(req_all["prev_wait_min"].median())
req_all["cluster_name"] = req_all["PULocationID"].map(info["cluster_name"]).fillna("Sparse")
req_all["band"] = req_all["request_datetime"].dt.hour.map(config.BAND_OF_HOUR)
req_all = req_all[[(c, b) in seg_index for c, b in zip(req_all["cluster_name"], req_all["band"])]].copy()
req_all["minutes"] = req_all["trip_time"] / 60
req_all["fare"] = req_all["base_passenger_fare"]
from ridepay.pay import floor_pay  # noqa: E402

req_all["floor"] = floor_pay(req_all["trip_miles"], req_all["minutes"], req_all["request_datetime"], req_all["DOLocationID"])
req_all["offer_sq"] = np.maximum(req_all["driver_pay"], req_all["floor"])
req_all = req_all.join(simulate.difficulty_features(req_all, info))
req_all["seg"] = [seg_index[(c, b)] for c, b in zip(req_all["cluster_name"], req_all["band"])]
req_all["d_hat"] = req_all[accept.FEATURES].to_numpy() @ gamma_mean
req_all["q"] = np.searchsorted(edges, req_all["d_hat"]) + 1
req_all["cell"] = req_all["cluster_name"] + "|" + req_all["band"] + "|q" + req_all["q"].astype(str)
req_all["offer_policy"] = req_all["floor"] * (1 + req_all["cell"].map(robust_premium).fillna(0.0))
days = sorted(req_all["request_datetime"].dt.date.unique())
print(f"{len(req_all):,} requests on {[str(d) for d in days]}")

# %%
def day_inputs(day, seed=0):
    r = req_all[req_all["request_datetime"].dt.date == day].sort_values("request_datetime").reset_index(drop=True)
    start = pd.Timestamp(day)
    r["t_sec"] = (r["request_datetime"] - start).dt.total_seconds()
    hourly = r.groupby(r["request_datetime"].dt.hour).size().reindex(range(24), fill_value=0).to_numpy()
    sp = (r.groupby(r["request_datetime"].dt.hour).apply(lambda g: g["trip_miles"].sum() / (g["trip_time"].sum() / 3600),
                                                          include_groups=False).reindex(range(24)).ffill().bfill().to_numpy())
    idx = list(zip(r["cluster_name"], r["band"]))
    a, b = truth_tab.loc[idx, "a"].to_numpy(), truth_tab.loc[idx, "beta"].to_numpy()
    u = np.random.default_rng(100 + seed).normal(0, simulate.U_SD, len(r))   # same u for every policy (common random numbers)
    d_obs = simulate.observed_difficulty(r)

    def p_of(offer):
        return expit(a - d_obs - u + b * np.log(offer / r["floor"].to_numpy()))

    return r, hourly, sp, p_of


def simulate_day(day, offer_col, matching, supply, seed=0, batch_sec=30):
    """One simulated day. ``supply``: 24 multipliers on driver arrivals (0.25 per request in that hour)."""
    r, hourly, sp, p_of = day_inputs(day, seed)
    offer = r[offer_col].to_numpy()
    cfg = dispatch.SimConfig(matching=matching, supply_scale=1.0, seed=seed, batch_sec=batch_sec)
    res = dispatch.run(r, p_of(offer), offer, cents, hourly * 0.25 * np.asarray(supply), sp, cfg)
    out = dispatch.summarize(r, res, r["fare"].to_numpy(), offer, policy.CANCEL_COST)
    hours = r["request_datetime"].dt.hour.to_numpy()
    out["wait_by_hour"] = pd.Series(res["wait_sec"] / 60).groupby(hours).mean().reindex(range(24)).to_numpy()
    return out


real = {d: req_all[(req_all["request_datetime"].dt.date == d) & req_all["wait_valid"]] for d in days}
real_hourly = {d: (real[d].groupby(real[d]["request_datetime"].dt.hour)["wait_sec"].mean().reindex(range(24)) / 60).to_numpy()
               for d in days}
target = {d: (real[d]["wait_sec"] / 60).mean() for d in days}
t0 = time.time()
calib, calib_path = {}, []
for d in days:
    supply = np.full(24, 0.5)
    for it in range(3 if FAST else 6):
        sim_h = simulate_day(d, "offer_sq", "batch", supply)["wait_by_hour"]
        ratio = np.where(np.isfinite(sim_h / real_hourly[d]), sim_h / real_hourly[d], 1.0)
        calib_path.append({"day": str(d), "iteration": it, "mean_abs_gap_min": float(np.nanmean(np.abs(sim_h - real_hourly[d])))})
        # too long a wait -> more drivers that hour; smoothed with the neighbouring hours
        step = np.clip(ratio, 0.5, 2.0) ** 0.8
        step = (np.roll(step, 1) + 2 * step + np.roll(step, -1)) / 4
        supply = np.clip(supply * step, 0.05, 5.0)
    calib[d] = supply
print(pd.DataFrame(calib_path).pivot(index="iteration", columns="day", values="mean_abs_gap_min").round(3),
      f"\ncalibrated in {time.time() - t0:.0f}s")

# %%
val = {d: simulate_day(d, "offer_sq", "batch", calib[d]) for d in days}
fig, axes = plots.subplots(1, len(days), figsize=(11, 3.2), sharey=True)
check = []
for ax, d in zip(np.atleast_1d(axes), days):
    ax.plot(range(24), real_hourly[d], color=plots.INK, lw=1.6, label="real")
    ax.plot(range(24), val[d]["wait_by_hour"], color=plots.SERIES[0], lw=1.6, label="simulated (today's pay)")
    ax.set(title=f"Mean wait by hour, {pd.Timestamp(d):%a %d %b}", xlabel="hour")
    w = real[d]["wait_sec"] / 60
    check.append({"day": str(d), "real mean": w.mean(), "sim mean": val[d]["wait_mean_min"],
                  "real median (not targeted)": w.median(), "sim median": val[d]["wait_p50_min"],
                  "real p90 (not targeted)": w.quantile(0.9), "sim p90": val[d]["wait_p90_min"],
                  "hourly corr": float(np.corrcoef(real_hourly[d], val[d]["wait_by_hour"])[0, 1])})
np.atleast_1d(axes)[0].legend()
np.atleast_1d(axes)[0].set_ylabel("minutes")
fig.tight_layout()
data.savefig(fig, "06_sim_validation")
check = pd.DataFrame(check).set_index("day")
corr = check["hourly corr"].to_dict()
check.round(2)


# %% [markdown]
# **How to read the figure.** Mean rider wait by hour on sample days, real against simulated under today's pay. The simulator is only trusted to compare policies because it reproduces the real pattern under today's pay.

# %% [markdown]
# **The experiments**: today's pay vs the robust policy, crossed with greedy vs batched matching, on
# both days and three random seeds each (the seed changes driver arrivals and decisions; the riders'
# unobserved difficulty is held fixed across policies so comparisons are paired).

# %%
t0 = time.time()
runs = []
for d in days:
    for seed in range(2 if FAST else 3):
        for offer_col, pname in [("offer_sq", "Today's pay"), ("offer_policy", "Robust policy")]:
            for matching in ["greedy", "batch"]:
                s = simulate_day(d, offer_col, matching, calib[d], seed=seed)
                s.pop("wait_by_hour")
                runs.append({"day": str(d), "seed": seed, "pay": pname, "matching": matching, **s})
runs = pd.DataFrame(runs)
print(f"{len(runs)} runs in {time.time() - t0:.0f}s")
KPIS = ["completion", "first_offer_acceptance", "wait_mean_min", "wait_p90_min", "pickup_mean_min",
        "margin_per_request", "pay_per_trip", "earnings_per_online_hour"]
sim_table = runs.groupby(["pay", "matching"])[KPIS].mean()
sim_table.round(3)

# %%
paired = runs[runs["matching"] == "batch"].set_index(["day", "seed", "matching", "pay"])[KPIS].unstack("pay")
diff_pay = (paired.xs("Robust policy", axis=1, level=1) - paired.xs("Today's pay", axis=1, level=1))
paired_m = runs.set_index(["day", "seed", "pay", "matching"])[KPIS].unstack("matching")
diff_match = paired_m.xs("batch", axis=1, level=1) - paired_m.xs("greedy", axis=1, level=1)


def mean_ci(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return pd.Series({"mean": x.mean(), "lo": x.mean() - 2.0 * se, "hi": x.mean() + 2.0 * se})


effects = pd.concat({"Robust policy − today's pay": diff_pay.apply(mean_ci).T,
                     "Batched − greedy matching": diff_match.apply(mean_ci).T})
effects.round(3)

# %%
record = {
    "fit_months": FIT, "held_months": HELD, "cells": len(fit_table.cells), "premium_options": policy.PREMIUMS.tolist(),
    "assumptions": {"reoffer_sec": policy.REOFFER_SEC, "cancel_per_decline": policy.CANCEL_PER_DECLINE,
                    "cancel_cost": policy.CANCEL_COST, "earnings_floor": policy.EARNINGS_FLOOR},
    "fit_summary": fit_summary.round(5).to_dict(orient="index"),
    "held_summary": held_summary.round(5).to_dict(orient="index"),
    "by_borough": by_borough.round(5).reset_index().to_dict(orient="records"),
    "held_wait_slips": {k: v.round(4).to_dict(orient="records") for k, v in wait_slips.items()},
    "stability": stability, "guardrails_in_draws": {"point": point_in_draws.to_dict(), "robust": robust_in_draws.to_dict()},
    "premium_table": cells_view.round(4).to_dict(orient="records"),
    "sim": {"days": [str(d) for d in days], "supply_profile": {str(k): np.round(v, 4).tolist() for k, v in calib.items()},
            "real_mean_wait": {str(k): v for k, v in target.items()}, "hourly_corr": corr,
            "validation": check.round(4).reset_index().to_dict(orient="records"),
            "table": sim_table.round(5).reset_index().to_dict(orient="records"),
            "effects": effects.round(5).reset_index().rename(columns={"level_0": "comparison", "level_1": "kpi"}).to_dict(orient="records")},
}
_ = write_metrics("06_policy_simulation", record, SOURCE)
policy_out = cells_view[["cell", "cluster_name", "band", "q", "premium_robust", "premium_point", "premium_today"]]
_ = data.save_table(policy_out, "pay_policy")
_ = data.save_table(pd.DataFrame({"edge": edges}), "difficulty_edges")

# %% [markdown]
# ## Verdict

# %%
H = held_summary
rob, today = H.loc["Optimized (robust)"], H.loc["Today"]
ep = effects.loc["Robust policy − today's pay"]
em = effects.loc["Batched − greedy matching"]
all_ok = bool(rob[["C3_wait", "C4_acceptance", "C5_earnings"]].all())
rs = wait_slips["Optimized (robust)"]
GUARDS = ["C3_wait", "C4_acceptance", "C5_earnings"]
GUARD_NAMES = {"C3_wait": "wait", "C4_acceptance": "acceptance", "C5_earnings": "earnings"}
show_verdict(
    "Verdict: phase 6, pay policy and simulation",
    [
        f"**Held out (Apr–Jun 2025), scored against the truth, the robust policy raises margin per request by "
        f"{rob['margin_vs_today_pct']:+.1f}%** (${today['margin_per_request']:.2f} → ${rob['margin_per_request']:.2f}) "
        f"with first-offer acceptance {today['first_offer_acceptance']:.1%} → {rob['first_offer_acceptance']:.1%} and pay per "
        f"trip ${today['pay_per_trip']:.2f} → ${rob['pay_per_trip']:.2f}"
        + ("; guardrails C3–C5 all hold." if all_ok else
           "; acceptance and earnings guardrails "
           + ("hold" if rob["C4_acceptance"] and rob["C5_earnings"] else "do NOT both hold")
           + (f", and rider wait is at or below today in {len(wait_seg['Optimized (robust)']) - len(rs)} of "
              f"{len(wait_seg['Optimized (robust)'])} segments but slips in {len(rs)} (worst: {rs.iloc[0]['cluster']} "
              f"{rs.iloc[0]['band']}, {rs.iloc[0]['wait_today_sec']:.1f} → {rs.iloc[0]['wait_policy_sec']:.1f} s of extra wait "
              f"per trip). Robust to the model's uncertainty is not robust to its errors." if len(rs) else ".")),
        f"**Floor-only is a trap**: it pays least but {'breaks' if not H.loc['Floor only', ['C3_wait', 'C4_acceptance', 'C5_earnings']].all() else 'keeps'} "
        f"the guardrails (acceptance {H.loc['Floor only', 'first_offer_acceptance']:.1%}, margin "
        f"{H.loc['Floor only', 'margin_vs_today_pct']:+.1f}% once cancellations are counted).",
        f"**Robustness is insurance, at {H.loc['Optimized (point)', 'margin_vs_today_pct'] - rob['margin_vs_today_pct']:.1f} "
        f"points of margin**: the point policy (best guess only) meets the wait guardrail in "
        f"{point_in_draws['C3_wait']:.0%}, acceptance in {point_in_draws['C4_acceptance']:.0%} and earnings in "
        f"{point_in_draws['C5_earnings']:.0%} of the {D} posterior draws the robust policy is built to satisfy (the robust "
        f"policy meets all three in each of them by construction). Held out and scored against the truth, "
        + ("both policies meet all three" if H.loc[["Optimized (point)", "Optimized (robust)"], GUARDS].all(axis=None)
           else "the robust policy meets " + ("all three" if H.loc["Optimized (robust)", GUARDS].all() else "only "
                + ", ".join(GUARD_NAMES[g] for g in GUARDS if H.loc["Optimized (robust)", g]))
           + " and the point policy breaks the "
           + " and ".join(GUARD_NAMES[g] for g in GUARDS if not H.loc["Optimized (point)", g]) + " guardrail")
        + f". Cell premiums differ by {stability['mean_abs_diff_pts']:.1f} points on average across plausible "
        "worlds, so trust the policy at segment level, not cell by cell.",
        f"**In the dispatch simulation** (real Brooklyn requests; driver supply calibrated hour by hour to real waits; "
        f"untargeted p90 wait {check['sim p90'].mean():.1f} min simulated vs {check['real p90 (not targeted)'].mean():.1f} real), "
        f"the policy raises completion by {ep.loc['completion', 'mean']:+.1%} and first-offer acceptance by "
        f"{ep.loc['first_offer_acceptance', 'mean']:+.1%}, changes mean wait by {ep.loc['wait_mean_min', 'mean']:+.2f} min and "
        f"margin per request by {ep.loc['margin_per_request', 'mean']:+.2f} dollars. That matches the expected-value model for "
        f"Brooklyn ({by_borough.loc['Brooklyn', 'margin change']:+.2f} per request): the policy moves premium from Manhattan "
        f"({by_borough.loc['Manhattan', 'margin change']:+.2f}) to the outer boroughs, buying service there.",
        f"**Matching matters as much as pay.** Batching (one assignment problem every 30 s) cuts mean wait by "
        f"{-em.loc['wait_mean_min', 'mean']:.1f} min vs greedy nearest-driver, but by minimizing total pickup time it leaves "
        f"isolated requests waiting until some riders give up (completion {em.loc['completion', 'mean']:+.1%}). The next step "
        "is a cost that also values how long a request has waited (tried with a simple age weight: at today's tight supply it "
        "sent drivers on long pickups and made waits worse, so it needs tuning, not a guess).",
    ],
    "phase 7: package the policy as a quote service (FastAPI, Docker, CI/CD).",
)
