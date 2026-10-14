# Driver pay in a ride-hailing marketplace: plan and design decisions

Records the decisions made before building, so the reasoning survives alongside the code.

## What this project is

A rider requests a trip; the platform offers it to a driver at an upfront pay; the driver accepts
or declines. **The pay offered is the decision.** Pay too little and riders wait or cancel; pay too
much and margin disappears; cut driver earnings and drivers stop driving.

The same structure applies wherever a platform pays a supply side to accept work: delivery (courier
pay per batch), freight brokerage (rate offered to carriers), home services (contractor payouts).

## Data

**Real:** NYC Taxi & Limousine Commission High Volume For-Hire Vehicle trip records (public, monthly
Parquet files). Each row is a trip: request, pickup and drop-off times; pickup and drop-off taxi zone;
miles; minutes; rider fare; driver pay; tips; tolls and fees. About 20 million trips a month. The pay
actually offered, including the premium above the regulatory floor, is real.

**Simulated on top of the real trips:** whether the first driver offered a trip accepted it (a
planted model with known pay sensitivities), a randomized pay test on 8% of offers for four months,
drivers' positions and shifts in the dispatch simulation, and riders giving up after declines or
long waits. The planted truth is what lets the causal methods be checked.

**Volume:** 18 months (January 2024 – June 2025), about 8.5 GB of Parquet and 360 million trips. Raw
files are processed one month at a time into zone × hour tables, an origin–destination table and a
0.5% sample, then deleted; raw data is never stored in the repository.

## Phases

Each phase ends with an executed notebook (ending in a verdict), a JSON record of its results in
`reports/metrics/`, and tests for the code it adds.

| # | Phase | Data | Main techniques | Output |
|---|---|---|---|---|
| 0 | Brief, metrics, plan | — | Problem framing: objective, five stakeholder constraints | This file, `reports/scenario_brief.md`, `reports/metrics_framework.md` |
| 1 | Pipeline and marketplace health | Real | PySpark and SQL on ~360M trips; counted cleaning rules; health metrics; clustering of zones | Zone × hour tables; health figures; zone clusters |
| 2 | Demand forecast | Real | Seasonal-naive and four-week-mean baselines; gradient boosting with origin-safe features, holidays, year-ago values; rolling-origin backtest and tuning | Four-week forecast by zone × hour |
| 3 | Long-wait early warning | Real | Classification of requests that will wait more than 10 minutes; calibration; threshold from costs | Calibrated classifier on real labels |
| 4 | What the congestion fee did | Real | Year-over-year difference-in-differences, event study, placebo-date inference, dose–response | Causal estimates on real data |
| 5 | Pay-to-accept model | Simulated on real trips | Naive vs. double machine learning vs. the randomized test; Bayesian hierarchical model by segment | Acceptance curves with intervals; recovery vs. truth |
| 6 | Pay policy, matching and simulation | Real trips and requests + simulated drivers | Constrained pay optimization (MILP) robust to posterior uncertainty; driver–trip matching as an assignment problem; discrete-event dispatch simulation | Recommended policy vs. today's pay |
| 7 | Production | — | Installable package; FastAPI quote endpoint; Docker; CI and CD (image built and published on merge); model card | `docker run` → `POST /quote` |
| 8 | Rollout test and memo | Real (sizing) | Switchback design sized by simulation on real block-to-block variation; pre-registered decision rule; decision memo | `notebooks/07_switchback_design`, `reports/decision_memo.md` |

## Rules

- No phase starts before the previous one has run end to end.
- No number appears in the README or memo unless it is read from `reports/metrics/`, written by
  the code that produced it.
- Results from the synthetic test city (used by CI) never overwrite `reports/`.

## Scope boundaries

- Rider fares and rider surge pricing: out of scope.
- No pay personalization by driver (constraint C2).
- Long-run driver supply growth and strategic driver behaviour: flagged, watched in the rollout test.
- The TLC records combine several app companies; treated as one marketplace, stated as a simplification.
