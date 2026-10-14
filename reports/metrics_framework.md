# Metric framework

Defined before any data or modelling. Section 1 scores the **decision** (is the pay policy good for
the platform, riders and drivers?). Section 2 scores the **inference** (are we allowed to believe the
numbers behind it?).

Unit of analysis: the **trip request** for pay decisions; the **zone × hour** for health, forecasting
and the congestion-fee study.

---

## 1. Decision metrics

### Primary

| Metric | Definition | Why |
|---|---|---|
| **Expected margin per trip** | Rider fare − driver pay − expected cost of slow pickups (rider cancellation × lost fare) | What the platform keeps; counts the cost of underpaying, not only the saving |

### Guardrails (must not get worse)

| Guardrail | Threshold | Constraint |
|---|---|---|
| Pay at or above the minimum pay floor | always | C1 |
| Rider wait per zone cluster × hour band (in the pay model: extra wait from declined offers) | ≤ today | C3 |
| First-offer acceptance | ≥ today | C4 |
| Driver earnings per hour, per zone group | not more than 5% below today | C5 |

### Marketplace health (real data, by zone × hour)

| Metric | Definition | Reading |
|---|---|---|
| Trips | count of completed trips | Demand |
| Rider wait time | pickup time − request time (median, 90th percentile) | Supply shortage shows up here first |
| Driver pay per mile and per minute | driver pay ÷ miles, ÷ minutes | Driver economics |
| Driver share of fare | driver pay ÷ rider fare (before tips, tolls, fees) | Take rate: the platform's margin lever |
| Tips per trip | tips | Part of driver earnings, not paid by the platform |
| Dead-head proxy | drop-offs ÷ pickups in a zone | Above 1, drivers are stranded there after drop-off |

---

## 2. Inference metrics

### Demand forecast (real)
| Metric | Bar |
|---|---|
| Weighted MAPE on a rolling-origin backtest | Beats seasonal-naive (same hour last week) |
| Forecast bias | Within ±3% |

### Congestion-fee study (real, quasi-experimental)
| Metric | Bar |
|---|---|
| Pre-period trends, inside vs. outside the zone (event study) | Indistinguishable from zero before the fee |
| Placebo tests (fake start dates, untreated zones) | Null effects |
| Effect on trips, wait times, driver pay | Reported with intervals clustered by zone |

*Set before the analysis. In the event, the pre-period gap was not flat for trips, and zone-clustered
intervals proved far too narrow (most placebo dates came out "significant"), so notebook 04 judges each
effect against the placebo distribution instead.*

### Pay-to-accept model (simulated layer)
| Metric | Bar |
|---|---|
| Recovery of the true pay effect | Inside the 95% interval |
| Naive vs. double machine learning vs. randomized slice | Reported side by side |
| Calibration | Observed vs. predicted acceptance by decile within ±2 points |
| Hierarchical model diagnostics | R̂ < 1.01, effective sample size > 400, posterior predictive check |

### Policy and optimizer
| Metric | Bar |
|---|---|
| Constraint satisfaction | Checked in tests |
| Simulated margin, acceptance, wait time vs. today's rule | With intervals, on held-out weeks of real requests |
| Stability across posterior draws | How often the recommended pay changes |

### Production
| Metric | Bar |
|---|---|
| Quote API: latency, input validation, tests | p95 latency reported; invalid input rejected |
| Reproducibility | Same input → same quote; model version in every response |

---

## 3. Out of scope

- Rider fares and surge pricing for riders.
- Pay that differs by driver for the same trip (C2).
- Long-run driver supply growth (flagged).
- Strategic driver behaviour (declining to wait for incentives): watched in the rollout test.
