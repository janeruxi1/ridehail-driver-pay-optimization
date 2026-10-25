# Model card: ridepay quote model

| | |
|---|---|
| **What it does** | Recommends the driver pay for a trip: the NYC minimum pay for its miles and minutes, plus a premium of 0–30% chosen by a constrained optimizer |
| **Served by** | `POST /quote` (FastAPI, `src/ridepay/api.py`); artifact `models/quote_model.json` |
| **Version** | `model_version` in the artifact (content hash) and in every API response |
| **Status** | A scenario on public data, not a deployed system |

## Inputs and outputs

Input: pickup zone, drop-off zone, request time, trip miles and minutes, rider fare, and optionally
the mean rider wait in the pickup zone over the last hour.

Output: floor pay, recommended pay, premium %, segment (zone cluster × hour band), difficulty
quintile, predicted first-offer acceptance, completion probability and expected margin.

## How it was built

1. **Segments** (phase 1): k-means on 254 NYC taxi zones' behaviour (volume, daily shape, waits,
   trip economics, drop-off imbalance) over January 2024 – June 2025 of real TLC trip records,
   plus airports and sparse zones set by rule.
2. **Acceptance model** (phase 5): Bayesian hierarchical logistic regression of first-offer
   acceptance on log(pay ÷ floor) and three difficulty features (short trip, destination dead-heading,
   supply tightness in the last hour), with hour bands partially pooled within each zone cluster.
   **Acceptance decisions are simulated** (the public data doesn't record them), from a planted model,
   on real trips with their real pay. The model recovered the planted slopes (90% intervals cover the
   truth in 20 of 20 segments) and predicts what a randomized pay test measured.
3. **Policy** (phase 6): one premium per cell (segment × difficulty quintile), chosen by mixed-integer
   programming to maximize expected margin with guardrails on rider wait (C3), first-offer acceptance
   (C4) and driver earnings per hour (C5), imposed under 20 posterior draws at once (robust version).

## Evaluation

- Held out (April–June 2025 trips, scored against the planted truth): margin per request +3.7% vs
  today's pay, with all guardrails met; the non-robust policy violated the wait guardrail in 90% of
  the 20 posterior draws the robust policy is constrained to satisfy, and on the held-out months too.
- Dispatch simulation on two real days of Brooklyn requests: first-offer acceptance +1.3 points,
  margin −$0.04 per request in Brooklyn, where the policy deliberately spends margin to buy service;
  the change in p90 wait (−0.27 min, 95% interval −0.60 to +0.06) is within simulation noise
  (`reports/metrics/06_policy_simulation.json`).
- Quote latency: about 1 ms per quote in-process (p95 under 1.5 ms on a laptop CPU).

## Constraints enforced in code

- C1: pay ≥ the floor and ≤ 1.30 × the floor, enforced in `QuoteModel.quote` whatever the artifact
  says (tested in `tests/test_api.py`).
- C2: the quote depends only on the trip and its context; there is no driver input.
- Zones outside the segments (sparse zones) get the floor (`rule = "default (no premium)"`).

## Limitations and risks

- **The acceptance model is trained on simulated decisions.** Its parameters are only as real as the
  planted model; on a real platform it would be refitted on logged offers and validated against a
  randomized pay test, as done here.
- Treats all app companies in the TLC records as one marketplace.
- Static policy: supply response over weeks (more drivers online when pay rises) and strategic driver
  behaviour (waiting for premiums) are not modelled; the switchback test (phase 8) is designed to
  catch them.
- Floor rates are the per-trip version of a standard that is enforced on weekly earnings.
- Premiums above 15% are outside the range of the randomized test: extrapolation.

## Monitoring (if deployed)

- Input drift: share of requests per segment and difficulty quintile vs training.
- Calibration: predicted vs observed first-offer acceptance by decile, weekly.
- Guardrails: wait, acceptance and earnings per hour by segment, against the thresholds fixed before the test.
- Keep a small randomized pay slice running to re-check the causal slope.
