# Decision memo: driver pay policy

**To:** Head of Marketplace · **From:** Data Science (Marketplace Health & Pricing) · **Status:** recommendation, pending a switchback test
**Scenario on NYC TLC data; acceptance decisions simulated (see README).**

## Recommendation

Replace today's premium rules with the **robust pay policy** (a premium of 0–30% above the regulatory
floor, set per zone cluster × hour band × trip difficulty), and confirm it with an **8-week
switchback test** before full rollout.

## Why

1. **Today's premiums are spent in the wrong places.** The premium above the floor is
   7.9% of driver pay, but much of it goes to easy trips that drivers accept anyway,
   and too little to the hardest trips, where pay buys faster acceptance.
2. **Pay works better than the history suggests.** Because premiums go to the trips drivers dislike,
   the raw data credits pay with about 50% of its true effect: 3.2 points of first-offer acceptance
   per 10% more pay, against a true average of 6.3, more at night and where drivers are scarce.
   A randomized pay test measures 6.3 points, and double machine learning recovers the
   (differently weighted) average it targets; the acceptance model built on both predicts what
   the test measured.
3. **The policy meets every guardrail, out of sample.** On April–June 2025 trips it never saw:

| | Today | Robust policy |
|---|---|---|
| Margin per request | $5.76 | **$5.92 (+2.8%)** |
| First-offer acceptance | 70.6% | 71.2% |
| Trip completion | 97.3% | 97.5% |
| Extra wait from declines (s per trip) | 8.3 | 7.7 |
| Driver pay per trip | $21.48 | $21.32 |
| Driver earnings per engaged hour | $53.19 | $52.77 (-0.8%) |

4. **It moves money between boroughs.** Margin rises in Manhattan (+0.81 $ per request), where drivers
   accept anyway, and falls slightly in the outer boroughs (Brooklyn -0.18, Bronx
   -0.42), where the extra pay buys faster pickups. A dispatch simulation replaying real Brooklyn
   requests confirms the direction: first-offer acceptance +1.4 points, p90 wait
   -0.36 min, margin -0.11 $ per request there.

## What we rejected

- **Pay the floor only.** Looks like +27% margin, but acceptance falls to
  64.8% and driver earnings per hour by 8%: it breaks all three guardrails, and drivers would leave.
- **The best-guess (non-robust) policy.** 0.4 percentage points more margin growth, but it meets
  the wait guardrail in only 4 of the 20 plausible versions of the acceptance model that the robust
  policy is built to satisfy. (The robust policy meets them all by construction; held out and scored
  against the truth, both policies meet every guardrail, so the robust one is insurance against
  the model being wrong.)

## Risks

- The acceptance model is estimated; premiums above +15% are beyond the randomized test's range.
- Drivers may learn to wait for premiums; more pay may bring more drivers online over weeks. Neither is
  in the model; both would show in the switchback test.
- Matching matters as much as pay: batched assignment cut mean wait by 8.3 minutes vs greedy in
  simulation, but left some isolated requests waiting too long; dispatch should be tuned alongside.

## Test plan (fixed before the test)

- **Design:** randomize region × 2-hour blocks, balanced within each region-day; drop the first hour after
  each switch (carryover). 12 regions.
- **Primary metric:** margin per trip; minimum detectable effect $0.220 in 4 weeks against an
  expected $+0.163 → **8 weeks**.
- **Guardrails:** mean rider wait (non-inferiority margin 0.25 min; detectable change
  0.10 min), first-offer acceptance, driver earnings per hour within 5% by region.
- **Analysis:** randomization inference (clustered standard errors over-reject in A/A simulations).
- **Decision:** ship if margin improves with all guardrails met; stop only on a guardrail breach; if
  inconclusive, extend once.

## Also learned

- **Waits are longest around 4am on Sundays, not at rush hour**: target pay by supply–demand balance, not volume.
- **The congestion fee** made traffic in the zone ~5% faster; its effect on driver pay per trip is not
  measurable with the data available. Don't book one.
- **Demand forecasts:** about half of the zone × hour error is irreducible noise; plan supply at
  cluster × hour (error 6.7%).
