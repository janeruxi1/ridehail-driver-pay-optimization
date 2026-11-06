# Decision memo: driver pay policy

**To:** Head of Marketplace · **From:** Data Science (Marketplace Health & Pricing) · **Status:** recommendation, pending a switchback test
**Scenario on NYC TLC data; acceptance decisions simulated (see README).**

## Recommendation

Replace today's premium rules with the **robust pay policy** (a premium of 0–30% above the regulatory
floor, set per zone cluster × hour band × trip difficulty), and confirm it with a **4-week
switchback test** before full rollout.

## Why

1. **Today's premiums are spent in the wrong places.** The premium above the floor is
   7.9% of driver pay, but much of it goes to easy trips that drivers accept anyway,
   and too little to the hardest trips, where pay buys faster acceptance.
2. **Pay works better than the history suggests.** Because premiums go to the trips drivers dislike,
   the raw data credits pay with about half of its true effect: 3.2 points of first-offer acceptance
   per 10% more pay, against a true average of 6.3, more at night and where drivers are scarce.
   A randomized pay test measures 5.0 points over its own ±15% range (95% interval 4.3 to 5.8; truth there: 5.6), and double
   machine learning recovers the (differently weighted) average it targets; the acceptance model
   built on both predicts what the test measured.
3. **It holds up out of sample, with one small miss.** On April–June 2025 trips it never saw, margin and
   acceptance rise and driver earnings stay within the guardrail. Rider wait is at or below today in 18 of
   20 segments; at the airports it is about 1 second per trip worse in the evening (the
   model overstates how airport drivers respond to pay):

| | Today | Robust policy |
|---|---|---|
| Margin per request | $5.74 | **$5.95 (+3.7%)** |
| First-offer acceptance | 70.7% | 71.2% |
| Trip completion | 97.3% | 97.5% |
| Extra wait from declines (s per trip) | 8.3 | 7.7 |
| Driver pay per trip | $21.43 | $21.21 |
| Driver earnings per engaged hour | $53.15 | $52.58 (-1.1%) |

4. **It moves money between boroughs.** Margin rises in Manhattan (+0.86 $ per request), where drivers
   accept anyway, and falls slightly in the outer boroughs (Brooklyn -0.14, Bronx
   -0.33), where the extra pay buys faster pickups. A dispatch simulation replaying real Brooklyn
   requests confirms the direction: first-offer acceptance +1.4 points and margin -0.06 $ per
   request there; the change in waits is within simulation noise (p90 -0.38 min, 95% interval
   -0.76 to +0.00).

## What we rejected

- **Pay the floor only.** Looks like +27% margin, but acceptance falls to
  64.8% and driver earnings per hour by 8%: it breaks all three guardrails, and drivers would leave.
- **The best-guess (non-robust) policy.** 0.6 percentage points more margin growth, but it meets
  the wait guardrail in only 1 of the 20 plausible versions of the acceptance model that the robust
  policy is built to satisfy (the robust policy meets them all by construction). Held out, both slip
  on wait in two small segments: the insurance covers the plausible versions the model can see, not
  an error shared by all of them.
- **A less cautious policy that happened to pass.** Keeping the guardrails in 80% of the plausible
  versions instead of all of them would have earned +4.1% and met every guardrail held out. That is luck
  about where the model is wrong, which nobody sees before launch; it breaks them in a fifth of the
  plausible versions.

## Risks

- The acceptance model is estimated; premiums above +15% are beyond the randomized test's range. Checked
  from the test alone, the policy's acceptance gain is confirmed
  (+3.7 points, 95% interval +0.9 to +6.5; truth +3.3), but only on the 45% of the changed trips the test can judge. **Widen the randomized slice to
  ±30% in the switchback regions** so the rest can be checked.
- **Watch the airports first.** That is where the model is least accurate and where wait slipped held out;
  the switchback's wait guardrail is read per segment, airports included.
- Drivers may learn to wait for premiums; more pay may bring more drivers online over weeks. Neither is
  in the model; both would show in the switchback test.
- Matching matters as much as pay: batched assignment cut mean wait by 9.0 minutes vs greedy in
  simulation, but left some isolated requests waiting too long; dispatch should be tuned alongside.

## Test plan (fixed before the test)

- **Design:** randomize region × 2-hour blocks, balanced within each region-day; drop the first hour after
  each switch (carryover). 12 regions.
- **Primary metric:** margin per trip; minimum detectable effect $0.210 in 4 weeks against an
  expected $+0.224 → **4 weeks**.
- **Guardrails:** mean rider wait (non-inferiority margin 0.25 min; detectable change
  0.08 min), first-offer acceptance, driver earnings per hour within 5% by region.
- **Analysis:** randomization inference (clustered standard errors over-reject in A/A simulations).
- **Decision:** ship if margin improves with all guardrails met; stop only on a guardrail breach; if
  inconclusive, extend once.

## Also learned

- **Waits are longest around 4am on Sundays, not at rush hour**: target pay by supply–demand balance, not volume.
- **The congestion fee** made traffic in the zone ~5% faster; its effect on driver pay per trip is not
  measurable with the data available. Don't book one. (A fuller study with two pre-toll years,
  `nyc-congestion-toll-causal-impact`, finds no detectable change in trips in the zone once their pre-toll decline is
  allowed for.)
- **Demand forecasts:** about half of the zone × hour error is irreducible noise; plan supply at
  cluster × hour (error 6.7%).
