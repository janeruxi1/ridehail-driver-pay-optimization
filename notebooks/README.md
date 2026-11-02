# Notebooks: reading guide

Read them in order. Each notebook opens with its question and a short answer, ends with a
**Verdict**, and every figure has a short explanation next to it.

| # | Notebook | The question in plain words | Main figures |
|---|---|---|---|
| 01 | [Marketplace health](01_marketplace_health.ipynb) | What does the market look like, and where is it under strain? | Pay vs the floor; weekly rhythm; zone maps; zone clusters |
| 02 | [Demand forecast](02_demand_forecast.ipynb) | How many trips per zone and hour, four weeks ahead? | Error by horizon; forecast vs actual |
| 03 | [Long-wait early warning](03_long_wait_classifier.ipynb) | Will this request wait more than 10 minutes? | Calibration and precision–recall; cost by threshold |
| 04 | [Congestion fee](04_congestion_fee.ipynb) | What did New York's congestion fee do? | Exposure by zone; event study; placebo dates |
| 05 | [Pay to accept](05_pay_to_accept.ipynb) | How does driver acceptance respond to pay? | Estimates vs the planted truth; segment slopes |
| 06 | [Pay policy and simulation](06_pay_policy_simulation.ipynb) | Which pay policy, and does it hold up? | Premium map; simulator check |
| 07 | [Switchback design](07_switchback_design.ipynb) | How to test the policy live? | Power by design |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source); see the main [README](../README.md) for the data and the build command.
