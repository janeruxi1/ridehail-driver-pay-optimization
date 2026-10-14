# Business brief: driver pay in a ride-hailing marketplace

**From:** Head of Marketplace, a ride-hailing platform in New York City (scenario)
**To:** Data Science (Marketplace Health & Pricing)
**Re:** What should we offer drivers for a trip?
**Status:** phase 0, written before any data or modelling

The data is real: New York City's Taxi & Limousine Commission (TLC) publishes every trip made
through the high-volume app companies, with request, pickup and drop-off times, zones, distance,
rider fare and driver pay. This project treats those records as the history of one platform's
marketplace. The one thing the records don't contain, drivers accepting or declining offers, is
simulated on top of the real trips (see "What is real and what is simulated").

---

## Background

A rider requests a trip; the platform offers it to a nearby driver with an **upfront pay amount**;
the driver accepts or declines; a declined trip goes to the next driver, often with a higher offer.

- **Rider side:** fares are set by the rider-pricing team. Out of scope.
- **Driver side:** **the pay offered for each trip is the decision.** New York sets a minimum pay
  standard for app drivers (a rate per mile plus a rate per minute), so there is a floor. Most trips
  are paid exactly the floor; above it, operations adds premiums when trips are hard to place.

Margin per trip = rider fare − driver pay − the cost of slow pickups (riders who cancel and don't
come back).

## The problem

- **Finance:** premiums above the floor cost about 8% of all driver pay, and nobody can say what
  they buy.
- **Operations:** late at night and in outer-borough zones, riders wait much longer for a pickup and
  some cancel.

Both can be true: paying more than needed where drivers are plentiful and too little where they
are scarce.

> **For each trip, what pay offer maximizes expected margin, while keeping rider wait times and
> acceptance at least at today's level and without cutting driver earnings per hour?**

## Why the history can't be read naively

Operations adds incentives to trips that are hard to place: short trips, trips into areas with few
return riders, late night, zones short of drivers. Those trips are also the ones drivers decline. So
in the history, **extra pay arrives together with extra difficulty**, and a naive model credits pay
with far less than it actually does. Pay is confounded with how hard the trip is.

The platform also ran a **pay test**: for four months, the offer on 8% of trips was randomly moved
up or down by up to 15% (never below the floor). It is used to check the main causal method, not as
the main answer.

## Decisions sought

1. **Marketplace health**: trips, rider wait times, pay per mile and per minute, and the share of the
   fare paid to drivers, by zone and hour. Which zones behave alike?
2. **A demand forecast** of trip requests by zone and hour for the next four weeks, for planning driver
   incentives and supply.
3. **What a real policy change did**: in January 2025 New York started charging a congestion fee on
   trips into lower Manhattan. What did it do to trips, wait times and driver pay inside the zone
   compared with outside? (A natural experiment in the real data.)
4. **Pay-to-accept curves**: how acceptance responds to pay, by zone group and time of day, with
   uncertainty.
5. **A pay policy**: the recommended offer for a trip given its features, with expected margin,
   acceptance and a downside case, checked in a dispatch simulation.
6. **A quote service** dispatch can call: send a trip, get back the recommended pay.
7. **A rollout test**: drivers are shared within an area, so a per-trip A/B test would leak; design a
   switchback test (randomize by zone and time block).

## Constraints

| # | Constraint | Why |
|---|---|---|
| C1 | Pay at or above the minimum pay floor for the trip's time and distance, and no more than 1.30× the formula pay | Regulation; budget |
| C2 | Same trip, same offer: pay may depend on the trip, zone and time, not on which driver receives it | Fairness; avoids individualized pay |
| C3 | Rider wait per zone cluster × hour band not above today's (in the pay model: the extra wait caused by declined offers) | Rider experience |
| C4 | Share of trips accepted on the first offer not below today's | Speed |
| C5 | Driver earnings per hour, per zone group, not more than 5% below today's | Driver supply: drivers stop driving if earnings fall |

## What is real and what is simulated

| Real (public TLC records) | Simulated (calibrated to the real trips) |
|---|---|
| Every trip: request, pickup, drop-off times; pickup and drop-off zone; miles; minutes | Driver accept/decline decisions on offers |
| Rider fare, driver pay (including the premiums operations actually paid), tips, tolls, fees | The randomized pay test (8% of offers, four months) |
| Rider wait time (request to pickup) | Driver availability at each moment (inferred from real demand and wait times) |
| Congestion fee start (5 January 2025) | Riders cancelling after long waits |

The simulated layer has a **planted truth** (known pay sensitivities), so the causal method can be
checked against it; everything about demand, trips, fares and pay is real, including the confounding:
the real premiums go to the trips that are hardest to place.

## Risks and open questions

- **Driver learning**: if incentives become predictable, drivers may decline to wait for them.
  Watched in the rollout test.
- **Supply response**: better pay brings more drivers online over weeks. Flagged.
- **Extrapolation**: recommendations beyond the pay test's ±15% are flagged.
- **The records combine several app companies**; treating them as one platform is a simplification.

## What "done" looks like

A pay policy dispatch can call through an API, with expected margin, acceptance and wait times and
their uncertainty; a health view of the city by zone; a demand forecast; a causal estimate of the
congestion fee's effect; and a switchback test to confirm the policy.
