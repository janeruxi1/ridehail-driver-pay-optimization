"""Quote service: dispatch sends a trip, gets back the recommended driver pay.

    uvicorn ridepay.api:app --port 8000
    curl -X POST localhost:8000/quote -H 'content-type: application/json' \
         -d '{"pickup_zone": 61, "dropoff_zone": 17, "request_time": "2025-06-10T08:15:00",
              "trip_miles": 2.4, "trip_minutes": 14, "rider_fare": 18.5}'
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .quote import DEFAULT_MODEL, QuoteModel

log = logging.getLogger("ridepay.api")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")


class TripRequest(BaseModel):
    pickup_zone: int = Field(..., ge=1, le=263, description="TLC taxi zone of the pickup")
    dropoff_zone: int = Field(..., ge=1, le=265, description="TLC taxi zone of the drop-off (265 = outside NYC)")
    request_time: datetime = Field(..., description="local New York time of the request")
    trip_miles: float = Field(..., gt=0, le=100)
    trip_minutes: float = Field(..., ge=1, le=240)
    rider_fare: float = Field(..., gt=0, le=500, description="rider fare before tolls, tips and fees")
    prev_hour_wait_min: float | None = Field(None, ge=0, le=120,
                                             description="mean rider wait in the pickup zone last hour, if known")


class QuoteResponse(BaseModel):
    model_version: str
    floor_pay: float
    recommended_pay: float
    premium_pct: float
    segment: str
    difficulty_quintile: int
    accept_probability: float | None
    completion_probability: float | None
    expected_margin: float | None
    rule: str


@lru_cache(maxsize=1)
def get_model() -> QuoteModel:
    return QuoteModel.load(os.environ.get("RIDEPAY_MODEL", DEFAULT_MODEL))


app = FastAPI(title="ridepay quote service", version="1.0.0",
              description="Recommended driver pay per trip (portfolio project; scenario on NYC TLC data).")


@app.get("/health")
def health() -> dict:
    m = get_model()
    return {"status": "ok", "model_version": m.version}


@app.post("/quote", response_model=QuoteResponse)
def quote(req: TripRequest) -> QuoteResponse:
    t0 = time.perf_counter()
    try:
        q = get_model().quote(req.pickup_zone, req.dropoff_zone, req.request_time.replace(tzinfo=None), req.trip_miles,
                              req.trip_minutes, req.rider_fare, req.prev_hour_wait_min)
    except Exception as err:  # never leak internals to the caller; log them
        log.exception("quote failed")
        raise HTTPException(status_code=500, detail="quote failed") from err
    ms = (time.perf_counter() - t0) * 1000
    log.info(json.dumps({"event": "quote", "model_version": q.model_version, "segment": q.segment,
                         "premium_pct": q.premium_pct, "latency_ms": round(ms, 2)}))
    return QuoteResponse(**q.to_dict())
