"""Single bounded Alpaca IEX market-price adapter."""

from __future__ import annotations

from datetime import date
import math
import os
from typing import Any

from crew.research.common import (
    ExternalResearchConfigurationError,
    ExternalResearchError,
    _RequestBudget,
    _request_json,
)
from crew.schemas import ResearchSource


DEFAULT_ALPACA_DATA_API_BASE = "https://data.alpaca.markets"


def _market_data(
    ticker: str,
    *,
    budget: _RequestBudget,
) -> dict[str, Any]:
    api_key_id = os.getenv("ALPACA_API_KEY_ID", "").strip()
    secret_key = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
    if not api_key_id or not secret_key:
        raise ExternalResearchConfigurationError(
            "ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required for "
            "market reference data."
        )
    base = os.getenv(
        "ALPACA_DATA_API_BASE",
        DEFAULT_ALPACA_DATA_API_BASE,
    ).strip().rstrip("/")
    encoded_ticker = ticker.replace("-", ".")
    snapshot = _request_json(
        f"{base}/v2/stocks/{encoded_ticker}/snapshot?feed=iex",
        budget=budget,
        headers={
            "APCA-API-KEY-ID": api_key_id,
            "APCA-API-SECRET-KEY": secret_key,
        },
    )
    latest_trade = snapshot.get("latestTrade", {})
    daily_bar = snapshot.get("dailyBar", {})
    previous_bar = snapshot.get("prevDailyBar", {})
    observations = [
        observation
        for observation in (latest_trade, daily_bar, previous_bar)
        if isinstance(observation, dict)
    ]
    selected: tuple[dict[str, Any], float] | None = None
    for observation in observations:
        raw_price = observation.get("p", observation.get("c"))
        try:
            candidate = float(raw_price)
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate) and candidate > 0:
            selected = (observation, candidate)
            break
    price = selected[1] if selected else None
    if price is None:
        raise ExternalResearchError(
            "Alpaca market snapshot did not contain a valid reference price."
        )

    timestamp = selected[0].get("t") if selected else None
    price_date = None
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        try:
            price_date = date.fromisoformat(timestamp[:10])
        except ValueError:
            # Keep a valid reference price even if a provider timestamp changes
            # shape; the missing date remains explicit in the research schema.
            price_date = None
    return {
        "share_price": price,
        "share_price_date": price_date,
        "source": ResearchSource(
            source_id="market_alpaca_iex",
            title=f"{ticker} IEX market-data snapshot",
            publisher="Alpaca",
            url="https://docs.alpaca.markets/us/reference/stocksnapshotsingle",
            published_at=price_date,
            source_type="IEX market data via Alpaca",
        ),
    }
