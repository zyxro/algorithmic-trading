from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OHLCVSchema:
    date: str = "date"
    ticker: str = "ticker"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
