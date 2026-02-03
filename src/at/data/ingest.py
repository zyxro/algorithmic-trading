from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import OHLCVSchema


@dataclass(frozen=True)
class IngestConfig:
    date_col_candidates: tuple[str, ...] = ("date", "Date", "timestamp", "Datetime")
    ticker_col_candidates: tuple[str, ...] = ("ticker", "Ticker", "symbol", "Symbol")
    open_col_candidates: tuple[str, ...] = ("open", "Open")
    high_col_candidates: tuple[str, ...] = ("high", "High")
    low_col_candidates: tuple[str, ...] = ("low", "Low")
    close_col_candidates: tuple[str, ...] = ("close", "Close", "adj_close", "Adj Close")
    volume_col_candidates: tuple[str, ...] = ("volume", "Volume")


def _pick_column(cols: Iterable[str], candidates: tuple[str, ...]) -> str:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    raise ValueError(f"Could not find any of {candidates} in columns: {sorted(colset)}")


def load_ohlcv_csv(path: str | Path, cfg: IngestConfig | None = None) -> tuple[pd.DataFrame, OHLCVSchema]:
    """Load raw OHLCV and standardize column names.

    Returns (df, schema) where df columns are standardized to schema defaults.
    Expected standardized columns: date, ticker, open, high, low, close, volume
    """

    cfg = cfg or IngestConfig()
    path = Path(path)
    df = pd.read_csv(path)

    date_col = _pick_column(df.columns, cfg.date_col_candidates)
    ticker_col = _pick_column(df.columns, cfg.ticker_col_candidates)
    open_col = _pick_column(df.columns, cfg.open_col_candidates)
    high_col = _pick_column(df.columns, cfg.high_col_candidates)
    low_col = _pick_column(df.columns, cfg.low_col_candidates)
    close_col = _pick_column(df.columns, cfg.close_col_candidates)
    volume_col = _pick_column(df.columns, cfg.volume_col_candidates)

    schema = OHLCVSchema()

    df = df.rename(
        columns={
            date_col: schema.date,
            ticker_col: schema.ticker,
            open_col: schema.open,
            high_col: schema.high,
            low_col: schema.low,
            close_col: schema.close,
            volume_col: schema.volume,
        }
    )

    df[schema.date] = pd.to_datetime(df[schema.date], utc=False, errors="coerce")
    if df[schema.date].isna().any():
        raise ValueError("Found unparsable dates after conversion to datetime")

    df[schema.ticker] = df[schema.ticker].astype(str)

    # Sort for all downstream groupby logic
    df = df.sort_values([schema.ticker, schema.date]).reset_index(drop=True)
    return df, schema


def load_anonymized_asset_csv(
    path: str | Path,
    ticker: str,
    cfg: IngestConfig | None = None,
) -> tuple[pd.DataFrame, OHLCVSchema]:
    """Load an anonymized single-asset CSV (no ticker column) and attach `ticker`.

    Expected columns include Date/Open/High/Low/Close/Volume (case-insensitive via candidates).
    """

    cfg = cfg or IngestConfig()
    path = Path(path)
    df = pd.read_csv(path)

    date_col = _pick_column(df.columns, cfg.date_col_candidates)
    open_col = _pick_column(df.columns, cfg.open_col_candidates)
    high_col = _pick_column(df.columns, cfg.high_col_candidates)
    low_col = _pick_column(df.columns, cfg.low_col_candidates)
    close_col = _pick_column(df.columns, cfg.close_col_candidates)
    volume_col = _pick_column(df.columns, cfg.volume_col_candidates)

    schema = OHLCVSchema()
    df = df.rename(
        columns={
            date_col: schema.date,
            open_col: schema.open,
            high_col: schema.high,
            low_col: schema.low,
            close_col: schema.close,
            volume_col: schema.volume,
        }
    )

    df[schema.date] = pd.to_datetime(df[schema.date], utc=False, errors="coerce")
    if df[schema.date].isna().any():
        raise ValueError(f"Found unparsable dates in: {path}")

    df[schema.ticker] = str(ticker)
    df = df.sort_values([schema.ticker, schema.date]).reset_index(drop=True)
    return df, schema
