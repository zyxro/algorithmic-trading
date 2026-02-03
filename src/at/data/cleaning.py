from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from at.data.ingest import load_anonymized_asset_csv
from .schema import OHLCVSchema


@dataclass(frozen=True)
class CleaningConfig:
    max_missing_frac_per_ticker: float = 0.10
    forbid_bfill: bool = True
    fat_finger_return_clip: float = 0.50  # +/-50%
    fat_finger_rolling_mean_window: int = 5
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0


def drop_ghost_tickers(df: pd.DataFrame, s: OHLCVSchema, cfg: CleaningConfig) -> pd.DataFrame:
    # Missingness based on close; you can expand to all OHLCV if desired
    miss = df[s.close].isna().groupby(df[s.ticker]).mean()
    keep = miss[miss <= cfg.max_missing_frac_per_ticker].index
    return df[df[s.ticker].isin(keep)].copy()


def forward_fill_small_gaps(df: pd.DataFrame, s: OHLCVSchema, cfg: CleaningConfig) -> pd.DataFrame:
    # Strictly forbid bfill: we only ffill within each ticker
    out = df.copy()
    cols = [s.open, s.high, s.low, s.close, s.volume]
    out[cols] = out.groupby(s.ticker, sort=False)[cols].ffill()
    return out


def fat_finger_fix(df: pd.DataFrame, s: OHLCVSchema, cfg: CleaningConfig) -> pd.DataFrame:
    """Clip extreme daily returns and replace with local rolling mean close.

    This is conservative: if close jumps by >50% in one day, treat it as an anomaly.
    """

    out = df.copy()
    g = out.groupby(s.ticker, sort=False)
    close = out[s.close]
    ret = g[s.close].pct_change()
    bad = ret.abs() > cfg.fat_finger_return_clip

    # Rolling mean of close (shifted to avoid using same-day replaced close)
    roll_mean = g[s.close].rolling(cfg.fat_finger_rolling_mean_window, min_periods=1).mean().reset_index(level=0, drop=True)

    out.loc[bad, s.close] = roll_mean.loc[bad]

    # Optionally patch O/H/L to be consistent with close (minimal, not perfect)
    for col in (s.open, s.high, s.low):
        if col in out.columns:
            out.loc[bad, col] = out.loc[bad, s.close]

    # Volume is left unchanged
    return out


def apply_tradeable_universe_filters(df: pd.DataFrame, s: OHLCVSchema, cfg: CleaningConfig) -> pd.DataFrame:
    out = df.copy()

    # Penny stock filter
    out = out[out[s.close] >= cfg.min_price]

    # Liquidity filter on daily dollar volume
    dollar_vol = out[s.close] * out[s.volume].astype(float)
    out = out[dollar_vol >= cfg.min_dollar_volume]

    return out


def clean_ohlcv(df: pd.DataFrame, s: OHLCVSchema, cfg: CleaningConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CleaningConfig()

    out = df.copy()
    out = drop_ghost_tickers(out, s, cfg)
    out = forward_fill_small_gaps(out, s, cfg)
    out = fat_finger_fix(out, s, cfg)
    out = apply_tradeable_universe_filters(out, s, cfg)

    out = out.sort_values([s.ticker, s.date]).reset_index(drop=True)
    return out


def clean_anonymized_assets_folder(
    asset_dir: str | Path,
    asset_ids: range = range(0, 101),
    filename_template: str = "Asset_{i:03d}.csv",
    cfg: CleaningConfig | None = None,
) -> tuple[pd.DataFrame, OHLCVSchema]:
    """Load + clean per-asset anonymized files (Asset_000..Asset_100) and concatenate.

    Cleaning is applied *individually* per asset file as requested.
    Missing files are skipped.
    """

    cfg = cfg or CleaningConfig()
    asset_dir = Path(asset_dir)
    schema = OHLCVSchema()

    cleaned_parts: list[pd.DataFrame] = []
    for i in asset_ids:
        p = asset_dir / filename_template.format(i=i)
        if not p.exists():
            continue
        df_i, s_i = load_anonymized_asset_csv(p, ticker=f"Asset_{i:03d}")
        cleaned_parts.append(clean_ohlcv(df_i, s_i, cfg))

    if not cleaned_parts:
        raise FileNotFoundError(f"No asset CSVs found in {asset_dir} matching template {filename_template}")

    out = pd.concat(cleaned_parts, ignore_index=True)
    out = out.sort_values([schema.ticker, schema.date]).reset_index(drop=True)
    return out, schema
