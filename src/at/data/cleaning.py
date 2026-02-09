from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from at.data.ingest import load_anonymized_asset_csv
from .schema import OHLCVSchema


def apply_hampel_filter(
    df: pd.DataFrame,
    s: OHLCVSchema,
    column: str | None = None,
    window_size: int = 10,
    n_sigmas: float = 3.0,
    min_sigma_frac: float = 0.01,
    downward_only: bool = False,
) -> pd.DataFrame:
   

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    if min_sigma_frac < 0:
        raise ValueError("min_sigma_frac must be >= 0")

    out = df.copy()

    def _norm(name: str) -> str:
        return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())

    def _resolve_col(requested: str) -> str:
        if requested in out.columns:
            return requested

        # Case/format-insensitive match (e.g., Close vs close, Adj Close vs adj_close)
        requested_norm = _norm(requested)
        norm_map: dict[str, str] = {}
        for c in out.columns:
            norm_map.setdefault(_norm(c), c)
        if requested_norm in norm_map:
            return norm_map[requested_norm]

        # Common OHLCV synonyms
        synonyms = {
            "close": ("close", "adjclose", "adj_close", "adj close", "Close", "Adj Close"),
            "open": ("open", "Open"),
            "high": ("high", "High"),
            "low": ("low", "Low"),
            "volume": ("volume", "Volume"),
        }
        key = str(requested).strip().lower()
        for cand in synonyms.get(key, (requested,)):
            cand_norm = _norm(cand)
            if cand_norm in norm_map:
                return norm_map[cand_norm]

        raise KeyError(f"Column '{requested}' not found in DataFrame")

    requested = s.close if column is None else str(column)
    # Allow semantic names ("Close", "close") when a schema is provided
    semantic = str(requested).strip().lower()
    if semantic in {"open", "high", "low", "close", "volume"}:
        requested = getattr(s, semantic)

    col = _resolve_col(requested)

    x = pd.to_numeric(out[col], errors="coerce")

    # Compute per-ticker if the dataframe is panel data
    use_group = s.ticker in out.columns
    if use_group:
        g = out.groupby(s.ticker, sort=False)
        rolling_median = (
            g[col]
            .rolling(window=window_size, center=False, min_periods=1)
            .median()
            .reset_index(level=0, drop=True)
        )
        deviation = (x - rolling_median).abs()
        rolling_mad = (deviation.groupby(out[s.ticker], sort=False).rolling(window=window_size, center=False, min_periods=1).median().reset_index(level=0, drop=True))

        sigma = 1.4826 * rolling_mad
        min_sigma = float(min_sigma_frac) * rolling_median.abs()
        effective_sigma = np.maximum(sigma, min_sigma)
        group_std = g[col].transform("std")
        fallback = group_std.where(np.isfinite(group_std) & (group_std > 0), other=np.finfo(float).eps)
        effective_sigma = pd.Series(effective_sigma, index=out.index)
        effective_sigma_safe = effective_sigma.mask(
            ~np.isfinite(effective_sigma) | (effective_sigma <= 0),
            other=fallback,
        )
    else:
        rolling_median = x.rolling(window=window_size, center=False, min_periods=1).median()
        deviation = (x - rolling_median).abs()
        rolling_mad = deviation.rolling(window=window_size, center=False, min_periods=1).median()
        sigma = 1.4826 * rolling_mad
        min_sigma = float(min_sigma_frac) * rolling_median.abs()
        effective_sigma = np.maximum(sigma, min_sigma)

        series_std = float(x.dropna().std()) if x.notna().any() else float("nan")
        fallback = series_std if np.isfinite(series_std) and series_std > 0 else np.finfo(float).eps
        effective_sigma = pd.Series(effective_sigma, index=out.index)
        effective_sigma_safe = effective_sigma.mask(
            ~np.isfinite(effective_sigma) | (effective_sigma <= 0),
            other=fallback,
        )

    outlier_score = deviation / effective_sigma_safe
    is_anomaly = (outlier_score > float(n_sigmas))

    if downward_only:
        # Only consider it an anomaly if the price is BELOW the median (dip)
        is_anomaly = is_anomaly & (x < rolling_median)

    is_anomaly = is_anomaly.fillna(False)

    out["rolling_median"] = rolling_median
    out["outlier_score"] = outlier_score
    out["is_anomaly"] = is_anomaly
    return out


def plot_hampel_filter_result(
    df: pd.DataFrame,
    column: str = "Close",
    date_col: str = "Date",
    title: str | None = None,
):
    """Plot a series with Hampel anomalies highlighted.

    Expects columns produced by `apply_hampel_filter`: `is_anomaly`.
    If `date_col` exists it is used for the x-axis, otherwise the index is used.
    """

    import matplotlib.pyplot as plt

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    if "is_anomaly" not in df.columns:
        raise KeyError("Expected 'is_anomaly' column; run apply_hampel_filter first")

    x_axis = df[date_col] if date_col in df.columns else df.index
    y = pd.to_numeric(df[column], errors="coerce")
    mask = df["is_anomaly"].fillna(False).astype(bool)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_axis, y, linewidth=1.2, label=column)
    ax.scatter(x_axis[mask], y[mask], color="red", s=18, label="Anomaly")
    ax.set_title(title or f"Hampel anomalies — {column}")
    ax.set_xlabel(date_col if date_col in df.columns else "Index")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    return fig, ax


@dataclass(frozen=True)
class CleaningConfig:
    max_missing_frac_per_ticker: float = 0.10
    forbid_bfill: bool = True
    fat_finger_return_clip: float = 0.50  # +/-50%
    fat_finger_rolling_mean_window: int = 5
    min_price: float = 5.0
    min_dollar_volume: float = 1_000_000.0

    # Hampel filter controls
    hampel_window_size: int = 10
    hampel_n_sigmas: float = 3.0
    hampel_min_sigma_frac: float = 0.01
    hampel_downward_only: bool = True
    # Cleaning behavior: by default we only *flag* anomalies.
    # If `hampel_drop_anomalies` is True, rows flagged as anomalies are removed.
    # If `hampel_replace_with_median` is True, the target column is replaced by the rolling median.
    hampel_drop_anomalies: bool = False
    hampel_replace_with_median: bool = False


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

    # Liquidity filter on daily dollar volume
    dollar_vol = out[s.close] * out[s.volume].astype(float)
    out = out[dollar_vol >= cfg.min_dollar_volume]

    return out


def clean_ohlcv(df: pd.DataFrame, s: OHLCVSchema, cfg: CleaningConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CleaningConfig()

    out = df.copy()
    # out = drop_ghost_tickers(out, s, cfg)
    out = forward_fill_small_gaps(out, s, cfg)
    # out = fat_finger_fix(out, s, cfg)
    out = apply_hampel_filter(
        out,
        s=s,
        column=s.close,
        window_size=cfg.hampel_window_size,
        n_sigmas=cfg.hampel_n_sigmas,
        min_sigma_frac=cfg.hampel_min_sigma_frac,
        downward_only=cfg.hampel_downward_only,
    )

    if cfg.hampel_replace_with_median:
        mask = out["is_anomaly"].fillna(False).astype(bool)
        out.loc[mask, s.close] = out.loc[mask, "rolling_median"]
    elif cfg.hampel_drop_anomalies:
        out = out[~out["is_anomaly"].fillna(False).astype(bool)].copy()

    # out = apply_tradeable_universe_filters(out, s, cfg)
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
