from __future__ import annotations

import numpy as np
import pandas as pd

from at.data.schema import OHLCVSchema


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def add_rsi(df: pd.DataFrame, s: OHLCVSchema, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)

    def _rsi(close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    out[f"rsi_{period}"] = g[s.close].transform(_rsi)
    return out


def add_macd(df: pd.DataFrame, s: OHLCVSchema, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)

    def _macd(close: pd.Series) -> pd.DataFrame:
        ema_fast = _ema(close, fast)
        ema_slow = _ema(close, slow)
        macd = ema_fast - ema_slow
        sig = _ema(macd, signal)
        hist = macd - sig
        return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})

    macd_df = g[s.close].apply(_macd).reset_index(level=0, drop=True)
    out[f"macd_{fast}_{slow}"] = macd_df["macd"].values
    out[f"macd_signal_{signal}"] = macd_df["macd_signal"].values
    out[f"macd_hist_{fast}_{slow}_{signal}"] = macd_df["macd_hist"].values
    return out


def add_rolling_volatility(df: pd.DataFrame, s: OHLCVSchema, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)
    daily_ret = g[s.close].pct_change()
    out[f"vol_{window}d"] = daily_ret.groupby(out[s.ticker], sort=False).rolling(window, min_periods=window).std().reset_index(level=0, drop=True)
    return out


def add_atr(df: pd.DataFrame, s: OHLCVSchema, window: int = 14) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)

    prev_close = g[s.close].shift(1)
    tr = pd.concat(
        [
            (out[s.high] - out[s.low]).abs(),
            (out[s.high] - prev_close).abs(),
            (out[s.low] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out[f"atr_{window}"] = tr.groupby(out[s.ticker], sort=False).rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    return out


def add_vwap_ratio(df: pd.DataFrame, s: OHLCVSchema, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)

    typical = (out[s.high] + out[s.low] + out[s.close]) / 3.0
    pv = typical * out[s.volume].astype(float)

    roll_pv = pv.groupby(out[s.ticker], sort=False).rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    roll_v = out[s.volume].astype(float).groupby(out[s.ticker], sort=False).rolling(window, min_periods=window).sum().reset_index(level=0, drop=True)
    vwap = roll_pv / roll_v.replace(0, np.nan)

    out[f"vwap_{window}"] = vwap
    out[f"close_to_vwap_{window}"] = out[s.close] / vwap
    return out


def add_volume_spike(df: pd.DataFrame, s: OHLCVSchema, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)

    vol = out[s.volume].astype(float)
    vol_ma = g[s.volume].transform(lambda x: x.astype(float).rolling(window, min_periods=window).mean())
    out[f"vol_spike_{window}"] = vol / vol_ma
    return out