from __future__ import annotations

import pandas as pd


def logic_sieve_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    rsi_col: str = "rsi_14",
    ma_window: int = 20,
    rsi_overbought: float = 70.0,
    ticker_col: str = "ticker",
) -> pd.Series:
    """High-recall heuristic from the Plan.

    Signal is 1 for BUY candidates, else 0.
    Note: MA is computed per ticker.
    """

    g = df.groupby(ticker_col, sort=False)
    ma = g[price_col].transform(lambda x: x.rolling(ma_window, min_periods=ma_window).mean())

    cond = (df[price_col] > ma) & (df[rsi_col] < rsi_overbought)
    return cond.astype(int)


def mean_reversion_sieve_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    rsi_col: str = "rsi_14",
    ticker_col: str = "ticker",
    rsi_oversold: float = 30.0,
    long_ma_window: int = 200,
    volume_col: str = "volume",
    volume_ma_window: int = 20,
    volume_multiplier: float | None = 1.5,
) -> pd.Series:
    """Mean-reversion 'dip buy' sieve.

    New logic (your Step A/B):
    - RSI < 30 (oversold)
    - Price > 200D MA (long-term uptrend)
    - Volume > 1.5 * 20D avg volume (conviction filter)

    Returns 1 for BUY candidates, else 0.
    """

    g = df.groupby(ticker_col, sort=False)

    ma_long = g[price_col].transform(
        lambda x: x.rolling(long_ma_window, min_periods=long_ma_window).mean()
    )
    cond = (df[rsi_col] < rsi_oversold) & (df[price_col] > ma_long)

    # Optional conviction booster: require volume capitulation.
    if volume_multiplier is not None:
        vol_ma = g[volume_col].transform(
            lambda x: x.astype(float)
            .rolling(volume_ma_window, min_periods=volume_ma_window)
            .mean()
        )
        cond = cond & (df[volume_col].astype(float) > (float(volume_multiplier) * vol_ma))
    return cond.astype(int)
