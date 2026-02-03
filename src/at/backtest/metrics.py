from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(daily_returns: pd.Series, trading_days: int = 252) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    vol = r.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float(np.sqrt(trading_days) * r.mean() / vol)


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return float("nan")
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    """1-way turnover = 0.5 * sum(|w_t - w_{t-1}|) per day."""
    w = weights.fillna(0.0)
    dw = w.diff().abs().sum(axis=1)
    return 0.5 * dw
