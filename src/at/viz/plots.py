from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> None:
    ax = equity.plot(figsize=(10, 4), title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    plt.tight_layout()


def plot_underwater(equity: pd.Series, title: str = "Underwater (Drawdown)") -> None:
    s = pd.Series(equity, dtype=float).copy()
    # Be defensive: ensure a sorted, datetime-like index for stable cummax.
    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass
    s = s.sort_index()

    peak = s.cummax()
    # Guard against 0 or non-positive peaks (common if caller passes PnL/cumret starting at 0).
    peak_pos = peak.where(peak > 0)
    dd = (s / peak_pos) - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan)

    # If percent drawdown is nonsensical (e.g., huge magnitude due to tiny peaks), fall back to $ drawdown.
    dd_min = dd.min(skipna=True)
    use_dollars = dd.dropna().empty or (pd.notna(dd_min) and float(dd_min) < -5.0)
    if use_dollars:
        dd = s - s.cummax()
        ylab = "Drawdown ($)"
    else:
        ylab = "Drawdown"

    ax = dd.plot.area(figsize=(10, 3), title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylab)
    plt.tight_layout()


def plot_monthly_heatmap(returns: pd.Series, title: str = "Monthly Returns") -> None:
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    # Pandas >= 3.0: use 'ME' (month-end) instead of deprecated 'M'
    m = (1 + r).resample("ME").prod() - 1
    table = m.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")

    import seaborn as sns

    plt.figure(figsize=(10, 4))
    sns.heatmap(pivot, annot=False, center=0.0, cmap="RdYlGn")
    plt.title(title)
    plt.tight_layout()
