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


def plot_monthly_heatmap(
    returns: pd.Series,
    title: str = "Monthly Returns",
    *,
    show_total: bool = False,
    annot: bool = False,
) -> None:
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    # Pandas >= 3.0: use 'ME' (month-end) instead of deprecated 'M'
    m = (1 + r).resample("ME").prod() - 1
    table = m.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="ret")

    # Month labels for readability
    month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot = pivot.rename(columns=month_labels)

    if show_total and not pivot.empty:
        # Annual compounded return from monthly returns
        total = (1.0 + pivot.fillna(0.0)).prod(axis=1) - 1.0
        pivot = pivot.assign(Total=total)

    import seaborn as sns

    ncols = max(12, int(pivot.shape[1])) if not pivot.empty else 12
    plt.figure(figsize=(max(10, int(ncols * 0.75)), 4))
    sns.heatmap(pivot, annot=bool(annot), center=0.0, cmap="RdYlGn", fmt=".1%")
    plt.title(title)
    plt.tight_layout()
