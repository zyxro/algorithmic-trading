from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> None:
    ax = equity.plot(figsize=(10, 4), title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    plt.tight_layout()


def plot_underwater(equity: pd.Series, title: str = "Underwater (Drawdown)") -> None:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    ax = dd.plot.area(figsize=(10, 3), title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
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
