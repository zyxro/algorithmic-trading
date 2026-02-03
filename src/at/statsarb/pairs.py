from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def top_correlated_pairs(
    price_panel: pd.DataFrame,
    top_k: int = 50,
    method: str = "spearman",
    min_overlap: int = 200,
) -> pd.DataFrame:
    """Baseline: find highly correlated pairs on returns.

    price_panel: index=date, columns=tickers, values=close
    """

    rets = price_panel.pct_change()
    corr = rets.corr(method=method, min_periods=min_overlap)

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            c = corr.loc[a, b]
            if np.isfinite(c):
                pairs.append((a, b, float(c)))

    out = pd.DataFrame(pairs, columns=["a", "b", "corr"])
    out = out.sort_values("corr", ascending=False).head(top_k).reset_index(drop=True)
    return out


def cointegration_scan(
    price_panel: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    max_pairs: int = 50,
) -> pd.DataFrame:
    """Test cointegration on candidate pairs (Engle-Granger)."""

    rows = []
    for _, r in candidate_pairs.head(max_pairs).iterrows():
        a, b = r["a"], r["b"]
        s1 = price_panel[a].dropna()
        s2 = price_panel[b].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 200:
            continue
        stat, pval, _ = coint(s1.loc[idx], s2.loc[idx])
        rows.append({"a": a, "b": b, "coint_pvalue": float(pval), "coint_stat": float(stat), "corr": float(r.get("corr", np.nan))})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("coint_pvalue").reset_index(drop=True)
