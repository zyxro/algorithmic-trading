from __future__ import annotations

import numpy as np
import pandas as pd

from at.data.schema import OHLCVSchema


def add_log_returns(df: pd.DataFrame, s: OHLCVSchema, horizons: tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)

    for h in horizons:
        col = f"logret_{h}d"
        out[col] = g[s.close].transform(lambda x: np.log(x).diff(h))

    return out


def add_forward_returns(df: pd.DataFrame, s: OHLCVSchema, horizon: int = 1) -> pd.DataFrame:
    """Forward return target (no leakage)."""
    out = df.copy()
    g = out.groupby(s.ticker, sort=False)
    out[f"fwd_ret_{horizon}d"] = g[s.close].pct_change(periods=horizon).shift(-horizon)
    return out
