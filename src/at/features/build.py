from __future__ import annotations

import pandas as pd

from at.data.schema import OHLCVSchema
from at.features.returns import add_forward_returns, add_log_returns
from at.features.technical import (
    add_atr,
    add_macd,
    add_rolling_volatility,
    add_rsi,
    add_volume_spike,
    add_vwap_ratio,
)


def build_feature_frame(df: pd.DataFrame, s: OHLCVSchema) -> pd.DataFrame:
    out = df.copy()

    # Momentum
    out = add_log_returns(out, s, horizons=(1, 5, 10))
    out = add_rsi(out, s, period=14)
    out = add_macd(out, s)

    # Volatility
    out = add_rolling_volatility(out, s, window=20)
    out = add_atr(out, s, window=14)

    # Volume
    out = add_vwap_ratio(out, s, window=20)
    out = add_volume_spike(out, s, window=20)

    # Interactions
    if "vol_20d" in out.columns and "logret_5d" in out.columns:
        out["vol_x_mom"] = out["vol_20d"] * out["logret_5d"]

    # Targets
    out = add_forward_returns(out, s, horizon=1)

    return out
