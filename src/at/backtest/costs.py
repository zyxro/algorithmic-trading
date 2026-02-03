from __future__ import annotations

import numpy as np


def commission_bps(notional: float, bps: float = 10.0) -> float:
    return abs(notional) * (bps / 10_000.0)


def sqrt_impact_slippage(
    sigma: float,
    daily_dollar_vol: float,
    order_dollar: float,
    k: float = 0.10,
) -> float:
    """Square-root market impact model.

    Returns slippage as a *fractional* price impact.
    A common toy form: impact ~ k * sigma * sqrt(|Q| / ADV)

    - sigma: daily volatility estimate (e.g., 20d stdev of returns)
    - daily_dollar_vol: close * volume
    - order_dollar: order size in dollars
    """

    adv = max(daily_dollar_vol, 1e-9)
    q = abs(order_dollar)
    return float(k * sigma * np.sqrt(q / adv))
