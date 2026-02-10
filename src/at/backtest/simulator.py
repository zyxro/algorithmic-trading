from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from at.backtest.costs import commission_bps, sqrt_impact_slippage


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_bps: float = 10.0
    impact_k: float = 0.10
    max_gross_leverage: float = 1.0


def backtest_equal_weight(
    df: pd.DataFrame,
    signal_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "close",
    vol_col: str = "vol_20d",
    dollar_vol_col: str | None = None,
    weight_col: str | None = None,
    cfg: BacktestConfig | None = None,
) -> dict[str, pd.DataFrame | pd.Series | float]:
    """Simple realistic-ish backtest for a *universe* signal (Long/Short).

    - Each day: weights tickers with non-zero signal.
    - If weight_col is None, uses equal weights across active signals (normalized by gross count).
    - If weight_col provided, uses proportional weights.
    - Trades at close with (commission + slippage) applied.
    """

    cfg = cfg or BacktestConfig()

    d = df.copy()
    d = d.sort_values([date_col, ticker_col]).reset_index(drop=True)

    if dollar_vol_col is None:
        d["dollar_vol"] = d[price_col] * d.get("volume", 0).astype(float)
        dollar_vol_col = "dollar_vol"

    # Pivot prices and signals
    prices = d.pivot(index=date_col, columns=ticker_col, values=price_col).sort_index()
    sig = d.pivot(index=date_col, columns=ticker_col, values=signal_col).sort_index().fillna(0.0)

    # Optional per-name weights (e.g., inverse-vol sizing)
    if weight_col is not None:
        w_raw = d.pivot(index=date_col, columns=ticker_col, values=weight_col).sort_index().fillna(0.0)
    else:
        w_raw = None

    daily_ret = prices.pct_change().fillna(0.0)

    # Target weights
    if w_raw is not None:
        w_tgt = sig * w_raw
        w_sum = w_tgt.abs().sum(axis=1).replace(0, np.nan)
        w_tgt = w_tgt.div(w_sum, axis=0).fillna(0.0)
    else:
        # Normalize by count of active signals (Long + Short)
        n = sig.abs().sum(axis=1).replace(0, np.nan)
        w_tgt = sig.div(n, axis=0).fillna(0.0)

    # Enforce gross leverage cap
    gross = w_tgt.abs().sum(axis=1)
    scale = (cfg.max_gross_leverage / gross).clip(upper=1.0).fillna(1.0)
    w_tgt = w_tgt.mul(scale, axis=0)

    # Assume we rebalance daily at close
    w_prev = w_tgt.shift(1).fillna(0.0)
    turnover = 0.5 * (w_tgt - w_prev).abs().sum(axis=1)

    # Costs proxy: commission on traded notional + sqrt impact using per-name sigma and ADV
    vols = d.pivot(index=date_col, columns=ticker_col, values=vol_col).sort_index()
    adv = d.pivot(index=date_col, columns=ticker_col, values=dollar_vol_col).sort_index().fillna(0.0)

    equity = pd.Series(index=prices.index, dtype=float)
    equity.iloc[0] = cfg.initial_capital

    cost_series = pd.Series(index=prices.index, dtype=float)
    cost_series.iloc[0] = 0.0

    comm_series = pd.Series(index=prices.index, dtype=float)
    comm_series.iloc[0] = 0.0

    impact_series = pd.Series(index=prices.index, dtype=float)
    impact_series.iloc[0] = 0.0

    traded_notional_series = pd.Series(index=prices.index, dtype=float)
    traded_notional_series.iloc[0] = 0.0

    for i in range(1, len(prices.index)):
        dt = prices.index[i]
        prev_dt = prices.index[i - 1]

        eq_prev = float(equity.loc[prev_dt])
        w_prev_row = w_prev.loc[dt]
        w_tgt_row = w_tgt.loc[dt]

        # PnL from holding prev weights through today
        pnl = float((w_prev_row * daily_ret.loc[dt]).sum() * eq_prev)

        # Trading notional for rebalance
        dw = (w_tgt_row - w_prev_row).fillna(0.0)
        traded_notional = float(dw.abs().sum() * eq_prev)
        traded_notional_series.loc[dt] = traded_notional

        # Commission
        comm = commission_bps(traded_notional, bps=cfg.commission_bps)
        comm_series.loc[dt] = comm

        # Slippage/impact
        sigma_row = vols.loc[dt].fillna(0.0)
        adv_row = adv.loc[dt].fillna(0.0)

        impact_cost = 0.0
        for tkr, delta_w in dw.items():
            if delta_w == 0 or eq_prev == 0:
                continue
            order_dollar = float(delta_w * eq_prev)
            sigma = float(sigma_row.get(tkr, 0.0) or 0.0)
            adv_i = float(adv_row.get(tkr, 0.0) or 0.0)
            impact_frac = sqrt_impact_slippage(sigma=sigma, daily_dollar_vol=adv_i, order_dollar=order_dollar, k=cfg.impact_k)
            impact_cost += abs(order_dollar) * impact_frac

        impact_series.loc[dt] = impact_cost

        costs = comm + impact_cost
        cost_series.loc[dt] = costs

        equity.loc[dt] = eq_prev + pnl - costs

    strat_ret = equity.pct_change().fillna(0.0)

    return {
        "equity": equity,
        "returns": strat_ret,
        "weights": w_tgt,
        "turnover": turnover,
        "costs": cost_series,
        "commission": comm_series,
        "impact": impact_series,
        "traded_notional": traded_notional_series,
    }
