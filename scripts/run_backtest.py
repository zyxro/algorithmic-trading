from __future__ import annotations

import pandas as pd

from at.backtest.metrics import annualized_sharpe, max_drawdown
from at.backtest.simulator import backtest_long_only_equal_weight
from at.models.signals import logic_sieve_signals
from at.utils.paths import get_paths


def main() -> None:
    paths = get_paths()
    feat_path = paths.data_processed / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Run scripts/run_features.py first. Missing: {feat_path}")

    df = pd.read_parquet(feat_path)

    # Demo strategy: logic sieve only (no meta filter).
    df["signal"] = logic_sieve_signals(df)

    res = backtest_long_only_equal_weight(df, signal_col="signal")

    equity = res["equity"]
    rets = res["returns"]

    print(f"Sharpe: {annualized_sharpe(rets):.3f}")
    print(f"Max Drawdown: {max_drawdown(equity):.3%}")


if __name__ == "__main__":
    main()
