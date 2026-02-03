from __future__ import annotations

from pathlib import Path

import pandas as pd

from at.models.meta_label import fit_meta_label_model, predict_meta_probs
from at.models.signals import logic_sieve_signals
from at.utils.paths import get_paths


def main() -> None:
    paths = get_paths()
    feat_path = paths.data_processed / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Run scripts/run_features.py first. Missing: {feat_path}")

    df = pd.read_parquet(feat_path)

    # Baseline pipeline A: logic sieve -> meta labels
    df["signal_a"] = logic_sieve_signals(df)

    # Meta-label target: was the signal directionally correct?
    # For a BUY signal, label=1 if next-day return > 0 else 0.
    df["meta_y"] = ((df["signal_a"] == 1) & (df["fwd_ret_1d"] > 0)).astype(int)

    # Use only rows where we had a candidate signal
    cand = df[df["signal_a"] == 1].copy()

    # Minimal feature set for meta model; you can expand later
    feature_cols = [
        "vol_20d",
        "atr_14",
        "vol_spike_20",
        "close_to_vwap_20",
        "rsi_14",
        "macd_hist_12_26_9",
        "vol_x_mom",
    ]
    feature_cols = [c for c in feature_cols if c in cand.columns]

    cand = cand.dropna(subset=feature_cols + ["meta_y", "date"])

    # Time split (simple): last 2 years for test is handled in the notebook; here we just train on early chunk.
    # You will replace this with walk-forward in notebook 02.
    cand = cand.sort_values("date")
    cut = int(len(cand) * 0.7)
    train = cand.iloc[:cut]
    test = cand.iloc[cut:]

    model = fit_meta_label_model(train[feature_cols], train["meta_y"])
    test_probs = predict_meta_probs(model, test[feature_cols])

    out = test[["date", "ticker"]].copy()
    out["meta_prob"] = test_probs

    out_path = paths.data_processed / "meta_probs.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
