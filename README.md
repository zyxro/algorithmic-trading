# Algorithmic Trading Pipeline (Precog Task)

This repo is structured to match the task’s 4 parts:
1) Feature Engineering & Data Cleaning
2) Model Training & Strategy Formulation
3) Backtesting & Performance Analysis
4) Statistical Arbitrage Overlay

## Expected data
- Put the raw CSV here: `data/raw/daily_prices.csv`
- Outputs are written to:
  - `data/interim/` (cleaned but still “market-like”)
  - `data/processed/` (features + targets)

## Where to work
Notebooks (submission deliverables):
- `notebooks/01_data_cleaning_features.ipynb`
- `notebooks/02_modeling_meta_labeling.ipynb`
- `notebooks/03_backtest_analysis.ipynb`
- `notebooks/04_statsarb_overlay.ipynb`

Implementation code (importable modules):
- `src/at/data/` ingestion + cleaning
- `src/at/features/` feature engineering + target building
- `src/at/models/` signal model + meta-label model
- `src/at/backtest/` realistic simulation + metrics
- `src/at/statsarb/` pairs / groups discovery + overlay idea
- `src/at/viz/` plotting helpers

CLI entrypoints (optional, but nice for reproducibility):
- `scripts/run_features.py`
- `scripts/run_train.py`
- `scripts/run_backtest.py`

## Quick start
1. Create a venv and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Open the notebooks in order.

## Notes on leakage
- Never use backward fill.
- Targets must be strictly forward-looking (e.g., next-day return).
- Train/test must be time-split; any CV must be time-aware (walk-forward / embargo).
