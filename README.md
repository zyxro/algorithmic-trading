# Precog Algorithmic Trading Task - 2026

This repository contains an end-to-end algorithmic trading pipeline developed for the Precog Recruitment Task. It transforms raw anonymized stock data into trading signals using a Meta-Labeling approach and backtests the strategy.

## Project Structure

```
.
├── notebooks/          # Jupyter notebooks for exploration and reporting
│   ├── 00_end_to_end_pipeline.ipynb  # Main pipeline demonstration
│   ├── 01_data_cleaning_features.ipynb
│   ├── 02_modeling_meta_labeling.ipynb
│   ├── 03_backtest_analysis.ipynb
│   └── 04_statsarb_overlay.ipynb
├── scripts/            # Python scripts for running the pipeline components
│   ├── run_features.py   # Step 1: Feature Engineering
│   ├── run_train.py      # Step 2: Model Training
│   ├── run_backtest.py   # Step 3: Backtesting
│   └── plot_asset_5y.py  # Utility for data visualization
├── src/at/             # Core package code
│   ├── data/           # Data loading and cleaning
│   ├── features/       # Feature extraction logic
│   ├── models/         # ML models (Meta-Labeling)
│   ├── backtest/       # Backtesting engine
│   └── statsarb/       # Statistical Arbitrage logic
├── data/               # Data storage (ignored by git usually)
│   ├── raw/            # Raw CSVs
│   ├── processed/      # Parquet files with features
├── reports/            # Generated figures and analysis
└── requirements.txt    # Project dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd algorithmic-trading
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/Mac
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies and the package:**
    This installs the dependencies and the local `at` package in editable mode.
    ```bash
    pip install -e .
    pip install -r requirements.txt
    ```

## Usage

The pipeline handles data in three main stages: Feature Engineering, Model Training, and Backtesting. You can run these using the provided scripts.

### 1. Feature Engineering
Reads raw CSV data, cleans it, generates technical features, and saves the result to `data/processed/features.parquet`.
```bash
python scripts/run_features.py
```

### 2. Model Training
Loads the processed features, labels the data (triple-barrier method), and trains the Meta-Labeling model (RandomForest).
```bash
python scripts/run_train.py
```

### 3. Backtesting
Simulates the trading strategy using the trained model or logic sieve signals.
```bash
python scripts/run_backtest.py
```

### 4. Visualization
You can plot the price history of specific assets using the utility script:
```bash
# Plot last 8 years for Asset 1
python scripts/plot_asset_5y.py 1 --years 8
```

## Notebooks

For a detailed walkthrough of the methodology and results:
*   Start with **`notebooks/00_end_to_end_pipeline.ipynb`** for a complete overview.
*   Explore `notebooks/01_data_cleaning_features.ipynb` for data analysis.
*   See `notebooks/02_modeling_meta_labeling.ipynb` for model performance metrics.
*   Check `notebooks/03_backtest_analysis.ipynb` for strategy performance (Sharpe, Drawdown).

## Methodology

This solution uses a **Meta-Labeling** approach:
1.  **Logic Sieve**: A base logic filters for high-momentum / mean-reverting conditions.
2.  **Meta Model**: A Secondary ML model (RandomForest) predicts the probability of the base signal resulting in a profit.
3.  **Position Sizing**: Trades are taken only when the meta-model confidence exceeds a threshold.

Dependencies include `pandas`, `scikit-learn`, and `statsmodels`.
