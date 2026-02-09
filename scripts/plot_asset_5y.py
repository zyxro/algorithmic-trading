#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    asset_number: int
    column: str
    years: int
    data_dir: Path
    output: Path | None
    show: bool


def _asset_path(asset_number: int, data_dir: Path) -> Path:
    if asset_number <= 0:
        raise ValueError("asset_number must be a positive integer")
    return data_dir / f"Asset_{asset_number:03d}.csv"


def _pick_default_column(df: pd.DataFrame) -> str:
    for candidate in ("Close", "Adj Close", "Open", "High", "Low"):
        if candidate in df.columns:
            return candidate

    numeric_cols = [c for c in df.columns if c != "Date" and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]

    raise ValueError("No numeric column found to plot")


def plot_asset_last_years(cfg: PlotConfig) -> Path | None:
    csv_path = _asset_path(cfg.asset_number, cfg.data_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Asset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError(f"Expected a 'Date' column in {csv_path.name}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=False)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    if df.empty:
        raise ValueError(f"No valid dates found in {csv_path.name}")

    column = cfg.column
    if column.lower() == "auto":
        column = _pick_default_column(df)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in {csv_path.name}. Available: {', '.join(df.columns)}"
        )

    end_date = df["Date"].max()
    start_date = end_date - pd.DateOffset(years=cfg.years)
    df_5y = df[df["Date"] >= start_date].copy()
    if df_5y.empty:
        raise ValueError(f"No rows found in the last {cfg.years} years for {csv_path.name}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df_5y["Date"], df_5y[column], linewidth=1.2)
    ax.set_title(f"Asset_{cfg.asset_number:03d} — {column} (last {cfg.years} years)")
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()

    saved_to: Path | None = None
    if cfg.output is not None:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.output, dpi=150, bbox_inches="tight")
        saved_to = cfg.output

    if cfg.show:
        plt.show()

    plt.close(fig)
    return saved_to


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot an asset's values over the last N years (default: 5)."
    )
    parser.add_argument(
        "asset_number",
        type=int,
        help="Asset file number (e.g., 1 -> Asset_001.csv)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years to plot ending at the latest date (default: 5)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="Close",
        help="Column to plot (default: Close). Use 'auto' to pick a sensible numeric column.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/anonymized_data"),
        help="Directory containing Asset_XXX.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (e.g., reports/figures/asset_001_close_5y.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a window; useful in headless runs",
    )

    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path(
            f"reports/figures/asset_{args.asset_number:03d}_{args.column.lower()}_{args.years}y.png"
        )

    cfg = PlotConfig(
        asset_number=args.asset_number,
        column=args.column,
        years=args.years,
        data_dir=args.data_dir,
        output=output,
        show=not args.no_show,
    )

    saved_to = plot_asset_last_years(cfg)
    if saved_to is not None:
        print(f"Saved plot to: {saved_to}")


if __name__ == "__main__":
    main()
