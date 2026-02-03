from __future__ import annotations

from pathlib import Path

import pandas as pd

from at.data.cleaning import CleaningConfig, clean_anonymized_assets_folder, clean_ohlcv
from at.data.ingest import load_ohlcv_csv
from at.features.build import build_feature_frame
from at.utils.paths import get_paths


def main() -> None:
    paths = get_paths()

    cfg = CleaningConfig()
    raw_path = paths.data_raw / "daily_prices.csv"
    asset_dir = paths.data_raw / "anonymized_data"

    if raw_path.exists():
        df, schema = load_ohlcv_csv(raw_path)
        df_clean = clean_ohlcv(df, schema, cfg)
    elif asset_dir.exists():
        # Load Asset_000..Asset_100 individually, clean individually, then concatenate
        df_clean, schema = clean_anonymized_assets_folder(asset_dir=asset_dir, cfg=cfg)
    else:
        raise FileNotFoundError(
            f"Expected either {raw_path} or a folder at {asset_dir} containing Asset_###.csv files"
        )

    feats = build_feature_frame(df_clean, schema)

    out_path = paths.data_processed / "features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
