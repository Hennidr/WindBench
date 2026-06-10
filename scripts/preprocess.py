"""Preprocess raw farm Parquet files and save feature-engineered versions.

Usage
-----
    python scripts/preprocess.py --raw-dir data/raw --out-dir data/processed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from windbench.data.loader import load_all_farms
from windbench.data.preprocessing import make_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw farm Parquet files into feature-engineered datasets."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw .parquet files")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for processed files")
    parser.add_argument("--target", default="energy_total", help="Name of the production target column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading farms from {raw_dir} ...")
    farms = load_all_farms(raw_dir, target_col=args.target)
    print(f"Found {len(farms)} farm(s): {list(farms)}")

    for name, df in farms.items():
        print(f"  [{name}] {len(df)} rows → engineering features ...", end=" ")
        df_feat = make_features(df, target_col=args.target)
        out_path = out_dir / f"{name}.parquet"
        df_feat.to_parquet(out_path)
        print(f"{len(df_feat)} rows → saved to {out_path}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
