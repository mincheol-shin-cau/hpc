"""Select performance counters and split datasets by platform and mode."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anti_column import adaptive_variable_selection

METADATA_COLUMNS = [
    "Application",
    "mode",
    "cpu",
    "memory",
    "cpu_next",
    "memory_next",
    "cpu_score",
]
REQUIRED_COLUMNS = {"Application", "mode", "cpu", "memory"}
TARGET_DEVICES = ("cache", "epyc", "flat", "slx")
VARIANCE_THRESHOLDS = {
    "30": 0.30,
    "50": 0.50,
    "60": 0.60,
    "70": 0.70,
    "80": 0.80,
}


def save_by_device_and_mode(df: pd.DataFrame, out_path: Path, pca_suffix: str) -> int:
    """Write all non-empty device/mode subsets and return the file count."""
    saved = 0
    modes = df["mode"].astype("string")

    for dev in TARGET_DEVICES:
        df_dev = df[modes.str.contains(dev, case=False, na=False)]

        if df_dev.empty:
            continue

        base_name = f"cpu_{dev}_{pca_suffix}"
        dev_modes = df_dev["mode"].astype("string")

        df_multi = df_dev[dev_modes.str.contains("multi", case=False, na=False)]
        if not df_multi.empty:
            df_multi.to_csv(out_path / f"{base_name}_multi.csv", index=False)
            saved += 1

        df_single = df_dev[dev_modes.str.contains("single", case=False, na=False)]
        if not df_single.empty:
            df_single.to_csv(out_path / f"{base_name}_single.csv", index=False)
            saved += 1

    return saved


def make_pca_csvs(csv_path: Path, out_dir: Path, verbose: bool) -> None:
    csv_file = csv_path.expanduser()
    if not csv_file.exists():
        raise SystemExit(f"Input CSV not found: {csv_file}")

    out_path = out_dir.expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {csv_file.name}")
    df_raw = pd.read_csv(csv_file, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS.difference(df_raw.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise SystemExit(f"Input CSV is missing required columns: {missing_text}")

    existing_metas = [c for c in METADATA_COLUMNS if c in df_raw.columns]

    meta_cols = df_raw[existing_metas]
    feature_cols = df_raw.drop(columns=existing_metas)
    numeric_features = feature_cols.select_dtypes(include="number")
    non_numeric = sorted(set(feature_cols.columns) - set(numeric_features.columns))
    if non_numeric:
        raise SystemExit("Non-numeric feature columns found: " + ", ".join(non_numeric))
    if numeric_features.empty:
        raise SystemExit("Input CSV contains no numeric feature columns")

    print(f"  Metadata columns kept: {existing_metas}")
    print(f"  Numeric feature columns: {len(numeric_features.columns)}")
    print("-" * 50)

    full_count = save_by_device_and_mode(df_raw, out_path, "full")
    if full_count == 0:
        raise SystemExit("No recognized device/mode combinations found in 'mode'")
    print(f"  [done] Full split: {full_count} file(s)")

    for suffix, var_thresh in VARIANCE_THRESHOLDS.items():
        _, selected_features = adaptive_variable_selection(
            df=numeric_features,
            variance_threshold=var_thresh,
            corr_threshold=0.90,
            verbose=verbose,
        )

        df_selected_features = numeric_features[selected_features]

        df_final = pd.concat([meta_cols, df_selected_features], axis=1)
        saved_count = save_by_device_and_mode(df_final, out_path, f"pca{suffix}")
        print(
            f"  [done] PCA {suffix}%: {len(selected_features)} feature(s), "
            f"{saved_count} file(s)"
        )

    print(f"\n[ok] All files written under '{out_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA-loading feature selection and device/mode CSV splitting."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Merged input CSV")
    parser.add_argument(
        "--out_dir", type=Path, default=Path("pca_file"), help="Output directory"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    make_pca_csvs(args.csv, args.out_dir, args.verbose)


if __name__ == "__main__":
    main()
