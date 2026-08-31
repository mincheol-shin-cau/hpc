"""Merge the eight per-platform measurement CSV files.

For compatibility with the original experiment, the final observation in each
``Application`` group is removed from every source file before concatenation.
"""

import argparse
from pathlib import Path

import pandas as pd

SOURCE_FILES = [
    "cpu_cache_full_multi.csv",
    "cpu_cache_full_single.csv",
    "cpu_epyc_full_multi.csv",
    "cpu_epyc_full_single.csv",
    "cpu_flat_full_multi.csv",
    "cpu_flat_full_single.csv",
    "cpu_slx_full_multi.csv",
    "cpu_slx_full_single.csv",
]
REQUIRED_COLUMNS = {"Application", "mode", "cpu", "memory"}


def drop_last_per_application(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` without the final row of each Application group."""
    if "Application" not in df.columns:
        raise ValueError("Missing required column: Application")

    chunks = []
    for _, g in df.groupby("Application", sort=False):
        if len(g) <= 1:
            continue
        chunks.append(g.iloc[:-1])

    if not chunks:
        return df.iloc[0:0].copy()
    return pd.concat(chunks, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the eight downloaded CPU measurement CSVs after removing "
            "the final row of each Application group."
        )
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the eight downloaded source CSVs (default: data)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cpu_full_merged.csv"),
        help="Output path; relative paths are resolved under --data_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    out_arg = args.out.expanduser()
    out_path = out_arg if out_arg.is_absolute() else data_dir / out_arg
    source_paths = {(data_dir / name).resolve() for name in SOURCE_FILES}
    if out_path.resolve() in source_paths:
        raise SystemExit("--out must not overwrite one of the eight source CSVs")

    parts: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None

    for name in SOURCE_FILES:
        p = data_dir / name
        if not p.exists():
            raise SystemExit(f"Required source file not found: {p}")

        df = pd.read_csv(p, encoding="utf-8-sig")
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise SystemExit(f"{p.name}: missing required columns: {missing_text}")
        if df[list(REQUIRED_COLUMNS)].isna().any().any():
            raise SystemExit(f"{p.name}: required metadata/label columns contain NaN")

        if expected_columns is None:
            expected_columns = df.columns.tolist()
        elif df.columns.tolist() != expected_columns:
            raise SystemExit(f"{p.name}: column names/order differ from the first CSV")

        before = len(df)
        df = drop_last_per_application(df)
        after = len(df)
        parts.append(df)
        print(f"{name}: {before} -> {after} (dropped {before - after})")

    merged = pd.concat(parts, axis=0, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_path} (rows={len(merged)})")


if __name__ == "__main__":
    main()
