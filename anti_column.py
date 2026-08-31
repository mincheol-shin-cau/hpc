"""PCA-loading and correlation-based feature selection."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_METADATA_COLUMNS = [
    "Application",
    "mode",
    "cpu",
    "memory",
    "cpu_next",
    "memory_next",
    "cpu_score",
]


def adaptive_variable_selection(
    df: pd.DataFrame,
    target_n: int = 30,
    variance_threshold: float = 0.9,
    corr_threshold: float = 0.70,
    loading_ratio: float = 0.9,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """Select representative numeric variables using PCA loadings.

    Returns the initial loading-based list followed by the correlation-refined
    list. PCA is fitted after dropping rows with missing numeric values.
    """
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1]")
    if not 0 <= corr_threshold <= 1:
        raise ValueError("corr_threshold must be in [0, 1]")
    if not 0 < loading_ratio <= 1:
        raise ValueError("loading_ratio must be in (0, 1]")
    if target_n < 1:
        raise ValueError("target_n must be at least 1")

    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    num_df = num_df.loc[:, ~num_df.columns.str.startswith("Unnamed")]
    feature_names = num_df.columns.tolist()
    n_features = len(feature_names)
    if n_features == 0:
        raise ValueError("No numeric feature columns available for PCA")

    if verbose:
        print("=" * 60)
        print("[STEP 1] Data preprocessing (StandardScaler)")
        print(f"  Input features: {n_features}")

    before = len(num_df)
    num_df = num_df.dropna()
    dropped = before - len(num_df)
    if len(num_df) < 2:
        raise ValueError("PCA requires at least two complete rows")
    if verbose and dropped > 0:
        print(
            f"  Dropped {dropped} rows containing NaN ({before} -> {len(num_df)} rows)"
        )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_df.values)

    if verbose:
        print("\n[STEP 2] PCA and cumulative explained variance")

    n_components = min(len(num_df), n_features)
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    eigenvalues = pca.explained_variance_
    explained_ratios = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_ratios)
    loadings = pca.components_

    if verbose:
        print(f"  {'PC':<6} {'Eigenvalue':>12} {'Var Ratio':>12} {'Cum. Var':>12}")
        print("  " + "-" * 46)
        for i in range(min(n_components, 20)):
            print(
                f"  PC{i + 1:<4} {eigenvalues[i]:>12.4f} "
                f"{explained_ratios[i]:>12.4f} {cumulative_var[i]:>12.4f}"
            )
        if n_components > 20:
            print(f"  ... ({n_components} PCs total)")

    if verbose:
        print(f"\n[STEP 3] Number of PCs (variance threshold: {variance_threshold})")

    exceed_indices = np.where(cumulative_var >= variance_threshold)[0]

    if len(exceed_indices) == 0:
        A = n_components
    else:
        A = exceed_indices[0] + 1

    if A > target_n:
        if verbose:
            print(
                f"  Warning: minimum PCs for variance ({A}) exceeds target_n ({target_n})."
                f"\n    Clipping to target_n={target_n}, with at least 1 PC."
            )
        A = max(1, target_n)

    A = min(A, n_components)

    if A < 1:
        A = 1
        if verbose:
            print(
                f"  Warning: A=0 (variance_threshold={variance_threshold:.2f} too low).\n"
                f"    Forcing PC1 (cumulative variance={cumulative_var[0]:.4f})"
            )

    if verbose:
        print(
            f"  Selected number of PCs A = {A}  "
            f"(requested: {variance_threshold:.2f}, actual cum. var: {cumulative_var[A - 1]:.4f})"
        )

    if verbose:
        print(f"\n[STEP 4] Loading-based selection (loading_ratio={loading_ratio})")

    selected_set = []
    seen = set()

    for pc_idx in range(A):
        pc_loadings = np.abs(loadings[pc_idx])
        max_loading = pc_loadings.max()
        threshold = loading_ratio * max_loading

        selected_idx = np.where(pc_loadings >= threshold)[0]
        selected_vars = [feature_names[i] for i in selected_idx]

        if verbose:
            print(
                f"  PC{pc_idx + 1}: max_loading={max_loading:.4f}, "
                f"threshold={threshold:.4f} -> {len(selected_vars)} variables"
            )
            for v in selected_vars:
                print(f"    - {v}")

        for var in selected_vars:
            if var not in seen:
                seen.add(var)
                selected_set.append(var)

    initial_selected_list = selected_set

    if len(initial_selected_list) == 0:
        fallback_var = feature_names[int(np.argmax(np.abs(loadings[0])))]
        initial_selected_list = [fallback_var]
        if verbose:
            print(
                f"\n  Warning: no variables selected; using dominant PC1 variable: '{fallback_var}'"
            )

    if verbose:
        print(
            f"\n[STEP 5] Pearson correlation deduplication (threshold={corr_threshold})"
        )

    refined = list(initial_selected_list)

    if len(refined) > 1:
        corr_matrix = num_df[refined].corr(method="pearson").abs()

        to_remove = set()
        cols = corr_matrix.columns.tolist()

        for i in range(len(cols)):
            if cols[i] in to_remove:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove:
                    continue
                if corr_matrix.loc[cols[i], cols[j]] >= corr_threshold:
                    to_remove.add(cols[j])
                    if verbose:
                        print(
                            f"  Remove: '{cols[j]}'  "
                            f"<- corr('{cols[i]}', '{cols[j]}') = "
                            f"{corr_matrix.loc[cols[i], cols[j]]:.4f}"
                        )

        refined = [v for v in refined if v not in to_remove]

    final_refined_list = refined

    if verbose:
        print(f"\n  --- Final_Refined_List ({len(final_refined_list)}) ---")
        for v in final_refined_list:
            print(f"    - {v}")
        print()
        print("=" * 60)
        print("  [Summary]")
        print(f"  Initial_Selected_List : {len(initial_selected_list)}")
        print(f"  Final_Refined_List    : {len(final_refined_list)}")
        print("=" * 60)

    return initial_selected_list, final_refined_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive variable selection: PCA loadings + Pearson correlation filtering"
    )
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--target_n",
        type=int,
        default=30,
        help="Target number of variables (default: 30)",
    )
    parser.add_argument(
        "--variance",
        type=float,
        default=0.80,
        help="Cumulative variance threshold V (default: 0.80)",
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.90,
        help="Correlation deduplication threshold (default: 0.90)",
    )
    parser.add_argument(
        "--loading_ratio",
        type=float,
        default=0.9,
        help="Loading selection ratio vs max loading per PC (default: 0.9)",
    )
    parser.add_argument(
        "--feature_start_col",
        type=int,
        default=None,
        help=(
            "Optional first feature column index (0-based). By default, known "
            "metadata/target columns are removed by name."
        ),
    )
    args = parser.parse_args()

    print(f"\nInput file: {args.csv}")
    if not args.csv.exists():
        raise SystemExit(f"Input CSV not found: {args.csv}")

    df_raw = pd.read_csv(args.csv, encoding="utf-8-sig")
    print(f"Full shape: {df_raw.shape}")

    if args.feature_start_col is None:
        removed = [c for c in DEFAULT_METADATA_COLUMNS if c in df_raw.columns]
        df_features = df_raw.drop(columns=removed)
        print(f"Metadata columns removed: {removed}")
    else:
        if not 0 <= args.feature_start_col < len(df_raw.columns):
            raise SystemExit("--feature_start_col is outside the CSV column range")
        df_features = df_raw.iloc[:, args.feature_start_col :]
        print(f"Feature slice starts at column {args.feature_start_col}")
    print(f"Candidate feature columns: {df_features.shape[1]}\n")

    initial_list, refined_list = adaptive_variable_selection(
        df=df_features,
        target_n=args.target_n,
        variance_threshold=args.variance,
        corr_threshold=args.corr_threshold,
        loading_ratio=args.loading_ratio,
        verbose=True,
    )

    return initial_list, refined_list


if __name__ == "__main__":
    main()
