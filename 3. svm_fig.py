"""Evaluate RBF-SVM classifiers with Leave-One-Application-Out validation."""

import argparse
import re
import time
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

simplefilter(action="ignore", category=FutureWarning)

META_DROP = [
    "Application",
    "mode",
    "cpu",
    "memory",
    "cpu_next",
    "memory_next",
    "cpu_score",
]

ARCH_ORDER = ["KNL_Cache", "KNL_Flat", "EPYC", "SLX"]

ARCH_SLUG = {
    "EPYC": "epyc",
    "SLX": "slx",
    "KNL_Cache": "knl_cache",
    "KNL_Flat": "knl_flat",
}


DATASET_NAME_PATTERN = re.compile(
    r"^cpu_(cache|flat|epyc|slx)_(full|pca(?:30|50|60|70|80))_"
    r"(multi|single)\.csv$",
    flags=re.IGNORECASE,
)


def parse_filename(filename: str) -> tuple[str, str, str] | None:
    """Parse a generated dataset filename, returning None if unrecognized."""
    match = DATASET_NAME_PATTERN.fullmatch(filename)
    if match is None:
        return None

    device, level, mode = (part.lower() for part in match.groups())
    if device == "cache":
        arch = "KNL_Cache"
    elif device == "flat":
        arch = "KNL_Flat"
    elif device == "epyc":
        arch = "EPYC"
    else:
        arch = "SLX"

    pca = "full" if level == "full" else level.removeprefix("pca")
    return arch, mode, pca


def resolve_arch_filter(arch_args: list[str] | None) -> set[str]:
    if not arch_args:
        return set(ARCH_ORDER)
    out: set[str] = set()
    for s in arch_args:
        if s == "epyc":
            out.add("EPYC")
        elif s == "slx":
            out.add("SLX")
        elif s == "cache":
            out.add("KNL_Cache")
        elif s == "flat":
            out.add("KNL_Flat")
        elif s == "knl":
            out.update(["KNL_Cache", "KNL_Flat"])
    return out


def ordered_archs(allowed: set[str]) -> list[str]:
    return [a for a in ARCH_ORDER if a in allowed]


def evaluate_single_dataset(c, g, X_values, y_values, groups, arch, mode, pca):
    """Evaluate one platform/mode/PCA dataset and return macro fold accuracy."""
    logo = LeaveOneGroupOut()
    fold_accs = []
    skipped_folds = 0

    if len(np.unique(y_values)) < 2:
        print(f"  [skip] {arch}_{mode}_{pca}: only one class in labels.")
        return {
            "Arch": arch,
            "Mode": mode,
            "PCA": pca,
            "Acc": np.nan,
            "N_samples": len(y_values),
            "N_groups": len(np.unique(groups)),
            "N_folds": 0,
            "Skipped_folds": 0,
        }

    for train_idx, test_idx in logo.split(X_values, y_values, groups=groups):
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]

        if len(np.unique(y_train)) < 2:
            skipped_folds += 1
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(kernel="rbf", C=c, gamma=g, cache_size=2000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        fold_accs.append(accuracy_score(y_test, y_pred))

    avg_acc = float(np.mean(fold_accs)) if fold_accs else np.nan
    return {
        "Arch": arch,
        "Mode": mode,
        "PCA": pca,
        "Acc": avg_acc,
        "N_samples": len(y_values),
        "N_groups": len(np.unique(groups)),
        "N_folds": len(fold_accs),
        "Skipped_folds": skipped_folds,
    }


def _safe_filename_part(x: float) -> str:
    s = f"{x:g}".replace(".", "p").replace("+", "").replace("-", "m")
    s = re.sub(r"[^0-9a-zA-Zp]", "_", s)
    return s


def _draw_single_arch_bars(ax, arch_df, pca_order, colors):
    multi_data, single_data = [], []
    for pca in pca_order:
        m_val = arch_df[(arch_df["PCA"] == pca) & (arch_df["Mode"] == "multi")][
            "Acc"
        ].values
        multi_data.append(m_val[0] if len(m_val) > 0 else np.nan)

        s_val = arch_df[(arch_df["PCA"] == pca) & (arch_df["Mode"] == "single")][
            "Acc"
        ].values
        single_data.append(s_val[0] if len(s_val) > 0 else np.nan)

    x = np.arange(len(pca_order))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2, multi_data, width, label="multi", color=colors["multi"]
    )
    rects2 = ax.bar(
        x + width / 2, single_data, width, label="single", color=colors["single"]
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pca_order)
    ax.set_xlabel("PCA level")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    autolabel(rects1)
    autolabel(rects2)


def plot_and_save_per_arch(
    df_results: pd.DataFrame,
    c: float,
    g: float,
    target: str,
    output_dir: Path,
    arch_list: list[str],
) -> list[str]:
    pca_order = ["full", "30", "50", "60", "70", "80"]
    colors = {"multi": "#4472C4", "single": "#ED7D31"}
    c_part = _safe_filename_part(float(c))
    g_part = _safe_filename_part(float(g))
    saved: list[str] = []

    for arch in arch_list:
        arch_df = df_results[df_results["Arch"] == arch]
        if arch_df.empty:
            print(f"  [skip plot] no rows for arch={arch}")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        slug = ARCH_SLUG.get(arch, arch.replace(" ", "_"))
        ax.set_title(
            f"{arch} | target={target} | C={c:g}, gamma={g:g}",
            fontsize=14,
            fontweight="bold",
        )
        _draw_single_arch_bars(ax, arch_df, pca_order, colors)
        fig.tight_layout()

        filename = output_dir / f"Accuracy_{target}_{slug}_C_{c_part}_G_{g_part}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(filename))

    return saved


def save_metrics_csv(
    df: pd.DataFrame,
    output_dir: Path,
    c: float,
    g: float,
    targets: list[str],
    allowed_archs: set[str],
) -> str:
    c_part = _safe_filename_part(float(c))
    g_part = _safe_filename_part(float(g))
    target_part = "-".join(sorted(set(targets)))
    arch_part = "-".join(ARCH_SLUG[a] for a in ordered_archs(allowed_archs))
    path = output_dir / (
        f"logo_metrics_{target_part}_{arch_part}_C_{c_part}_G_{g_part}.csv"
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def load_datasets_for_target(path_dir_pca: Path, target: str, allowed_archs: set[str]):
    file_list = sorted(path_dir_pca.glob("*.csv"))
    datasets = []

    for path in file_list:
        parsed = parse_filename(path.name)
        if parsed is None:
            print(f"  [skip] Unrecognized dataset filename: {path.name}")
            continue

        arch, mode, pca = parsed
        if arch not in allowed_archs:
            continue

        df = pd.read_csv(path, encoding="utf-8-sig")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        required = {"Application", target}
        missing = required.difference(df.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            print(f"  [skip] {path.name}: missing columns: {missing_text}")
            continue

        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped:
            print(f"  [info] {path.name}: dropped {dropped} row(s) with NaN")
        if df.empty:
            print(f"  [skip] {path.name}: no complete rows")
            continue

        feature_frame = df.drop(columns=META_DROP, errors="ignore")
        if feature_frame.empty:
            print(f"  [skip] {path.name}: no feature columns")
            continue
        non_numeric = feature_frame.select_dtypes(exclude="number").columns.tolist()
        if non_numeric:
            print(
                f"  [skip] {path.name}: non-numeric features: " + ", ".join(non_numeric)
            )
            continue
        if df["Application"].nunique() < 2:
            print(f"  [skip] {path.name}: LOGO requires at least two groups")
            continue

        groups = df["Application"].values
        X = feature_frame.values
        y = df[target].values

        datasets.append(
            {
                "X": X,
                "y": y,
                "groups": groups,
                "Arch": arch,
                "Mode": mode,
                "PCA": pca,
            }
        )

    return datasets


def run_one_target(
    c: float,
    g: float,
    target: str,
    path_dir_pca: Path,
    output_dir: Path,
    n_jobs: int,
    allowed_archs: set[str],
):
    print(f"\n--- target={target} ---")
    arch_list = ordered_archs(allowed_archs)
    datasets = load_datasets_for_target(path_dir_pca, target, allowed_archs)
    if not datasets:
        print(f"  No datasets loaded for target '{target}' (check --arch filter).")
        return pd.DataFrame(), []

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(evaluate_single_dataset)(
            c, g, ds["X"], ds["y"], ds["groups"], ds["Arch"], ds["Mode"], ds["PCA"]
        )
        for ds in datasets
    )
    df_results = pd.DataFrame(results)
    paths = plot_and_save_per_arch(df_results, c, g, target, output_dir, arch_list)
    print(f"  Done in {time.time() - t0:.2f}s -> {len(paths)} figure(s)")
    for p in paths:
        print(f"    {p}")
    return df_results, paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVM LOGO accuracy: one PNG per architecture, optional --arch filter."
    )
    parser.add_argument(
        "--c",
        type=float,
        required=True,
        help="SVM C (e.g. 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        required=True,
        help="RBF gamma (e.g. 0.01)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        choices=["cpu", "memory"],
        help="Label columns (default: cpu memory). Ignored if --target is used.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        dest="target_append",
        choices=["cpu", "memory"],
        metavar="LABEL",
        help="Single label (cpu or memory). Overrides --targets when set.",
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        default=None,
        metavar="NAME",
        choices=["epyc", "slx", "cache", "flat", "knl"],
        help=(
            "Architectures to include: epyc, slx, cache (KNL cache), "
            "flat (KNL flat), knl (both KNL). Default: all four platforms."
        ),
    )
    parser.add_argument(
        "--pca_dir",
        type=Path,
        default=Path("pca_file"),
        help="Directory containing PCA CSV files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("fig"),
        help="Directory for PNG figures and logo_metrics_*.csv",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="joblib Parallel n_jobs (-1 = all cores)",
    )
    args = parser.parse_args()

    if args.c <= 0:
        parser.error("--c must be greater than 0")
    if args.gamma <= 0:
        parser.error("--gamma must be greater than 0")
    if args.n_jobs == 0:
        parser.error("--n_jobs cannot be 0")

    if args.target_append:
        args.targets = list(dict.fromkeys(args.target_append))
    elif args.targets is None:
        args.targets = ["cpu", "memory"]
    else:
        args.targets = list(dict.fromkeys(args.targets))

    allowed = resolve_arch_filter(args.arch)
    if not allowed:
        raise SystemExit("No architectures after --arch; check arguments.")

    path_dir_pca = args.pca_dir.expanduser()
    output_dir = args.out_dir.expanduser()

    if not path_dir_pca.is_dir():
        raise SystemExit(f"PCA directory not found: {path_dir_pca}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"C={args.c:g}, gamma={args.gamma:g}, targets={args.targets}, "
        f"arch={sorted(allowed)}, pca_dir={path_dir_pca}, out_dir={output_dir}"
    )

    saved: list[str] = []
    metric_frames: list[pd.DataFrame] = []
    for target in args.targets:
        df_res, paths = run_one_target(
            args.c,
            args.gamma,
            target,
            path_dir_pca,
            output_dir,
            args.n_jobs,
            allowed,
        )
        saved.extend(paths)
        if not df_res.empty:
            chunk = df_res.copy()
            chunk.insert(0, "target", target)
            chunk["C"] = args.c
            chunk["gamma"] = args.gamma
            metric_frames.append(chunk)

    if metric_frames:
        metrics_df = pd.concat(metric_frames, ignore_index=True)
        arch_rank = {arch: i for i, arch in enumerate(ARCH_ORDER)}
        pca_rank = {
            level: i for i, level in enumerate(["full", "30", "50", "60", "70", "80"])
        }
        metrics_df = metrics_df.sort_values(
            by=["target", "Arch", "Mode", "PCA"],
            key=lambda series: (
                series.map(arch_rank)
                if series.name == "Arch"
                else series.map(pca_rank)
                if series.name == "PCA"
                else series
            ),
        ).reset_index(drop=True)
        csv_path = save_metrics_csv(
            metrics_df,
            output_dir,
            args.c,
            args.gamma,
            args.targets,
            allowed,
        )
        print(f"\n[CSV] {csv_path} ({len(metrics_df)} rows)")

    if saved:
        print(f"Saved {len(saved)} figure(s) under '{output_dir}'.")
    else:
        print("\nNo figures saved.")


if __name__ == "__main__":
    main()
