"""Exploratory data analysis for PhishLite.

Built for the Mendeley "Phishing Websites Dataset" (Vrbančič, 2020,
DOI 10.17632/72ptz43s9v.1), full variant: dataset_full.csv, 88,647 rows,
111 pre-extracted numeric features, target column 'phishing' (1 = phishing,
0 = legitimate). No raw URL text is retained in this dataset.

Usage:
    python eda.py --path data/dataset_full.csv --label phishing --out reports/eda

Prints a text summary and saves plots to the output directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd

from load import DEFAULT_LABEL, DEFAULT_PATH, load_dataset


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def overview(df: pd.DataFrame) -> None:
    section("OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print("\nDtypes:")
    print(df.dtypes.value_counts().to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values:", "none" if missing.empty else "")
    if not missing.empty:
        print(pd.DataFrame({"count": missing, "pct": (missing / len(df) * 100).round(2)}).to_string())

    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(f"\nConstant columns: {const_cols or 'none'}")


def redundant_features(df: pd.DataFrame, label: str, corr_threshold: float = 0.98) -> None:
    """Flag near-duplicate / highly collinear feature pairs.

    Useful for this dataset since 111 engineered features often include
    redundant counts (e.g. qty_dot_url vs qty_dot_domain vs qty_dot_directory).
    """
    section("REDUNDANT / HIGHLY CORRELATED FEATURES")
    num = df.select_dtypes("number").drop(columns=[label], errors="ignore")
    corr = num.corr().abs()
    upper = corr.where(pd.DataFrame(
        [[i < j for j in range(len(corr))] for i in range(len(corr))],
        index=corr.index, columns=corr.columns,
    ))
    pairs = (
        upper.stack()
        .loc[lambda s: s > corr_threshold]
        .sort_values(ascending=False)
    )
    if pairs.empty:
        print(f"No feature pairs with |corr| > {corr_threshold}.")
    else:
        print(f"{len(pairs)} feature pairs with |corr| > {corr_threshold}:")
        print(pairs.head(30).round(4).to_string())


def class_balance(df: pd.DataFrame, label: str, out: Path) -> None:
    section("CLASS BALANCE")
    counts = df[label].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    print(pd.DataFrame({"count": counts, "pct": pct}).to_string())

    fig, ax = plt.subplots(figsize=(4, 3.5))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#e45756"][: len(counts)])
    ax.set_title("Class balance")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)


def numeric_summary(df: pd.DataFrame, label: str, out: Path, max_hist: int = 12) -> list[str]:
    num = df.select_dtypes("number").drop(columns=[label], errors="ignore")
    section("NUMERIC FEATURES")
    if num.empty:
        print("No numeric features found.")
        return []

    print(num.describe().T.round(3).to_string())

    corr = num.corrwith(df[label]).dropna().sort_values(key=abs, ascending=False)
    print("\nTop correlations with label:")
    print(corr.head(15).round(3).to_string())

    # Histograms of the most label-correlated features, split by class
    top = list(corr.head(max_hist).index)
    if top:
        cols = 3
        rows = (len(top) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, col in zip(axes, top):
            for cls, grp in df.groupby(label):
                ax.hist(grp[col].dropna(), bins=30, alpha=0.55, label=str(cls))
            ax.set_title(col, fontsize=9)
        for ax in axes[len(top):]:
            ax.axis("off")
        axes[0].legend(title=label, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "feature_histograms.png", dpi=150)
        plt.close(fig)

    # Correlation heatmap of the top features
    heat_cols = list(corr.head(20).index)
    if len(heat_cols) > 1:
        cm = df[heat_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(heat_cols)), heat_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(heat_cols)), heat_cols, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Correlation (top label-correlated features)")
        fig.tight_layout()
        fig.savefig(out / "correlation_heatmap.png", dpi=150)
        plt.close(fig)

    return top


def categorical_summary(df: pd.DataFrame, label: str) -> None:
    cat = df.select_dtypes(exclude="number").columns.tolist()
    section("CATEGORICAL / TEXT FEATURES")
    if not cat:
        print("None.")
        return
    for col in cat:
        print(f"{col}: {df[col].nunique():,} unique | e.g. {df[col].dropna().iloc[0]!r}"[:150])


def main() -> None:
    parser = argparse.ArgumentParser(description="PhishLite EDA")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--out", default="reports/eda")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.path, args.label)
    if args.label not in df.columns:
        raise SystemExit(f"Label column '{args.label}' not found. Columns: {list(df.columns)}")

    overview(df)
    class_balance(df, args.label, out)
    numeric_summary(df, args.label, out)
    categorical_summary(df, args.label)
    redundant_features(df, args.label)

    print(f"\nPlots saved to {out.resolve()}")


if __name__ == "__main__":
    main()