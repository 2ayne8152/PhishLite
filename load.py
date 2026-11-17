"""Data loading utilities for PhishLite.

Usage:
    from load import load_dataset, split_xy
    df = load_dataset("data/dataset.csv", label_col="label")
    X, y = split_xy(df)

CLI quick check:
    python load.py --path data/dataset.csv --label label
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/dataset_full.csv")
DEFAULT_LABEL = "phishing"  # Mendeley 72ptz43s9v: 0 = legitimate, 1 = phishing

# String labels -> binary (1 = phishing, 0 = legitimate).
# Numeric labels are left as-is; check your dataset's convention
# (some use -1 for phishing and 1 for legitimate).
_LABEL_MAP = {
    "phishing": 1, "phish": 1, "malicious": 1, "bad": 1, "spam": 1, "true": 1, "1": 1,
    "legitimate": 0, "legit": 0, "benign": 0, "good": 0, "ham": 0, "safe": 0,
    "false": 0, "0": 0,
}


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_labels(series: pd.Series) -> pd.Series:
    """Map string labels to 0/1; pass numeric labels through unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(_LABEL_MAP)
    unknown = sorted(cleaned[mapped.isna()].unique())
    if unknown:
        raise ValueError(f"Unrecognised label values: {unknown}. Update _LABEL_MAP in load.py.")
    return mapped.astype(int)


def load_dataset(
    path: str | Path = DEFAULT_PATH,
    label_col: str = DEFAULT_LABEL,
    clean: bool = True,
) -> pd.DataFrame:
    """Load a dataset from disk and apply light, non-destructive cleaning."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = _read_file(path)

    if clean:
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").drop_duplicates().reset_index(drop=True)
        if label_col in df.columns:
            df = df.dropna(subset=[label_col]).reset_index(drop=True)
            df[label_col] = normalize_labels(df[label_col])

    return df


def split_xy(df: pd.DataFrame, label_col: str = DEFAULT_LABEL):
    """Split a dataframe into features (X) and target (y)."""
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not in columns: {list(df.columns)}")
    return df.drop(columns=[label_col]), df[label_col]


def train_test(df: pd.DataFrame, label_col: str = DEFAULT_LABEL, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split."""
    from sklearn.model_selection import train_test_split

    X, y = split_xy(df, label_col)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick dataset load check")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--label", default=DEFAULT_LABEL)
    args = parser.parse_args()

    data = load_dataset(args.path, args.label)
    print(f"Loaded {data.shape[0]:,} rows x {data.shape[1]} columns")
    print(data.head())