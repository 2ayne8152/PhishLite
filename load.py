# pip install numpy pandas scikit-learn matplotlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FULL_FILE = "./data/dataset_full.csv"
RANDOM_SEED = 42
DEFAULT_LABEL = "phishing"

# String labels -> binary (1 = phishing, 0 = legitimate).
_LABEL_MAP = {
    "phishing": 1, "phish": 1, "malicious": 1, "bad": 1,
    "spam": 1, "true": 1, "1": 1,
    "legitimate": 0, "legit": 0, "benign": 0, "good": 0,
    "ham": 0, "safe": 0, "false": 0, "0": 0,
}

def _read_file(path: Path) -> pd.DataFrame:
    """Read a supported tabular file."""
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {suffix}")

def normalize_labels(series: pd.Series) -> pd.Series:
    """Map string labels to 0/1; pass numeric labels through unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        # The target is a binary classification label; this check does not apply to features.
        numeric = pd.to_numeric(series, errors="raise")
        values = set(numeric.dropna().unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(
                "The target column must contain only 0 and 1 after missing target rows "
                f"are removed. Found target values: {sorted(values, key=str)}"
            )
        return numeric.astype("int64")

    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(_LABEL_MAP)
    unknown = sorted(cleaned[mapped.isna()].unique())
    if unknown:
        raise ValueError(f"Unrecognised label values: {unknown}. Update _LABEL_MAP in load.py.")
    return mapped.astype(int)


def load_dataset(
    path: str | Path = FULL_FILE,
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

        if label_col not in df.columns:
            raise KeyError(f"Label column '{label_col}' not in columns: {list(df.columns)}")

        df = df.dropna(subset=[label_col]).reset_index(drop=True)
        df[label_col] = normalize_labels(df[label_col])

    return df


def split_xy(df: pd.DataFrame, label_col: str = DEFAULT_LABEL):
    """Split a dataframe into features (X) and target (y)."""
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not in columns: {list(df.columns)}")
    return df.drop(columns=[label_col]), df[label_col]

def train_val_test(
        df: pd.DataFrame, label_col: str = DEFAULT_LABEL,
        val_size: float = 0.2, test_size: float = 0.2, seed: int = 42):
    """Stratified train/val/test split. Sizes are fractions of the WHOLE dataset."""

    if not (0 < val_size < 1 and 0 < test_size < 1):
        raise ValueError("val_size and test_size must both be between 0 and 1.")

    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")

    X, y = split_xy(df, label_col)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_size/(1-test_size), stratify=y_tmp ,random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    df = load_dataset()  # full, deduplicated
    # 60/20/20
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df, val_size=0.2, test_size=0.2)

    print(f"Dataset rows: {len(df):,}")
    print(
        f"Train: {len(X_train):,} | Validation: {len(X_val):,} "
        f"| Test: {len(X_test):,}"
    )
    print("Class distribution:")
    print(df[DEFAULT_LABEL].value_counts(normalize=True).sort_index())