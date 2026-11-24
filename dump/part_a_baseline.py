"""
PhishLite - Part A: baseline, evaluation and analysis (single-file version).

Dataset: Vrbancic et al. (2020) phishing URL dataset, dataset_full.csv (111 features + `phishing`).

Usage (run from the project folder that contains ./data/dataset_full.csv):

    python part_a_baseline.py              # all stages, in order
    python part_a_baseline.py eda          # descriptive analysis of the dataset
    python part_a_baseline.py baseline     # 5-fold CV of the 8 models + error analysis
    python part_a_baseline.py analysis     # feature groups, importance, refinement, shortlist
    python part_a_baseline.py curves       # full-size learning curves + iteration curves
    python part_a_baseline.py all --data path/to/dataset_full.csv

Design rules
- Data: exact-duplicate rows are removed, then a stratified 60/20/20 train/validation/test split.
- The baseline is deliberately raw: library defaults, no scaling, no feature selection, no
  class balancing. Only documented execution-related settings differ (max_iter, epochs, SVM subsample).
- Baseline cross-validation uses the TRAINING split only; analysis and curves score on the
  VALIDATION split; the held-out TEST split is never used in Part A.

Outputs: reports/eda, reports/baseline, reports/analysis, reports/curves (figures as PNG, tables as CSV).

Sections: 1 data loading | 2 EDA | 3 models + cross-validation | 4 evaluation (learning curves,
error analysis) | 5 analysis (ablation, importance, refinement, shortlist) | 6 curves | 7 command line
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
import traceback
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


###############################################################################
# 1. DATA LOADING (dedupe, label check, stratified train/val/test split)
###############################################################################

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
    path: str | Path | None = None,
    label_col: str = DEFAULT_LABEL,
    clean: bool = True,
) -> pd.DataFrame:
    """Load a dataset from disk and apply light, non-destructive cleaning."""
    path = Path(FULL_FILE if path is None else path)
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


###############################################################################
# 2. EDA (descriptive; run on the whole deduplicated dataset)
###############################################################################

EDA_DIR = Path("./reports/eda")

# Features that need a network lookup or an external service (paper Table 6).
EXTERNAL_FEATURES = {
    "time_response", "domain_spf", "asn_ip", "time_domain_activation",
    "time_domain_expiration", "qty_ip_resolved", "qty_nameservers",
    "qty_mx_servers", "ttl_hostname", "tls_ssl_certificate", "qty_redirects",
    "url_google_index", "domain_google_index", "url_shortened",
}
GROUP_ORDER = ["url", "domain", "directory", "file", "params", "external", "other"]
GROUP_COLORS = {
    "url": "#4c78a8", "domain": "#f58518", "directory": "#54a24b",
    "file": "#b279a2", "params": "#9d755d", "external": "#e45756", "other": "#999999",
}


def feature_group(name: str) -> str:
    """Map a column name to its paper table (URL part it was computed from)."""
    if name in EXTERNAL_FEATURES:
        return "external"
    # length / flag features whose names do not end with the part name
    if name in {"domain_length", "domain_in_ip"}:
        return "domain"
    for part in ("directory", "file", "params", "domain", "url"):
        if name.endswith(f"_{part}") or name.startswith(f"{part}_"):
            return part
    return "other"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def numeric_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    return df.select_dtypes("number").drop(columns=[label], errors="ignore")


# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
def overview(df: pd.DataFrame, label: str) -> None:
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

    # How many features belong to each paper group
    groups = pd.Series({c: feature_group(c) for c in df.columns if c != label})
    print("\nFeatures per group:")
    print(groups.value_counts().reindex(GROUP_ORDER).dropna().astype(int).to_string())


# --------------------------------------------------------------------------- #
# 2. Class balance
# --------------------------------------------------------------------------- #
def class_balance(df: pd.DataFrame, label: str, out: Path) -> None:
    section("CLASS BALANCE")
    counts = df[label].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    print(pd.DataFrame({"count": counts, "pct": pct}).to_string())

    fig, ax = plt.subplots(figsize=(4, 3.5))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#e45756"][: len(counts)])
    ax.set_title("Class balance (0 = legitimate, 1 = phishing)")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Sentinel values (-1): "component absent" or "lookup failed"
# --------------------------------------------------------------------------- #
def sentinel_check(df: pd.DataFrame, label: str, out: Path, top_n: int = 12) -> None:
    section("SENTINEL VALUES (-1)")
    X = numeric_features(df, label)
    is_neg = X.eq(-1)
    n_neg = is_neg.sum()
    cols = n_neg[n_neg > 0].sort_values(ascending=False).index.tolist()
    if not cols:
        print("No -1 values found.")
        return

    base_rate = df[label].mean()
    table = pd.DataFrame({
        "group": [feature_group(c) for c in cols],
        "count_-1": n_neg[cols].values,
        "pct_-1": (n_neg[cols] / len(df) * 100).round(2).values,
        # share of phishing among the rows that have -1 (compare with base rate)
        "phishing_rate_when_-1": [df.loc[is_neg[c], label].mean() for c in cols],
    }, index=cols)
    print(f"{len(cols)} columns contain -1. Overall phishing rate = {base_rate:.3f}")
    print(table.head(25).round(3).to_string())
    table.to_csv(out / "sentinel_summary.csv")

    # Do the *_params columns turn to -1 together? (supports the "component absent" reading)
    params_cols = [c for c in cols if feature_group(c) == "params"]
    if params_cols:
        all_neg = is_neg[params_cols].all(axis=1).mean()
        any_neg = is_neg[params_cols].any(axis=1).mean()
        print(f"\nParams columns: all -1 together in {all_neg:.1%} of rows, "
              f"any -1 in {any_neg:.1%} of rows.")

    # Stacked bars: share of -1 / 0 / positive per class for the most affected columns
    plot_cols = cols[:top_n]
    classes = sorted(df[label].unique())
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(plot_cols)), 4.5))
    width = 0.38
    seg_colors = {"-1": "#e45756", "0": "#cccccc", ">0": "#4c78a8"}
    for i, col in enumerate(plot_cols):
        for j, cls in enumerate(classes):
            v = df.loc[df[label] == cls, col]
            parts = {"-1": (v == -1).mean(), "0": (v == 0).mean(), ">0": (v > 0).mean()}
            x = i + (j - 0.5) * width
            bottom = 0.0
            for key, val in parts.items():
                ax.bar(x, val, width * 0.95, bottom=bottom, color=seg_colors[key],
                       label=key if (i == 0 and j == 0) else None)
                bottom += val
            ax.text(x, 1.01, str(cls), ha="center", fontsize=7)
    ax.set_xticks(range(len(plot_cols)), plot_cols, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Share of rows")
    ax.set_title("-1 / 0 / positive share by class (left bar = 0, right bar = 1)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "sentinel_by_class.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. Single-feature separation (ROC-AUC) + per-group summary + leakage flag
# --------------------------------------------------------------------------- #
def single_feature_auc(df: pd.DataFrame, label: str, out: Path,
                       leak_threshold: float = 0.95, top_n: int = 25) -> pd.Series:
    """Oriented AUC per feature: 0.5 = no signal, 1.0 = perfect single-feature separator."""
    section("SINGLE-FEATURE SEPARATION (ROC-AUC)")
    X = numeric_features(df, label)
    y = df[label]

    raw = pd.Series({c: roc_auc_score(y, X[c]) for c in X.columns})
    auc = np.maximum(raw, 1 - raw).sort_values(ascending=False)   # direction does not matter
    auc.rename("auc").to_csv(out / "single_feature_auc.csv")

    print(f"Top {min(15, len(auc))} features by single-feature AUC:")
    print(auc.head(15).round(4).to_string())

    suspicious = auc[auc > leak_threshold]
    if suspicious.empty:
        print(f"\nNo single feature above AUC {leak_threshold}.")
    else:
        print(f"\n[!] {len(suspicious)} feature(s) above AUC {leak_threshold} "
              f"(check for leakage / a source shortcut): {list(suspicious.index)}")

    # Per-group summary: how much signal does each feature group carry on its own?
    grp = pd.DataFrame({"auc": auc, "group": [feature_group(c) for c in auc.index]})
    summary = grp.groupby("group")["auc"].agg(n_features="count", mean_auc="mean", max_auc="max")
    summary["best_feature"] = grp.loc[grp.groupby("group")["auc"].idxmax()].reset_index().set_index("group")["index"]
    summary = summary.reindex([g for g in GROUP_ORDER if g in summary.index])
    print("\nSignal per feature group:")
    print(summary.round(4).to_string())
    summary.to_csv(out / "group_auc_summary.csv")

    # Bar chart of the top features, coloured by group
    top = auc.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(top) + 1.5))
    ax.barh(top.index, top.values, color=[GROUP_COLORS[feature_group(c)] for c in top.index])
    ax.axvline(leak_threshold, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Single-feature ROC-AUC (0.5 = no signal)")
    ax.set_title(f"Top {len(top)} features by separation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_ORDER if g in set(grp["group"])]
    ax.legend(handles, [g for g in GROUP_ORDER if g in set(grp["group"])], fontsize=7, loc="lower right")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "single_feature_auc.png", dpi=150)
    plt.close(fig)

    return auc


# --------------------------------------------------------------------------- #
# 5. Redundancy: clustered correlation heatmap + redundancy clusters
# --------------------------------------------------------------------------- #
def redundancy_analysis(df: pd.DataFrame, label: str, out: Path, cluster_corr: float = 0.95) -> None:
    section("REDUNDANT / HIGHLY CORRELATED FEATURES")
    X = numeric_features(df, label)
    X = X.loc[:, X.nunique() > 1]               # constant columns give NaN correlations
    if X.shape[1] < 2:
        print("Not enough varying numeric features.")
        return

    corr = X.corr(method="spearman").abs().fillna(0.0)   # rank-based: robust to heavy tails
    values = corr.to_numpy()

    iu = np.triu_indices_from(values, k=1)
    pair_vals = pd.Series(values[iu], index=pd.MultiIndex.from_arrays(
        [corr.index[iu[0]], corr.columns[iu[1]]]))
    high = pair_vals[pair_vals > cluster_corr].sort_values(ascending=False)
    print(f"{len(high)} feature pairs with |Spearman| > {cluster_corr}")
    print(high.head(20).round(4).to_string())

    # Hierarchical clustering on distance = 1 - |corr|
    dist = 1 - values
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(link)

    # Redundancy clusters: features within (1 - cluster_corr) of each other
    labels = fcluster(link, t=1 - cluster_corr, criterion="distance")
    clusters = pd.Series(corr.index, index=labels).groupby(level=0).apply(list)
    multi = [c for c in clusters if len(c) > 1]
    print(f"\n{len(multi)} redundancy clusters (|r| > {cluster_corr}); "
          f"keeping one feature per cluster would drop {sum(len(c) - 1 for c in multi)} columns.")
    for c in sorted(multi, key=len, reverse=True)[:12]:
        print(f"  [{len(c)}] {', '.join(c[:8])}{' ...' if len(c) > 8 else ''}")
    pd.DataFrame({"cluster": [i for i, c in enumerate(multi) for _ in c],
                  "feature": [f for c in multi for f in c]}).to_csv(out / "redundancy_clusters.csv", index=False)

    # Clustered heatmap (rows/columns ordered so correlated features sit together)
    ordered = corr.index[order]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.loc[ordered, ordered].to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=4)
    ax.set_yticks(range(len(ordered)), ordered, fontsize=4)
    fig.colorbar(im, ax=ax, fraction=0.03, label="|Spearman correlation|")
    ax.set_title("Clustered feature correlation (bright blocks = redundant groups)")
    fig.tight_layout()
    fig.savefig(out / "correlation_clustered.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Categorical / text columns (expected: none for this dataset)
# --------------------------------------------------------------------------- #
def categorical_summary(df: pd.DataFrame) -> None:
    cat = df.select_dtypes(exclude="number").columns.tolist()
    section("CATEGORICAL / TEXT FEATURES")
    if not cat:
        print("None.")
        return
    for col in cat:
        sample = df[col].dropna()
        example = repr(sample.iloc[0] if not sample.empty else "No valid values")
        print(f"{col}: {df[col].nunique():,} unique | e.g. {example}"[:150])


def run_eda() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    label = DEFAULT_LABEL

    overview(df, label)
    class_balance(df, label, EDA_DIR)
    sentinel_check(df, label, EDA_DIR)
    auc = single_feature_auc(df, label, EDA_DIR)
    redundancy_analysis(df, label, EDA_DIR)
    categorical_summary(df)
    print(f"\nEDA outputs saved to {EDA_DIR.resolve()}")


###############################################################################
# 3. MODELS + STRATIFIED K-FOLD CROSS-VALIDATION (baseline)
###############################################################################

BASELINE_DIR = Path('./reports/baseline')

CV_FOLDS = 5
# RBF SVC scales roughly quadratically with rows and CV multiplies the cost, so each
# fold trains the SVM on a stratified subsample. Set to None to use every row.
# Disclose this in the report if you keep it.
SVM_MAX_TRAIN_ROWS: int | None = 20000
CPU_JOBS = max(1, (os.cpu_count() or 2) - 2)
RUN_LEARNING_CURVES = True

warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False


ALL_MODEL_NAMES = [
    "Logistic Regression", "Naive Bayes",   # Linear & Prob
    "k-NN", "SVM",                          # Distance & Geometrical
    "Random Forest", "LightGBM",            # Tree-based model: RF (bagging), LightGBM (boosting)
    "TabNet", "MLP"                         # Neural network
]

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "MCC"]


# --------------------------------------------------------------------------- #
# Progress helpers (visual feedback only -- zero impact on model behavior)
# --------------------------------------------------------------------------- #
@contextmanager
def live_spinner(label: str):
    """Display an animated spinner with a live elapsed-time counter during blocking C/Cython calls."""
    stop_event = threading.Event()
    start = time.time()

    def _spin():
        for frame in itertools.cycle(["|", "/", "-", "\\"]):
            if stop_event.is_set():
                break
            elapsed = time.time() - start
            sys.stdout.write(f"\r  {frame} {label} ... {elapsed:6.1f}s elapsed")
            sys.stdout.flush()
            time.sleep(0.2)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()


def lgbm_tqdm_callback(total_trees: int):
    """Return a LightGBM callback that updates a tqdm bar per boosting iteration."""
    if not HAS_TQDM:
        return None
    pbar = tqdm(total=total_trees, desc="  LightGBM trees", unit="tree", leave=False)

    def _callback(env):
        pbar.update(1)
        if env.iteration + 1 >= env.end_iteration:
            pbar.close()

    _callback.order = 10
    return _callback


# --------------------------------------------------------------------------- #
# Model registry -- library defaults only, plus the minimum needed to run
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    """Build available models using default or minimally adjusted settings."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "k-NN": KNeighborsClassifier(n_jobs=max(2, CPU_JOBS)),
        "SVM": SVC(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=max(2, CPU_JOBS)),
        "MLP": MLPClassifier(random_state=RANDOM_SEED, max_iter=300),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1)
    else:
        print("[skip] LightGBM not installed (`pip install lightgbm`) -- skipping.")
    if HAS_TABNET:
        models["TabNet"] = None     # built fresh inside fit_predict (not sklearn-clonable)
    else:
        print("[skip] pytorch-tabnet not installed (`pip install pytorch-tabnet torch`) -- skipping.")
    return models


def fit_predict(name: str, model, X_train, y_train, X_eval, svm_max_train: int | None):
    """Fit one model on (X_train, y_train) and predict X_eval.

    Used once per CV fold: X_train is the fold's training part and X_eval is its
    held-out part. Returns (y_pred, y_score); y_score feeds ROC-AUC.
    """
    if name == 'SVM' and svm_max_train is not None:
        if svm_max_train < 1:
            raise ValueError("svm_max_train must be a positive integer.")

        if len(X_train) > svm_max_train:
            X_fit, _, y_fit, _ = train_test_split(
                X_train, y_train,
                train_size=svm_max_train,
                stratify=y_train,
                random_state=RANDOM_SEED,
            )
        else:
            X_fit, y_fit = X_train, y_train
    else:
        X_fit, y_fit = X_train, y_train

    if name == 'TabNet':
        clf = TabNetClassifier(seed=RANDOM_SEED, verbose=0)

        X_fit_array = X_fit.to_numpy(dtype=np.float32)
        X_eval_array = X_eval.to_numpy(dtype=np.float32)
        y_fit_array = y_fit.to_numpy(dtype=np.int64)

        clf.fit(X_fit_array,
                y_fit_array,
                max_epochs=50,
                patience=50,
                batch_size=1024,
                virtual_batch_size=128
        )
        y_pred = clf.predict(X_eval_array)
        y_score = clf.predict_proba(X_eval_array)[:, 1]
        return y_pred, y_score

    clf = model

    if name == "LightGBM" and HAS_TQDM:
        cb = lgbm_tqdm_callback(clf.n_estimators)
        clf.fit(X_fit, y_fit, callbacks=[cb])
    else:
        clf.fit(X_fit, y_fit)

    y_pred = clf.predict(X_eval)

    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X_eval)
        positive_index = list(clf.classes_).index(1)
        y_score = probabilities[:, positive_index]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_eval)
    else:
        y_score = None

    return y_pred, y_score


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    try:
        roc_auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    except ValueError:
        roc_auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "FPR": fpr,
        "ROC_AUC": roc_auc,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate_model(name: str, model, X: pd.DataFrame, y: pd.Series, n_splits: int = CV_FOLDS):
    """Stratified k-fold CV for one model.

    Returns:
        fold_df   one row per fold with the metrics and the fold time
        oof_pred  out-of-fold class prediction for every row of X
        oof_score out-of-fold score (probability or decision value) for every row of X
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = pd.Series(np.nan, index=X.index, dtype=float)
    oof_score = pd.Series(np.nan, index=X.index, dtype=float)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        start = time.time()
        # fresh, unfitted copy per fold so nothing leaks between folds
        fold_model = clone(model) if model is not None else None
        # LightGBM draws its own tqdm bar; a spinner on top would garble the output
        spinner = nullcontext() if (name == "LightGBM" and HAS_TQDM) else live_spinner(f"{name} fold {fold}/{n_splits}")
        with spinner:
            y_pred, y_score = fit_predict(name, fold_model, X_tr, y_tr, X_te, SVM_MAX_TRAIN_ROWS)
        elapsed = time.time() - start

        metrics = compute_metrics(y_te, y_pred, y_score)
        metrics.update({"Fold": fold, "Time_Sec": round(elapsed, 2)})
        rows.append(metrics)
        oof_pred.iloc[test_idx] = y_pred
        if y_score is not None:
            oof_score.iloc[test_idx] = y_score
        print(f"  fold {fold}/{n_splits} | {elapsed:6.1f}s | "
              f"F1={metrics['F1']:.4f} FPR={metrics['FPR']:.4f} ROC-AUC={metrics['ROC_AUC']:.4f}")

    return pd.DataFrame(rows), oof_pred.astype(int), oof_score


def summarise_cv(fold_results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse per-fold metrics into mean/std tables (numeric + report-friendly)."""
    numeric_rows = []
    for name, fold_df in fold_results.items():
        row = {"Model": name, "Mean_Fold_Time_Sec": fold_df["Time_Sec"].mean()}
        for col in METRIC_COLS:
            row[f"{col}_mean"] = fold_df[col].mean()
            row[f"{col}_std"] = fold_df[col].std(ddof=1)
        numeric_rows.append(row)

    numeric = pd.DataFrame(numeric_rows).sort_values("F1_mean", ascending=False).reset_index(drop=True)

    pretty = numeric[["Model"]].copy()
    for col in METRIC_COLS:
        pretty[col] = numeric[f"{col}_mean"].map("{:.4f}".format) + " ± " + numeric[f"{col}_std"].map("{:.4f}".format)
    pretty["Mean_Fold_Time_Sec"] = numeric["Mean_Fold_Time_Sec"].round(2)
    return numeric, pretty


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_baseline() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df, val_size=0.2, test_size=0.2, seed=RANDOM_SEED)

    non_numeric = X_train.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(
            "Baseline models require numeric features. "
            f"Non-numeric columns found: {non_numeric[:20]}. "
            "Handle encoding in the separately documented preprocessing stage."
        )

    print(
        f"Train: {len(X_train):,} | Validation: {len(X_val):,} "
        f"| Held-out test: {len(X_test):,} | Features: {X_train.shape[1]}"
    )
    print(f"{CV_FOLDS}-fold stratified CV on the training split only "
          f"(validation and test splits are not used in this stage).")

    models = build_models()
    runnable_names = [name for name in ALL_MODEL_NAMES if name in models]

    fold_results: dict[str, pd.DataFrame] = {}
    oof: dict[str, tuple[pd.Series, pd.Series]] = {}

    for idx, name in enumerate(runnable_names, start=1):
        print(f"\n[{idx}/{len(runnable_names)}] {name}")
        try:
            fold_df, oof_pred, oof_score = cross_validate_model(name, models[name], X_train, y_train)
            fold_results[name] = fold_df
            oof[name] = (oof_pred, oof_score)
            print(f"  mean F1 = {fold_df['F1'].mean():.4f} ± {fold_df['F1'].std(ddof=1):.4f}")
        except Exception as exc:
            print(f"  [error] {name} failed: {exc}")

    if not fold_results:
        print("No models produced results.")
        return

    # Save everything before plotting, so a plotting error cannot lose the CV run
    numeric, pretty = summarise_cv(fold_results)
    numeric.to_csv(BASELINE_DIR / "cv_results_numeric.csv", index=False)
    pretty.to_csv(BASELINE_DIR / "cv_results.csv", index=False)
    pd.concat([d.assign(Model=n) for n, d in fold_results.items()]).to_csv(
        BASELINE_DIR / "cv_fold_metrics.csv", index=False)
    pd.DataFrame({**{f"{n}_pred": p for n, (p, _) in oof.items()},
                  **{f"{n}_score": s for n, (_, s) in oof.items()},
                  "true": y_train}).to_csv(BASELINE_DIR / "oof_predictions.csv")

    print(f"\n{'=' * 70}\nBASELINE CV RESULTS (saved to {BASELINE_DIR.resolve()})\n{'=' * 70}")
    print(pretty.to_string(index=False))

    # Error analysis and learning curves (see evaluate.py)
    try:
        preds = {n: (p.to_numpy(), s.to_numpy() if s.notna().all() else None) for n, (p, s) in oof.items()}
        run_evaluation(models, X_train, y_train, preds, BASELINE_DIR,
                                do_curves=RUN_LEARNING_CURVES)
    except Exception as exc:
        print(f"\n[error] evaluation stage failed: {type(exc).__name__}: {exc}")
        print("CV results above were already saved; re-run evaluate on oof_predictions.csv if needed.")


###############################################################################
# 4. EVALUATION (learning curves, error analysis on out-of-fold predictions)
###############################################################################

# Models that scale badly with rows get a stratified subsample for learning curves.
SLOW_MODELS = {"SVM", "k-NN"}


def _subsample(X, y, max_rows: int | None, seed: int = RANDOM_SEED):
    """Stratified subsample so slow models stay tractable (disclose this in the report)."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=seed)
    return X_s, y_s


# --------------------------------------------------------------------------- #
# 1. Learning curves: error vs training-set size (sklearn-compatible models)
# --------------------------------------------------------------------------- #
def learning_curves(models: dict, X_train, y_train, out_dir: Path,
                    n_splits: int = 3, slow_max_rows: int | None = 15000) -> None:
    """Error = 1 - F1. Large train/CV gap = overfitting; both high and flat = underfitting.

    TabNet (model is None) is not sklearn-clonable and is skipped.
    """
    usable = {n: m for n, m in models.items() if m is not None}
    if not usable:
        return
    ncols = 4
    nrows = (len(usable) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for ax, (name, model) in zip(axes, usable.items()):
        X_use, y_use = _subsample(X_train, y_train, slow_max_rows if name in SLOW_MODELS else None)
        print(f"[learning curve] {name} ...")
        try:
            sizes, tr, va = learning_curve(clone(model), X_use, y_use, cv=cv, scoring="f1",
                                           train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1)
        except Exception as exc:
            print(f"  [error] {name}: {exc}")
            ax.set_title(f"{name} (failed)", fontsize=9)
            continue
        for scores, label, color in [(tr, "train", "#4c78a8"), (va, "CV", "#e45756")]:
            err = 1 - scores
            ax.plot(sizes, err.mean(axis=1), "o-", color=color, label=label)
            ax.fill_between(sizes, err.mean(axis=1) - err.std(axis=1),
                            err.mean(axis=1) + err.std(axis=1), color=color, alpha=0.15)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Training rows", fontsize=8)
        ax.set_ylabel("Error (1 - F1)", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(usable):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning curves (error vs training size)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Error analysis on out-of-fold predictions
# --------------------------------------------------------------------------- #
def error_analysis(preds: dict, y_true, out_dir: Path) -> pd.DataFrame:
    """Confusion matrices + ROC / PR / DET curves + FP/FN table for all models.

    preds: {model_name: (y_pred, y_score)}  (y_score may be None)
    Note: OOF scores come from different fold models, so pooled curves are a close
    approximation (most noticeable for SVM decision values, whose scale varies by fold).
    """
    y_true = np.asarray(y_true)
    names = list(preds)

    # Confusion matrices (row-normalised: each row sums to 1)
    ncols = 4
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()
    rows = []
    for ax, name in zip(axes, names):
        y_pred = np.asarray(preds[name][0])
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=[0, 1], display_labels=["legit", "phish"],
            normalize="true", values_format=".3f", cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(name, fontsize=9)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({"Model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "FPR": fp / (fp + tn), "FNR": fn / (fn + tp)})
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices, out-of-fold predictions (row-normalised)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    errors = pd.DataFrame(rows).sort_values("FPR").reset_index(drop=True)
    errors.to_csv(out_dir / "error_counts.csv", index=False)
    print("\nERROR COUNTS (out-of-fold, training split)")
    print(errors.round(4).to_string(index=False))

    # ROC / PR / DET overlays
    fig, (ax_roc, ax_pr, ax_det) = plt.subplots(1, 3, figsize=(16, 4.6))
    for name in names:
        score = preds[name][1]
        if score is None:
            continue
        score = np.asarray(score)
        fpr, tpr, _ = roc_curve(y_true, score)
        prec, rec, _ = precision_recall_curve(y_true, score)
        ax_roc.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, score):.4f})")
        ax_pr.plot(rec, prec, label=f"{name} ({average_precision_score(y_true, score):.4f})")
        ax_det.plot(np.clip(fpr, 1e-4, 1), np.clip(1 - tpr, 1e-4, 1), label=name)
    ax_roc.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax_roc.set(title="ROC (AUC)", xlabel="False positive rate", ylabel="True positive rate")
    ax_pr.set(title="Precision-Recall (AP)", xlabel="Recall", ylabel="Precision", ylim=(0.5, 1.01))
    ax_det.set(title="DET: lower-left is better", xlabel="False positive rate (log)",
               ylabel="False negative rate (log)", xscale="log", yscale="log")
    for ax in (ax_roc, ax_pr, ax_det):
        ax.grid(alpha=0.25)
    ax_roc.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_det.png", dpi=150)
    plt.close(fig)
    return errors


def misclassification_profile(name: str, y_true, y_pred, X: pd.DataFrame,
                              out_dir: Path, top_k: int = 10) -> None:
    """Which features differ most between errors and correctly classified rows?

    FP vs TN: legit sites flagged as phishing, compared with legit sites handled correctly.
    FN vs TP: phishing sites missed, compared with phishing sites caught.
    Effect = (mean_error_group - mean_reference_group) / overall std of the feature.
    """
    y_true = pd.Series(np.asarray(y_true), index=X.index)
    y_pred = pd.Series(np.asarray(y_pred), index=X.index)
    std = X.std().replace(0, np.nan)
    pairs = {
        "FP_vs_TN": ((y_pred == 1) & (y_true == 0), (y_pred == 0) & (y_true == 0)),
        "FN_vs_TP": ((y_pred == 0) & (y_true == 1), (y_pred == 1) & (y_true == 1)),
    }
    safe = name.replace(" ", "_")
    print(f"\nMISCLASSIFICATION PROFILE: {name}")
    for key, (err_mask, ref_mask) in pairs.items():
        if err_mask.sum() == 0 or ref_mask.sum() == 0:
            print(f"  {key}: no rows in one of the groups")
            continue
        effect = ((X[err_mask].mean() - X[ref_mask].mean()) / std).dropna()
        top = effect.reindex(effect.abs().sort_values(ascending=False).index).head(top_k)
        print(f"  {key}: {int(err_mask.sum())} error rows vs {int(ref_mask.sum())} reference rows")
        print(top.round(3).to_string())
        top.rename("standardised_mean_diff").to_csv(out_dir / f"misclassified_{key}_{safe}.csv")
        X[err_mask].assign(true=y_true[err_mask], pred=y_pred[err_mask]).to_csv(
            out_dir / f"misclassified_rows_{key}_{safe}.csv")


# --------------------------------------------------------------------------- #
# Convenience wrapper (called from baseline.main)
# --------------------------------------------------------------------------- #
def run_evaluation(models: dict, X_train, y_train, preds: dict, out_dir: Path,
                   do_curves: bool = True) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if do_curves:
        learning_curves(models, X_train, y_train, out_dir)

    if preds:
        errors = error_analysis(preds, y_train, out_dir)
        # profile the lowest-FPR model that still catches most phishing (FNR <= 10%)
        candidates = errors[errors["FNR"] <= 0.10]
        pick = (candidates if not candidates.empty else errors).iloc[0]["Model"]
        misclassification_profile(pick, y_train, preds[pick][0], X_train, out_dir)
    print(f"\nEvaluation outputs saved to {out_dir.resolve()}")


###############################################################################
# 5. ANALYSIS (feature-group ablation, importance, refinement, shortlist)
###############################################################################

ANALYSIS_DIR = Path("./reports/analysis")
CV_RESULTS = BASELINE_DIR / "cv_results_numeric.csv"

# Fast models for the ablation / importance runs (add "k-NN", "SVM" if you have time).
ABLATION_MODELS = ["Logistic Regression", "Naive Bayes", "Random Forest", "LightGBM"]
IMPORTANCE_MODELS = ["Logistic Regression", "Random Forest", "LightGBM", "MLP"]
SCALE_SENSITIVE = {"Logistic Regression", "k-NN", "SVM", "MLP"}   # distance / gradient based

# 3 = easiest to interpret. This is a judgement call: edit it and justify it in the report.
INTERPRETABILITY = {"Logistic Regression": 3, "Naive Bayes": 3, "Random Forest": 2,
                    "LightGBM": 2, "k-NN": 2, "SVM": 1, "MLP": 1, "TabNet": 1}


def _sklearn_models(names: list[str]) -> dict:
    """Pick the requested models; TabNet (None) and uninstalled libraries are dropped."""
    all_models = build_models()
    return {n: all_models[n] for n in names if all_models.get(n) is not None}


def _evaluate(name: str, model, X_tr, y_tr, X_va, y_va) -> dict:
    """Fit a fresh copy on train, score on validation (reuses baseline.fit_predict)."""
    y_pred, y_score = fit_predict(name, clone(model), X_tr, y_tr, X_va, SVM_MAX_TRAIN_ROWS)
    return compute_metrics(y_va, y_pred, y_score)


# --------------------------------------------------------------------------- #
# 1. Feature-group ablation: does the detector need the lookup features?
# --------------------------------------------------------------------------- #
def feature_group_ablation(X_tr, y_tr, X_va, y_va, out_dir: Path,
                           model_names: list[str] = ABLATION_MODELS) -> pd.DataFrame:
    groups = {c: feature_group(c) for c in X_tr.columns}

    def cols(*gs):
        return [c for c in X_tr.columns if groups[c] in gs]

    url_based = ("url", "domain", "directory", "file", "params")
    steps = {
        "URL": cols("url"),
        "+Domain": cols("url", "domain"),
        "+Directory": cols("url", "domain", "directory"),
        "+File": cols("url", "domain", "directory", "file"),
        "+Params\n(all URL-based)": cols(*url_based),
        "+External\n(all features)": list(X_tr.columns),
        "External\nonly": cols("external"),
    }
    seen, feature_sets = set(), {}
    for label, c in steps.items():                    # drop empty or repeated sets
        if c and tuple(c) not in seen:
            seen.add(tuple(c))
            feature_sets[label] = c

    models = _sklearn_models(model_names)
    rows = []
    for fs_name, c in feature_sets.items():
        for name, model in models.items():
            print(f"[ablation] {fs_name!r:28} | {name} ({len(c)} features)")
            m = _evaluate(name, model, X_tr[c], y_tr, X_va[c], y_va)
            rows.append({"Model": name, "Feature_set": fs_name.replace("\n", " "),
                         "N_features": len(c), **m})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "feature_group_ablation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = [k.replace("\n", " ") for k in feature_sets]
    for ax, metric in zip(axes, ["F1", "FPR"]):
        for name in models:
            sub = table[table.Model == name].set_index("Feature_set").reindex(labels)
            ax.plot(range(len(labels)), sub[metric], "o-", label=name)
        ax.set_xticks(range(len(labels)), list(feature_sets), fontsize=8)
        ax.set_ylabel(metric + (" (lower is better)" if metric == "FPR" else ""))
        ax.set_title(f"{metric} by feature set (validation)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_group_ablation.png", dpi=150)
    plt.close(fig)

    print("\nFEATURE-GROUP ABLATION (validation F1)")
    print(table.pivot(index="Feature_set", columns="Model", values="F1")
          .reindex([k.replace("\n", " ") for k in feature_sets]).round(4).to_string())
    return table


# --------------------------------------------------------------------------- #
# 2. Influencing factors: grouped + per-feature permutation importance
# --------------------------------------------------------------------------- #
def group_permutation_importance(fitted, X_va, y_va, n_repeats: int = 5) -> pd.DataFrame:
    """Drop in F1 when ALL columns of a feature group are shuffled together.

    Shuffling a whole group jointly is fairer than per-feature importance when
    features are redundant (a shuffled feature is "covered" by its twin).
    """
    base = f1_score(y_va, fitted.predict(X_va))
    groups = {}
    for c in X_va.columns:
        groups.setdefault(feature_group(c), []).append(c)

    rows = []
    for g, cols in groups.items():
        drops = []
        for r in range(n_repeats):
            perm = np.random.default_rng(RANDOM_SEED + r).permutation(len(X_va))
            X_p = X_va.copy()
            X_p[cols] = X_va[cols].to_numpy()[perm]
            drops.append(base - f1_score(y_va, fitted.predict(X_p)))
        rows.append({"group": g, "n_features": len(cols),
                     "f1_drop_mean": np.mean(drops), "f1_drop_std": np.std(drops)})
    return pd.DataFrame(rows)


def feature_importance(X_tr, y_tr, X_va, y_va, out_dir: Path,
                       model_names: list[str] = IMPORTANCE_MODELS) -> None:
    models = _sklearn_models(model_names)
    group_rows = []

    for name, model in models.items():
        print(f"[importance] fitting {name} ...")
        fitted = clone(model)
        fit_predict(name, fitted, X_tr, y_tr, X_va, SVM_MAX_TRAIN_ROWS)   # fits `fitted` in place

        g = group_permutation_importance(fitted, X_va, y_va)
        group_rows.append(g.assign(Model=name))

        # Per-feature permutation importance (+ built-in importances where available)
        perm = permutation_importance(fitted, X_va, y_va, scoring="f1", n_repeats=3,
                                      random_state=RANDOM_SEED, n_jobs=1)
        imp = pd.Series(perm.importances_mean, index=X_va.columns).sort_values(ascending=False)
        panels = [("Permutation importance (drop in F1, validation)", imp.head(15))]
        if hasattr(fitted, "feature_importances_"):
            built_in = pd.Series(fitted.feature_importances_, index=X_va.columns)
            panels.append(("Built-in importance (training)", built_in.sort_values(ascending=False).head(15)))

        fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), squeeze=False)
        for ax, (title, s) in zip(axes.ravel(), panels):
            s = s[::-1]
            ax.barh(s.index, s.values, color=[GROUP_COLORS[feature_group(c)] for c in s.index])
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.tick_params(axis="y", labelsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[k]) for k in GROUP_ORDER]
        axes.ravel()[-1].legend(handles, GROUP_ORDER, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / f"feature_importance_{name.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)
        imp.rename("permutation_f1_drop").to_csv(out_dir / f"feature_importance_{name.replace(' ', '_')}.csv")

    if not group_rows:
        return
    grp = pd.concat(group_rows, ignore_index=True)
    grp.to_csv(out_dir / "group_importance.csv", index=False)
    print("\nGROUP PERMUTATION IMPORTANCE (mean F1 drop when the whole group is shuffled)")
    print(grp.pivot(index="group", columns="Model", values="f1_drop_mean").round(4).to_string())

    pivot = grp.pivot(index="group", columns="Model", values="f1_drop_mean").reindex(
        [g for g in GROUP_ORDER if g in set(grp["group"])])
    ax = pivot.plot(kind="bar", figsize=(9, 4.5))
    ax.set_ylabel("F1 drop when group is shuffled")
    ax.set_title("Which feature groups does each model rely on? (validation)")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "group_importance.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 3. Refinement: raw vs scaled vs scaled + redundancy-pruned
# --------------------------------------------------------------------------- #
def redundancy_prune(X_tr: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Keep one feature per cluster of |Spearman| > threshold. Fit on TRAINING data only."""
    X = X_tr.loc[:, X_tr.nunique() > 1]                 # constant columns carry no information
    corr = X.corr(method="spearman").abs().fillna(0.0)
    dist = 1 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"),
                      t=1 - threshold, criterion="distance")
    kept = pd.Series(corr.index).groupby(labels).first().tolist()
    return [c for c in X_tr.columns if c in set(kept)]


def refinement_comparison(X_tr, y_tr, X_va, y_va, out_dir: Path) -> pd.DataFrame:
    kept = redundancy_prune(X_tr)
    print(f"\n[refine] redundancy pruning keeps {len(kept)} / {X_tr.shape[1]} features")
    variants = {
        "raw": (list(X_tr.columns), False),
        "scaled": (list(X_tr.columns), True),
        "scaled + pruned": (kept, True),
    }
    models = _sklearn_models(["Logistic Regression", "Naive Bayes", "k-NN", "SVM",
                              "Random Forest", "LightGBM", "MLP"])
    rows = []
    for variant, (cols, scale) in variants.items():
        for name, model in models.items():
            print(f"[refine] {variant:16} | {name}")
            est = make_pipeline(StandardScaler(), clone(model)) if (scale and name in SCALE_SENSITIVE) else model
            m = _evaluate(name, est, X_tr[cols], y_tr, X_va[cols], y_va)
            rows.append({"Model": name, "Variant": variant, "N_features": len(cols), **m})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "refinement_results.csv", index=False)

    print("\nREFINEMENT (validation F1 / FPR)")
    for metric in ["F1", "FPR"]:
        print(f"\n{metric}")
        print(table.pivot(index="Model", columns="Variant", values=metric)[list(variants)].round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric in zip(axes, ["F1", "FPR"]):
        table.pivot(index="Model", columns="Variant", values=metric)[list(variants)].plot(kind="bar", ax=ax)
        ax.set_title(f"{metric} by variant (validation)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "refinement_results.png", dpi=150)
    plt.close(fig)
    return table


# --------------------------------------------------------------------------- #
# 4. Shortlist table
# --------------------------------------------------------------------------- #
def shortlist_table(out_dir: Path, cv_path: Path = CV_RESULTS) -> pd.DataFrame | None:
    """Rank models on CV F1, FPR, ROC-AUC, stability (F1 std) and speed.

    The rank table is a starting point; the shortlist still needs a written
    justification (performance, interpretability, speed, family diversity).
    """
    if not cv_path.exists():
        print(f"[shortlist] {cv_path} not found -- run a_baseline.py first.")
        return None
    cv = pd.read_csv(cv_path)
    table = pd.DataFrame({
        "Model": cv["Model"],
        "F1": cv["F1_mean"], "F1_std": cv["F1_std"],
        "FPR": cv["FPR_mean"], "ROC_AUC": cv["ROC_AUC_mean"],
        "Time_Sec": cv["Mean_Fold_Time_Sec"],
        "Interpretability": cv["Model"].map(INTERPRETABILITY),
    })
    ranks = pd.DataFrame({
        "rank_F1": table["F1"].rank(ascending=False),
        "rank_FPR": table["FPR"].rank(ascending=True),
        "rank_AUC": table["ROC_AUC"].rank(ascending=False),
        "rank_stability": table["F1_std"].rank(ascending=True),
        "rank_speed": table["Time_Sec"].rank(ascending=True),
    })
    table["avg_rank"] = ranks.mean(axis=1)
    table = pd.concat([table, ranks], axis=1).sort_values("avg_rank").reset_index(drop=True)
    table.to_csv(out_dir / "shortlist.csv", index=False)
    print("\nSHORTLIST CANDIDATES (lower avg_rank is better)")
    print(table[["Model", "F1", "F1_std", "FPR", "ROC_AUC", "Time_Sec",
                 "Interpretability", "avg_rank"]].round(4).to_string(index=False))
    return table


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_analysis() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X_tr, X_va, X_te, y_tr, y_va, y_te = train_val_test(df, val_size=0.2, test_size=0.2, seed=RANDOM_SEED)
    print(f"Train: {len(X_tr):,} | Validation: {len(X_va):,} (test split not used here)")

    for step, fn in [
        ("feature-group ablation", lambda: feature_group_ablation(X_tr, y_tr, X_va, y_va, ANALYSIS_DIR)),
        ("feature importance", lambda: feature_importance(X_tr, y_tr, X_va, y_va, ANALYSIS_DIR)),
        ("refinement comparison", lambda: refinement_comparison(X_tr, y_tr, X_va, y_va, ANALYSIS_DIR)),
        ("shortlist", lambda: shortlist_table(ANALYSIS_DIR)),
    ]:
        try:
            fn()
        except Exception as exc:          # one failing step should not stop the others
            print(f"\n[error] {step} failed: {type(exc).__name__}: {exc}")
    print(f"\nAnalysis outputs saved to {ANALYSIS_DIR.resolve()}")


###############################################################################
# 6. CURVES (full-size learning curves incl. TabNet, iteration curves)
###############################################################################

CURVES_DIR = Path("./reports/curves")


# --------------------------------------------------------------------------- #
# 1. Learning curves to the full training size (all models, TabNet included)
# --------------------------------------------------------------------------- #
def learning_curves_full(models: dict, X_train, y_train, X_val, y_val, out_dir: Path,
                         fractions=(0.1, 0.25, 0.5, 0.75, 1.0), repeats: int = 2,
                         probe_rows: int = 8000) -> pd.DataFrame:
    """Error = 1 - F1. Train error is measured on (up to) probe_rows rows of the training subset."""
    rows = []
    for name, model in models.items():
        for frac in fractions:
            n = int(round(frac * len(X_train)))
            reps = 1 if n >= len(X_train) else repeats      # the full set is deterministic
            for r in range(reps):
                if n >= len(X_train):
                    X_sub, y_sub = X_train, y_train
                else:
                    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=n, stratify=y_train,
                                                          random_state=RANDOM_SEED + r)
                probe = X_sub.iloc[:min(len(X_sub), probe_rows)]
                X_eval = pd.concat([X_val, probe])              # one fit, two evaluations
                print(f"[learning curve] {name:20} n={n:>6,} repeat {r + 1}/{reps}")
                try:
                    y_pred, _ = fit_predict(name, clone(model) if model is not None else None,
                                            X_sub, y_sub, X_eval, SVM_MAX_TRAIN_ROWS)
                except Exception as exc:
                    print(f"  [error] {name} n={n}: {exc}")
                    continue
                y_pred = np.asarray(y_pred)
                rows.append({
                    "Model": name, "n_requested": n,
                    "n_used": min(n, SVM_MAX_TRAIN_ROWS) if (name == "SVM" and SVM_MAX_TRAIN_ROWS) else n,
                    "repeat": r,
                    "val_error": 1 - f1_score(y_val, y_pred[:len(X_val)]),
                    "train_error": 1 - f1_score(y_sub.iloc[:len(probe)], y_pred[len(X_val):]),
                })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.to_csv(out_dir / "learning_curves_full.csv", index=False)

    names = list(models)
    ncols = 4
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        sub = table[table.Model == name]
        if sub.empty:
            ax.set_title(f"{name} (no results)", fontsize=9)
            continue
        g = sub.groupby("n_used")[["train_error", "val_error"]].agg(["mean", "std"]).fillna(0)
        for col, label, color in [("train_error", "train", "#4c78a8"), ("val_error", "validation", "#e45756")]:
            ax.plot(g.index, g[(col, "mean")], "o-", color=color, label=label)
            ax.fill_between(g.index, g[(col, "mean")] - g[(col, "std")],
                            g[(col, "mean")] + g[(col, "std")], color=color, alpha=0.15)
        cap = " (capped)" if (name == "SVM" and SVM_MAX_TRAIN_ROWS and sub["n_requested"].max() > SVM_MAX_TRAIN_ROWS) else ""
        ax.set_title(name + cap, fontsize=9)
        ax.set_xlabel("Training rows used", fontsize=8)
        ax.set_ylabel("Error (1 - F1)", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning curves to full training size (scored on the validation split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves_full.png", dpi=150)
    plt.close(fig)
    return table


# --------------------------------------------------------------------------- #
# 2. Iteration curves: where does each iterative model stop improving?
# --------------------------------------------------------------------------- #
def iteration_curves(X_train, y_train, X_val, y_val, out_dir: Path, rf_max_trees: int = 300,
                     mlp_epochs: int = 60, lgbm_trees: int = 1000, tabnet_epochs: int = 150) -> None:
    panels = {}     # title -> list of (label, x, y, ylabel)
    conv_rows = []

    # Random Forest: validation F1 as trees are added (warm_start keeps earlier trees)
    rf = RandomForestClassifier(n_estimators=25, warm_start=True, random_state=RANDOM_SEED, n_jobs=CPU_JOBS)
    xs, ys = [], []
    for n in range(25, rf_max_trees + 1, 25):
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)
        xs.append(n)
        ys.append(f1_score(y_val, rf.predict(X_val)))
        print(f"[iteration] Random Forest trees={n} val F1={ys[-1]:.4f}")
    panels["Random Forest: validation F1 vs trees"] = [("val F1", xs, ys, "F1")]
    conv_rows.append({"Model": "Random Forest", "setting": "n_estimators", "default": 100,
                      "best_at": xs[int(np.argmax(ys))], "note": f"max val F1 {max(ys):.4f}"})

    # MLP: raw vs scaled inputs, epoch by epoch
    scaler = StandardScaler().fit(X_train)
    curves = []
    for label, Xt, Xv in [("raw inputs", X_train.to_numpy(), X_val.to_numpy()),
                          ("scaled inputs", scaler.transform(X_train), scaler.transform(X_val))]:
        mlp = MLPClassifier(random_state=RANDOM_SEED)
        ep, f1s = [], []
        for epoch in range(1, mlp_epochs + 1):
            mlp.partial_fit(Xt, y_train, classes=[0, 1])
            ep.append(epoch)
            f1s.append(f1_score(y_val, mlp.predict(Xv)))
        print(f"[iteration] MLP {label}: final val F1={f1s[-1]:.4f}, best={max(f1s):.4f} at epoch {ep[int(np.argmax(f1s))]}")
        curves.append((label, ep, f1s, "F1"))
    panels["MLP: validation F1 vs epoch"] = curves

    # LightGBM: validation log-loss per tree, best iteration marked
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(n_estimators=lgbm_trees, random_state=RANDOM_SEED, verbosity=-1)
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=["binary_logloss", "binary_error"])
        res = lgbm.evals_result_["valid_0"]
        ll = res["binary_logloss"]
        best = int(np.argmin(ll)) + 1
        panels["LightGBM: validation log-loss vs trees"] = [("val log-loss", list(range(1, len(ll) + 1)), ll, "log-loss")]
        conv_rows.append({"Model": "LightGBM", "setting": "n_estimators", "default": 100, "best_at": best,
                          "note": f"min val log-loss {min(ll):.4f}; error {min(res['binary_error']):.4f}"})
    except ImportError:
        print("[iteration] LightGBM not installed -- skipped")

    # TabNet: validation log-loss per epoch with early stopping
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        tab = TabNetClassifier(seed=RANDOM_SEED, verbose=0)
        tab.fit(X_train.to_numpy(np.float32), y_train.to_numpy(np.int64),
                eval_set=[(X_val.to_numpy(np.float32), y_val.to_numpy(np.int64))],
                eval_name=["val"], eval_metric=["logloss"],
                max_epochs=tabnet_epochs, patience=20, batch_size=1024, virtual_batch_size=128)
        vl = tab.history["val_logloss"]
        panels["TabNet: validation log-loss vs epoch"] = [("val log-loss", list(range(1, len(vl) + 1)), vl, "log-loss")]
        conv_rows.append({"Model": "TabNet", "setting": "max_epochs", "default": 50, "best_at": int(tab.best_epoch) + 1,
                          "note": f"stopped after {len(vl)} epochs (patience 20)"})
    except ImportError:
        print("[iteration] pytorch-tabnet not installed -- skipped")

    # Did the default LR / MLP settings actually converge? (the baseline silences ConvergenceWarning)
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000).fit(X_train, y_train)
    mlp_default = MLPClassifier(random_state=RANDOM_SEED, max_iter=300).fit(X_train, y_train)
    conv_rows.append({"Model": "Logistic Regression", "setting": "max_iter", "default": 1000,
                      "best_at": int(lr.n_iter_[0]), "note": "converged" if lr.n_iter_[0] < 1000 else "HIT max_iter (not converged)"})
    conv_rows.append({"Model": "MLP (raw inputs)", "setting": "max_iter", "default": 300,
                      "best_at": int(mlp_default.n_iter_), "note": "converged" if mlp_default.n_iter_ < 300 else "HIT max_iter (not converged)"})
    table = pd.DataFrame(conv_rows)
    table.to_csv(out_dir / "convergence_table.csv", index=False)
    print("\nCONVERGENCE CHECK")
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), squeeze=False)
    for ax, (title, series) in zip(axes.ravel(), panels.items()):
        for label, x, y, ylabel in series:
            ax.plot(x, y, label=label)
        if len(series) == 1:
            x, y = series[0][1], series[0][2]
            best = int(np.argmax(y)) if "F1" in title else int(np.argmin(y))
            ax.axvline(x[best], color="k", ls=":", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel(series[0][3], fontsize=8)
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "iteration_curves.png", dpi=150)
    plt.close(fig)


def run_curves() -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df, val_size=0.2, test_size=0.2, seed=RANDOM_SEED)
    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} (test split not used)")

    models = build_models()
    for step, fn in [
        ("iteration curves", lambda: iteration_curves(X_train, y_train, X_val, y_val, CURVES_DIR)),
        ("learning curves", lambda: learning_curves_full(models, X_train, y_train, X_val, y_val, CURVES_DIR)),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"\n[error] {step} failed: {type(exc).__name__}: {exc}")
    print(f"\nCurve outputs saved to {CURVES_DIR.resolve()}")


###############################################################################
# 7. COMMAND LINE
###############################################################################



STAGES = {
    "eda": run_eda,
    "baseline": run_baseline,
    "analysis": run_analysis,
    "curves": run_curves,
}


def main() -> None:
    global FULL_FILE
    parser = argparse.ArgumentParser(description="PhishLite Part A pipeline")
    parser.add_argument("stage", nargs="?", default="all", choices=[*STAGES, "all"],
                        help="which stage to run (default: all, in order)")
    parser.add_argument("--data", default=None,
                        help=f"path to dataset_full.csv (default: {FULL_FILE})")
    args = parser.parse_args()
    if args.data:
        FULL_FILE = args.data

    todo = list(STAGES) if args.stage == "all" else [args.stage]
    print(f"Dataset: {Path(FULL_FILE).resolve()}")
    for stage in todo:
        print(f"\n{'=' * 70}\nSTAGE: {stage}\n{'=' * 70}")
        try:
            STAGES[stage]()
        except FileNotFoundError as exc:
            print(f"[FILE ERROR] {exc}")
            if stage == "eda" or args.stage != "all":
                break
        except Exception as exc:                      # keep going so one failure does not hide the rest
            print(f"[{stage} FAILED] {type(exc).__name__}: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()