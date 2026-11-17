"""Part A baseline: run each model near out-of-the-box, no feature work.

Philosophy (per project spec):
- The dataset itself is NOT touched: no scaling, no feature selection,
    no de-duplication, no class-balancing, no leakage/multicollinearity
    fixes. Those all belong to Part B.
- Models use library DEFAULT hyperparameters. The only changes made are
    the minimum needed to make a model actually *run* to completion
    (e.g. raising max_iter so LogisticRegression/MLP don't error out,
    or giving TabNet an epoch budget since it has no sane default) --
    never changes aimed at improving performance.
- A single stratified train/test split is used (not touching the
    data, just evaluating it) with a fixed seed for reproducibility.

Models: Logistic Regression, Naive Bayes, k-NN, SVM, Random Forest,
LightGBM, TabNet, MLP.

Metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC, MCC.
Results are written to a CSV.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from load import DEFAULT_LABEL, DEFAULT_PATH

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_SEED = 42

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
    "Logistic Regression",
    "Naive Bayes",
    "k-NN",
    "SVM",
    "Random Forest",
    "LightGBM",
    "TabNet",
    "MLP",
]


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
# Data loading (deliberately raw -- no cleaning beyond what's needed to run)
# --------------------------------------------------------------------------- #
def load_raw(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    n_dupe = df.duplicated().sum()
    n_nan = df.isna().sum().sum()
    if n_dupe:
        print(f"[info] {n_dupe:,} duplicate rows present (left as-is; Part B concern).")
    if n_nan:
        print(f"[info] {n_nan:,} missing values present across the dataset.")
        num_cols = df.select_dtypes("number").columns.drop(label_col, errors="ignore")
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df = df.dropna(subset=[label_col])

    return df


# --------------------------------------------------------------------------- #
# Model registry -- library defaults only, plus the minimum needed to run
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_SEED, max_iter=1000
        ),
        "Naive Bayes": GaussianNB(),
        "k-NN": KNeighborsClassifier(),
        "SVM": SVC(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "MLP": MLPClassifier(random_state=RANDOM_SEED, max_iter=300),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1)
    else:
        print("[skip] LightGBM not installed (`pip install lightgbm`) -- skipping.")
    if HAS_TABNET:
        models["TabNet"] = None
    else:
        print("[skip] pytorch-tabnet not installed (`pip install pytorch-tabnet torch`) -- skipping.")
    return models


def fit_predict(name: str, model, X_train, y_train, X_test, svm_max_train: int | None):
    """Fit one model and return (y_pred, y_score) where y_score is used for ROC-AUC."""
    if name == "SVM" and svm_max_train is not None and len(X_train) > svm_max_train:
        X_fit, _, y_fit, _ = train_test_split(
            X_train, y_train,
            train_size=svm_max_train,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
        print(f"  (SVM trained on a stratified subsample of {svm_max_train:,} rows, not the full set)")
    else:
        X_fit, y_fit = X_train, y_train

    if name == "TabNet":
        # verbose=1 prints native epoch-by-epoch progress during training
        clf = TabNetClassifier(seed=RANDOM_SEED, verbose=1)
        clf.fit(
            X_fit.values.astype(np.float32),
            y_fit.values.astype(np.int64),
            max_epochs=50,
            patience=50,
            batch_size=1024,
            virtual_batch_size=128,
        )
        with live_spinner("TabNet predicting on test set"):
            y_pred = clf.predict(X_test.values.astype(np.float32))
            y_score = clf.predict_proba(X_test.values.astype(np.float32))[:, 1]
        return y_pred, y_score

    clf = model

    if name == "LightGBM" and HAS_TQDM:
        cb = lgbm_tqdm_callback(clf.n_estimators)
        clf.fit(X_fit, y_fit, callbacks=[cb])
    else:
        with live_spinner(f"Fitting {name}"):
            clf.fit(X_fit, y_fit)

    with live_spinner(f"Predicting with {name}"):
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
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
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr,
        "ROC_AUC": roc_auc,
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    global RANDOM_SEED
    parser = argparse.ArgumentParser(description="Part A: baseline model comparison")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", default="reports/baseline_results.csv")
    parser.add_argument(
        "--models", default="all",
        help="Comma-separated subset of: " + ", ".join(ALL_MODEL_NAMES) + " (default: all)",
    )
    parser.add_argument(
        "--svm-max-train", type=int, default=None,
        help="Stratified subsample size for SVM training only (RBF SVC scales poorly). "
             "Default: None = use the full training set (slow on ~70k rows).",
    )
    parser.add_argument(
        "--sample-frac", type=float, default=None,
        help="Optional stratified subsample of the WHOLE dataset, for a quick dry run. "
             "Default: None = use the full dataset (this is the real baseline).",
    )
    args = parser.parse_args()

    RANDOM_SEED = args.seed

    df = load_raw(Path(args.path), args.label)

    if args.sample_frac is not None:
        df, _ = train_test_split(
            df, train_size=args.sample_frac, stratify=df[args.label], random_state=RANDOM_SEED
        )
        print(f"[info] --sample-frac set: using {len(df):,} rows for this run (dry run only).")

    X = df.drop(columns=[args.label])
    y = df[args.label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Features: {X.shape[1]}")

    requested = ALL_MODEL_NAMES if args.models == "all" else [m.strip() for m in args.models.split(",")]
    models = build_models()

    # Filter runnable models first so the [i/N] counter is exact
    runnable = []
    for name in requested:
        if name not in ALL_MODEL_NAMES:
            print(f"[warn] Unknown model '{name}', skipping.")
            continue
        if name == "LightGBM" and not HAS_LGBM:
            continue
        if name == "TabNet" and not HAS_TABNET:
            continue
        runnable.append(name)

    results = []
    total = len(runnable)
    for idx, name in enumerate(runnable, start=1):
        print(f"\n[{idx}/{total}] {name}")
        start = time.time()
        try:
            y_pred, y_score = fit_predict(name, models.get(name), X_train, y_train, X_test, args.svm_max_train)
            elapsed = time.time() - start
            metrics = compute_metrics(y_test, y_pred, y_score)
            metrics["Model"] = name
            metrics["Train_Time_Sec"] = round(elapsed, 2)
            results.append(metrics)
            print(f"  done in {elapsed:.1f}s | "
                  f"Acc={metrics['Accuracy']:.4f} F1={metrics['F1']:.4f} ROC-AUC={metrics['ROC_AUC']:.4f}")
        except Exception as exc:
            print(f"  [error] {name} failed: {exc}")

    if not results:
        print("No models produced results.")
        return

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "MCC", "Train_Time_Sec"]
    results_df = pd.DataFrame(results)[cols].sort_values("F1", ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}\nBASELINE RESULTS (saved to {out_path.resolve()})\n{'=' * 70}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()