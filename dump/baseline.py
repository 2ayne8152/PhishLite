"""
Part A baseline: compare models with minimal model-specific adjustments.

Experimental rules:
- No scaling, feature selection, class balancing, or feature removal.
- The validation set is used for baseline comparison.
- The final test set is held back for later final evaluation.
- Library defaults are retained except for documented execution-related settings.

Models: Logistic Regression, Naive Bayes, k-NN, SVM, Random Forest, LightGBM, TabNet, MLP.

Metrics: Accuracy, Precision, Recall, F1, FPR, ROC-AUC, MCC.
Results are written to a CSV.
"""
from __future__ import annotations

import itertools, sys, threading, time, warnings, os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import numpy as np
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from load import RANDOM_SEED, train_val_test, load_dataset

OUT_DIR = Path('./reports/baseline')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SVM_MAX_TRAIN_ROWS: int | None = None
CPU_JOBS = max(1, (os.cpu_count() or 2) - 2)

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
        models["TabNet"] = None
    else:
        print("[skip] pytorch-tabnet not installed (`pip install pytorch-tabnet torch`) -- skipping.")
    return models


def fit_predict(name: str, model, X_train, y_train, X_val, svm_max_train: int | None):
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
            print(
                f"  (SVM trained on a stratified subset of "
                f"{len(X_fit):,} / {len(X_train):,} training rows)"
            )
        else:
            X_fit, y_fit = X_train, y_train

    else:
        X_fit, y_fit = X_train, y_train

    if name == 'TabNet':
        clf = TabNetClassifier(seed=RANDOM_SEED, verbose=1)

        X_fit_array = X_fit.to_numpy(dtype=np.float32)
        X_val_array = X_val.to_numpy(dtype=np.float32)
        y_fit_array = y_fit.to_numpy(dtype=np.int64)

        clf.fit(X_fit_array,
                y_fit_array,
                max_epochs = 50,
                patience = 50,
                batch_size = 1024,
                virtual_batch_size=128
        )
        with live_spinner("TabNet predicting on val set"):
            y_pred = clf.predict(X_val_array)
            y_score = clf.predict_proba(X_val_array)[:, 1]
        return y_pred, y_score

    clf = model

    if name == "LightGBM" and HAS_TQDM:
        cb = lgbm_tqdm_callback(clf.n_estimators)
        clf.fit(X_fit, y_fit, callbacks=[cb])
    else:
        with live_spinner(f"Fitting {name}"):
            clf.fit(X_fit, y_fit)

    with live_spinner(f"Predicting with {name}"):
        y_pred = clf.predict(X_val)

        if hasattr(clf, "predict_proba"):
            probabilities = clf.predict_proba(X_val)
            positive_index = list(clf.classes_).index(1)
            y_score = probabilities[:, positive_index]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_val)
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
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
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

    models = build_models()
    runnable_names = [ name for name in ALL_MODEL_NAMES if name in models]
    results = []
    for idx, name in enumerate(runnable_names, start=1):
        print(f"\n[{idx}/{len(runnable_names)}] {name}")
        start = time.time()

        try:
            y_pred, y_score = fit_predict(
                name, models[name],
                X_train, y_train,
                X_val, SVM_MAX_TRAIN_ROWS
            )
            elapsed = time.time() - start
            metrics = compute_metrics(y_val, y_pred, y_score)
            metrics["Model"] = name
            metrics["Total_Time_Sec"] = round(elapsed, 2)
            results.append(metrics)
            print(f"  done in {elapsed:.1f}s | "
                  f"Acc={metrics['Accuracy']:.4f} F1={metrics['F1']:.4f} ROC-AUC={metrics['ROC_AUC']:.4f}")
        except Exception as exc:
            print(f"  [error] {name} failed: {exc}")

    if not results:
        print("No models produced results.")
        return

    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "MCC", "Total_Time_Sec"]
    results_df = pd.DataFrame(results)[cols].sort_values("F1", ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / "baseline_results.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}\nBASELINE RESULTS (saved to {out_path.resolve()})\n{'=' * 70}")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()