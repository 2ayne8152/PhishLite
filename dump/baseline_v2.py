"""
Part A baseline: compare models with stratified k-fold cross-validation.

Experimental rules:
- No scaling, feature selection, class balancing, or feature removal.
- Cross-validation runs on the TRAINING split only. The validation and test
  splits are not touched here (reserved for Part B tuning and final evaluation).
- Library defaults are retained except for documented execution-related settings.
- Out-of-fold (OOF) predictions are kept so error analysis uses every training
  row exactly once, predicted by a model that never saw that row.

Models: Logistic Regression, Naive Bayes, k-NN, SVM, Random Forest, LightGBM, TabNet, MLP.

Metrics (per fold, then mean +/- std): Accuracy, Precision, Recall, F1, FPR, ROC-AUC, MCC.

Outputs (reports/baseline/):
    cv_results.csv          mean +/- std per model (for the report)
    cv_results_numeric.csv  same, numeric (for sorting / plotting)
    cv_fold_metrics.csv     one row per model per fold (for paired tests later)
    oof_predictions.csv     out-of-fold prediction and score per model
plus the figures/tables written by evaluate.py.
"""
from __future__ import annotations

import itertools, sys, threading, time, warnings, os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import clone
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from load import RANDOM_SEED, train_val_test, load_dataset
import evaluation

OUT_DIR = Path('./reports/baseline')

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

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    numeric.to_csv(OUT_DIR / "cv_results_numeric.csv", index=False)
    pretty.to_csv(OUT_DIR / "cv_results.csv", index=False)
    pd.concat([d.assign(Model=n) for n, d in fold_results.items()]).to_csv(
        OUT_DIR / "cv_fold_metrics.csv", index=False)
    pd.DataFrame({**{f"{n}_pred": p for n, (p, _) in oof.items()},
                  **{f"{n}_score": s for n, (_, s) in oof.items()},
                  "true": y_train}).to_csv(OUT_DIR / "oof_predictions.csv")

    print(f"\n{'=' * 70}\nBASELINE CV RESULTS (saved to {OUT_DIR.resolve()})\n{'=' * 70}")
    print(pretty.to_string(index=False))

    # Error analysis and learning curves (see evaluate.py)
    try:
        preds = {n: (p.to_numpy(), s.to_numpy() if s.notna().all() else None) for n, (p, s) in oof.items()}
        evaluation.run_evaluation(models, X_train, y_train, preds, OUT_DIR,
                                do_curves=RUN_LEARNING_CURVES)
    except Exception as exc:
        print(f"\n[error] evaluation stage failed: {type(exc).__name__}: {exc}")
        print("CV results above were already saved; re-run evaluate on oof_predictions.csv if needed.")


if __name__ == "__main__":
    main()