"""
a_baseline.py - Part A: untuned baseline models, stratified CV with learning curves, error analysis,
per-family shortlist.

Baseline cross-validation uses the TRAINING split only; the validation and test
splits are not used here. Library defaults, no tuning, no feature selection.
"""
from __future__ import annotations

import itertools
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load import RANDOM_SEED, load_dataset, train_val_test, make_seeds


BASELINE_DIR = Path('./reports/baseline')

CV_FOLDS = 5
SEEDS = make_seeds(n=3, master=RANDOM_SEED)
CPU_JOBS = max(1, (os.cpu_count() or 2) - 2)

# Training curves are recorded inside the CV loop (see cross_validate_model).
CURVE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)   # shares of each fold's training rows (1.0 = the baseline fit)
CURVE_PROBE_ROWS = 8000                        # rows used to measure training error (keeps k-NN/SVM scoring fast)
CURVE_MAX_FOLDS = {"SVM": 2, "TabNet": 2}      # slow models: partial-size fits on the first N folds only

# For robustness check under different conditions.
# Conditions to compare. A slice is skipped if its column is not in the data.
SLICES = {
    "URL has no query params": lambda X: X["params_length"] == -1,
    "URL has query params": lambda X: X["params_length"] != -1,
    "short URL (below median length)": lambda X: X["length_url"] < X["length_url"].median(),
    "long URL (median length or more)": lambda X: X["length_url"] >= X["length_url"].median(),
    "domain activation time unavailable (-1)": lambda X: X["time_domain_activation"] == -1,
}
# =========================
# For shortlisting only --

# Shortlist families (fixed before looking at results).
FAMILIES = {
    "Linear / probabilistic": ["Logistic Regression", "Naive Bayes"],
    "Distance / geometric": ["k-NN", "SVM"],
    "Tree-based": ["Random Forest", "LightGBM"],
    "Neural network": ["MLP", "TabNet"],
}
# 3 = easiest to interpret. A judgment call: justify it in the report.
INTERPRETABILITY = {"Logistic Regression": 3, "Naive Bayes": 3, "Random Forest": 2,
                    "LightGBM": 2, "k-NN": 2, "SVM": 1, "MLP": 1, "TabNet": 1}

# =========================


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

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "FPR", "ROC_AUC", "MCC", "PR_AUC"]

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
    """Build the baseline models: library defaults plus the minimum needed to run.

    Simple preprocessing: scale-sensitive models (LR, k-NN, SVM, MLP) get a StandardScaler
    inside a pipeline, so the scaler is re-fitted on each training fold (no leakage).
    Trees and Naive Bayes are scale-invariant and are left unscaled. TabNet is scaled
    inside fit_predict because it is not sklearn-clonable.
    """
    def scaled(estimator):
        return make_pipeline(StandardScaler(), estimator)

    models = {
        "Logistic Regression": scaled(LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)),
        "Naive Bayes": GaussianNB(),
        "k-NN": scaled(KNeighborsClassifier(n_jobs=max(2, CPU_JOBS))),
        "SVM": scaled(SVC(random_state=RANDOM_SEED)),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=max(2, CPU_JOBS)),
        "MLP": scaled(MLPClassifier(random_state=RANDOM_SEED, max_iter=300)),
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


def fit_predict(name: str, model, X_train, y_train, X_eval, seed: int=RANDOM_SEED):
    """Fit one model on (X_train, y_train) and predict X_eval, using ALL the rows given.

    Returns (y_pred, y_score, fit_seconds). y_score feeds ROC-AUC; fit_seconds times the
    fit only (not the prediction), so it can be compared fairly between models.
    """
    if name == "TabNet":
        scaler = StandardScaler().fit(X_train)            # fitted on the training rows only
        X_fit_array = scaler.transform(X_train).astype(np.float32)
        X_eval_array = scaler.transform(X_eval).astype(np.float32)
        clf = TabNetClassifier(seed=seed, verbose=0)
        start = time.time()
        clf.fit(X_fit_array, y_train.to_numpy(dtype=np.int64),
                max_epochs=50, patience=50, batch_size=1024, virtual_batch_size=128)
        fit_sec = time.time() - start
        return clf.predict(X_eval_array), clf.predict_proba(X_eval_array)[:, 1], fit_sec

    clf = model
    start = time.time()
    if name == "LightGBM" and HAS_TQDM:
        clf.fit(X_train, y_train, callbacks=[lgbm_tqdm_callback(clf.n_estimators)])
    else:
        clf.fit(X_train, y_train)
    fit_sec = time.time() - start

    y_pred = clf.predict(X_eval)
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(X_eval)
        y_score = probabilities[:, list(clf.classes_).index(1)]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_eval)
    else:
        y_score = None
    return y_pred, y_score, fit_sec


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
        "PR_AUC":average_precision_score(y_true, y_score) if y_score is not None else float("nan")
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def _error_pair(y_true, y_pred, y_score) -> tuple[float, float]:
    """(1 - ROC-AUC, 1 - F1). AUC is threshold-free and exists for every model, SVM included."""
    f1_err = 1 - f1_score(y_true, y_pred, zero_division=0)
    try:
        auc_err = 1 - roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    except ValueError:
        auc_err = float("nan")
    return auc_err, f1_err


def cross_validate_model(name: str, model, X: pd.DataFrame, y: pd.Series,
                         n_splits: int = CV_FOLDS,
                         seed: int = RANDOM_SEED, with_curves: bool = True):
    """Stratified k-fold CV for one model, with the learning curve recorded in the same loop.

    In every fold the model is fitted on CURVE_FRACTIONS of the fold's training rows. The 100%
    fit IS the baseline fit: its predictions on the held-out rows give the CV metrics and the
    out-of-fold predictions, so nothing is trained twice. Each fit also scores a probe of the
    rows it was trained on, which gives the training error of the curve.

    Returns:
        fold_df    one row per fold: baseline metrics + fit time (100% fits only)
        oof_pred   out-of-fold class prediction for every row of X
        oof_score  out-of-fold score (probability or decision value) for every row of X
        curve_df   one row per (fold, fraction): training / validation error and fit time
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = pd.Series(np.nan, index=X.index, dtype=float)
    oof_score = pd.Series(np.nan, index=X.index, dtype=float)
    rows, curve_rows = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        n_te = len(X_te)
        # slow models only get the partial-size fits on the first few folds (disclose in the report)
        fractions = CURVE_FRACTIONS if (with_curves and fold <= CURVE_MAX_FOLDS.get(name, n_splits)) else (1.0,)
        for frac in fractions:
            if frac >= 1.0:
                X_sub, y_sub = X_tr, y_tr
            else:
                X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=frac, stratify=y_tr,
                                                      random_state=seed + fold)
            probe = X_sub.iloc[:min(len(X_sub), CURVE_PROBE_ROWS)]     # rows are already shuffled
            y_probe = y_sub.iloc[:len(probe)]
            X_eval = pd.concat([X_te, probe])                          # one fit, two evaluations

            fold_model = clone(model) if model is not None else None   # fresh copy: nothing leaks
            if fold_model is not None:     # the model's own randomness follows the seed too
                fold_model.set_params(**{k: seed for k in fold_model.get_params() if k.endswith("random_state")})
            spinner = nullcontext() if (name == "LightGBM" and HAS_TQDM) else \
                live_spinner(f"{name} fold {fold}/{n_splits} @ {frac:.0%}")
            with spinner:
                y_pred, y_score, fit_sec = fit_predict(name, fold_model, X_sub, y_sub, X_eval, seed=seed)

            y_pred = np.asarray(y_pred)
            y_score = None if y_score is None else np.asarray(y_score)
            va_pred, tr_pred = y_pred[:n_te], y_pred[n_te:]
            va_score = None if y_score is None else y_score[:n_te]
            tr_score = None if y_score is None else y_score[n_te:]

            va_auc_err, va_f1_err = _error_pair(y_te, va_pred, va_score)
            tr_auc_err, tr_f1_err = _error_pair(y_probe, tr_pred, tr_score)
            curve_rows.append({"Model": name, "Fold": fold, "Frac": frac, "n_train": len(X_sub),
                               "train_auc_err": tr_auc_err, "val_auc_err": va_auc_err,
                               "train_f1_err": tr_f1_err, "val_f1_err": va_f1_err,
                               "fit_sec": round(fit_sec, 2)})

            if frac >= 1.0:                                            # the baseline fit
                metrics = compute_metrics(y_te, va_pred, va_score)
                metrics.update({"Fold": fold, "Time_Sec": round(fit_sec, 2)})
                rows.append(metrics)
                oof_pred.iloc[test_idx] = va_pred
                if va_score is not None:
                    oof_score.iloc[test_idx] = va_score
                print(f"  fold {fold}/{n_splits} | fit {fit_sec:6.1f}s | "
                      f"F1={metrics['F1']:.4f} FPR={metrics['FPR']:.4f} ROC-AUC={metrics['ROC_AUC']:.4f}")

    return pd.DataFrame(rows), oof_pred.astype(int), oof_score, pd.DataFrame(curve_rows)


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
# Learning-curve plot (the data comes from cross_validate_model)
# --------------------------------------------------------------------------- #
def plot_learning_curves(curve_df: pd.DataFrame, out_dir: Path) -> None:
    """Error (1 - ROC-AUC) vs training rows: mean +/- std over folds.

    Large train/validation gap = overfitting; both high and flat = underfitting.
    """
    names = [n for n in ALL_MODEL_NAMES if n in set(curve_df["Model"])]
    if not names:
        return
    ncols = 4
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        g = (curve_df[curve_df["Model"] == name].groupby("Frac")
             .agg(n=("n_train", "mean"),
                  tr_m=("train_auc_err", "mean"), tr_s=("train_auc_err", "std"),
                  va_m=("val_auc_err", "mean"), va_s=("val_auc_err", "std"))
             .fillna(0))
        for m, s, label, color in [("tr_m", "tr_s", "train", "#4c78a8"), ("va_m", "va_s", "validation", "#e45756")]:
            ax.plot(g["n"], g[m], "o-", color=color, label=label)
            ax.fill_between(g["n"], g[m] - g[s], g[m] + g[s], color=color, alpha=0.15)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Training rows", fontsize=8)
        ax.set_ylabel("Error (1 - ROC-AUC)", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Learning curves (validation = held-out CV fold)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

def first_fold(X: pd.DataFrame, y: pd.Series, n_splits: int = CV_FOLDS):
    """Fold-1 split of the same CV used in the baseline: (X_fit, y_fit, X_held, y_held)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    tr, te = next(iter(cv.split(X, y)))
    return X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]


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
    curves: list[pd.DataFrame] = []

    for idx, name in enumerate(runnable_names, start=1):
        print(f"\n[{idx}/{len(runnable_names)}] {name}")
        try:
            fold_df, oof_pred, oof_score, curve_df = cross_validate_model(name, models[name], X_train, y_train, seed=SEEDS[0], with_curves=True)
            for seed in SEEDS[1:]:         # extra seeds: metrics only, no curves
                extra_df, *_ = cross_validate_model(name, models[name], X_train, y_train, seed=seed, with_curves=False)
                fold_df = pd.concat([fold_df, extra_df], ignore_index=True)
            fold_results[name] = fold_df
            oof[name] = (oof_pred, oof_score)
            curves.append(curve_df)
            print(f"  mean F1 = {fold_df['F1'].mean():.4f} ± {fold_df['F1'].std(ddof=1):.4f}")
        except Exception as exc:
            print(f"  [error] {name} failed: {exc}")

    if not fold_results:
        print("No models produced results.")
        return

    # Save everything before plotting, so a plotting error cannot lose the CV run
    numeric, pretty = summarise_cv(fold_results)
    all_curves = pd.concat(curves, ignore_index=True)
    pretty.to_csv(BASELINE_DIR / "baseline_results.csv", index=False)

    print(f"\n{'=' * 70}\nBASELINE CV RESULTS (saved to {BASELINE_DIR.resolve()})\n{'=' * 70}")
    print(pretty.to_string(index=False))

    # Each step is independent: one failing step must not lose the others
    def step(label, fn):
        try:
            fn()
        except Exception as exc:
            print(f"\n[error] {label} failed: {type(exc).__name__}: {exc}")

    step("learning-curve plot", lambda: plot_learning_curves(all_curves, BASELINE_DIR))
    step("error analysis", lambda: run_evaluation(
        X_train, y_train,
        {n: (p.to_numpy(), s.to_numpy() if s.notna().all() else None) for n, (p, s) in oof.items()},
        BASELINE_DIR))
    step("influencing factors", lambda: influencing_factors(X_train, y_train))
    step("slice checks", lambda: check_slices(X_train))
    step("robustness by condition", lambda: robustness_by_slice(
        X_train, y_train, {n: (p.to_numpy(), s.to_numpy()) for n, (p, s) in oof.items()}))
    # Shortlist: the best model of each family = highest mean CV F1
    family_of = {m: f for f, members in FAMILIES.items() for m in members}
    shortlist = (numeric.assign(Family=numeric["Model"].map(family_of),
                                Interpretability=numeric["Model"].map(INTERPRETABILITY))
                 .sort_values("F1_mean", ascending=False).groupby("Family").head(1))
    print("\nSHORTLIST: best model of each family")
    print(shortlist[["Family", "Model", "F1_mean", "F1_std", "FPR_mean",
                     "Mean_Fold_Time_Sec", "Interpretability"]].round(4).to_string(index=False))
    print(f"\nBaseline outputs saved to {BASELINE_DIR.resolve()}")


###############################################################################
# 4. EVALUATION (error analysis on out-of-fold predictions)
###############################################################################

# --------------------------------------------------------------------------- #
# Error analysis on out-of-fold predictions
# --------------------------------------------------------------------------- #
def error_analysis(preds: dict, y_true, out_dir: Path) -> pd.DataFrame:
    """Confusion matrices + ROC / PR / DET (detection error tradeoff) curves + FP/FN table for all models.

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


def influencing_factors(X_train, y_train, top_k: int = 10) -> None:
    """Which features matter most? Printed only (no files).
    Fits on the training rows of CV fold 1 and measures on that fold's held-out rows."""
    X_fit, y_fit, X_held, y_held = first_fold(X_train, y_train)

    # Random Forest: permutation importance = how much F1 drops when ONE feature is shuffled
    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=CPU_JOBS).fit(X_fit, y_fit)
    perm = permutation_importance(rf, X_held, y_held, scoring="f1", n_repeats=5,
                                  random_state=RANDOM_SEED, n_jobs=CPU_JOBS)
    rf_top = pd.Series(perm.importances_mean, index=X_train.columns).nlargest(top_k)
    print(f"\nINFLUENCING FACTORS: Random Forest, F1 drop when the feature is shuffled (top {top_k})")
    print(rf_top.round(4).to_string())

    # Logistic Regression: coefficient on standardised features (sign = direction towards phishing)
    lr = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_SEED, max_iter=1000))
    lr.fit(X_fit, y_fit)
    coef = pd.Series(lr[-1].coef_[0], index=X_train.columns)
    print(f"\nINFLUENCING FACTORS: Logistic Regression, standardised coefficient (top {top_k} by size)")
    print(coef.reindex(coef.abs().nlargest(top_k).index).round(3).to_string())
    print("\nNote: correlated features share the credit, so one of a near-identical group can look weak.")

def check_slices(X) -> None:
    """Do the slice conditions mean what we think they mean? Printed only. Run once on X_train."""
    minus1 = (X == -1)

    # 1. 'No query params': all the *_params columns should be -1 on the SAME rows as params_length
    params_cols = [c for c in X.columns if c.endswith("_params") or c == "params_length"]
    differ = {c: int((minus1[c] != minus1["params_length"]).sum()) for c in params_cols}
    differ = {c: n for c, n in differ.items() if n > 0}
    print(f"1. params columns checked: {len(params_cols)}; "
          f"{'all have -1 on exactly the same rows as params_length' if not differ else 'these differ from params_length (rows): ' + str(differ)}")
    print(f"   rows with params_length == -1: {minus1['params_length'].mean():.1%}")

    # 2. Independent check: whole-URL feature qty_questionmark_url ('?' count) should agree
    has_q = X["qty_questionmark_url"].gt(0).map({True: "'?' in URL", False: "no '?' in URL"})
    has_params = minus1["params_length"].map({True: "params_length == -1", False: "params_length != -1"})
    print("\n2. Row shares: params_length vs '?' in the whole URL (expect two big cells on the diagonal)")
    print(pd.crosstab(has_params, has_q, normalize="index").round(3).to_string())

    # 3. length_url should never be -1 or 0 (a URL always has characters)
    print(f"\n3. length_url: min = {X['length_url'].min()}, median = {X['length_url'].median()}, "
          f"rows <= 0: {int((X['length_url'] <= 0).sum())}")

    # 4. What else is -1 when time_domain_activation is -1? (tells us which kind of lookup failed)
    others = [c for c in ["time_domain_expiration", "time_response", "qty_ip_resolved", "asn_ip",
                          "qty_nameservers", "qty_mx_servers", "ttl_hostname"] if c in X.columns]
    rows = minus1["time_domain_activation"]
    table = pd.DataFrame({"share -1 when activation is -1": minus1.loc[rows, others].mean(),
                          "share -1 overall": minus1[others].mean()})
    print(f"\n4. time_domain_activation == -1 in {rows.mean():.1%} of rows. Other columns that are also -1:")
    print(table.round(3).to_string())


def robustness_by_slice(X_train, y_train, preds: dict) -> None:
    """Does each model stay reliable under different conditions? Out-of-fold F1 and FPR per slice.
    preds = {model: (y_pred, y_score)}, the same dictionary that run_evaluation receives. Printed only."""
    y_true = pd.Series(np.asarray(y_train), index=X_train.index)
    f1_rows, fpr_rows = {}, {}
    for label, make_mask in SLICES.items():
        try:
            mask = make_mask(X_train)
        except KeyError:
            continue                                        # this column is not in the data
        if mask.sum() < 50 or y_true[mask].nunique() < 2:
            continue                                        # too small to judge
        key = f"{label} (n={int(mask.sum()):,}, {y_true[mask].mean():.0%} phishing)"
        f1_rows[key], fpr_rows[key] = {}, {}
        for name, (y_pred, _) in preds.items():
            y_pred = pd.Series(np.asarray(y_pred), index=X_train.index)
            f1_rows[key][name] = f1_score(y_true[mask], y_pred[mask], zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1]).ravel()
            fpr_rows[key][name] = fp / (fp + tn) if (fp + tn) else float("nan")
    print("\nROBUSTNESS BY CONDITION (out-of-fold F1)")
    print(pd.DataFrame(f1_rows).T.round(3).to_string())
    print("\nROBUSTNESS BY CONDITION (out-of-fold FPR)")
    print(pd.DataFrame(fpr_rows).T.round(3).to_string())

# --------------------------------------------------------------------------- #
# Convenience wrapper (called from run_baseline)
# --------------------------------------------------------------------------- #
def run_evaluation(X_train, y_train, preds: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if preds:
        errors = error_analysis(preds, y_train, out_dir)
        # profile the lowest-FPR model that still catches most phishing (FNR <= 10%)
        candidates = errors[errors["FNR"] <= 0.10]
        pick = (candidates if not candidates.empty else errors).iloc[0]["Model"]
        misclassification_profile(pick, y_train, preds[pick][0], X_train, out_dir)

if __name__ == "__main__":
    run_baseline()