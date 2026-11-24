"""Part A evaluation: learning curves and error analysis.

Called by baseline.main() after cross-validation. Data discipline:
- Learning curves use the TRAINING split only.
- Error analysis uses the OUT-OF-FOLD (OOF) predictions from baseline CV, so every
  training row is predicted once by a model that never saw it.
- The validation and held-out test sets are never touched here.

Cross-validation itself and the metric table live in a_baseline.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split

from load import RANDOM_SEED

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