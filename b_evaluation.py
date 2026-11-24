"""
b_eval.py - Part B Steps 11-13: single locked test evaluation, robustness & stress tests,
error analysis, and interpretability (grouped permutation importance + SHAP).

Run after b_ensemble.py:
    python b_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from a_baseline import SLICES, misclassification_profile  # noqa: E402
from a_eda import EXTERNAL_FEATURES, feature_group  # noqa: E402
from b_ensemble import ARTIFACT_DIR, EnsembleModel, prob_pipeline, tune_threshold  # noqa: F401, E402
from b_experiments import all_metrics, make_cv  # noqa: E402
from load import RANDOM_SEED, load_dataset, train_val_test  # noqa: E402


OUT_DIR = Path("reports/final")
N_BOOTSTRAP = 1000
N_SHAP_SAMPLES = 500


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_artifacts():
    bundle_path = ARTIFACT_DIR / "models.joblib"
    if not bundle_path.exists():
        sys.exit(f"{bundle_path} not found. Run b_ensemble.py first.")
    bundle = joblib.load(bundle_path)
    return bundle, bundle["meta"], bundle["baseline"], bundle["tuned"], bundle["ensemble"]


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, threshold: float, n_boot: int = N_BOOTSTRAP) -> dict:
    """95% bootstrap confidence intervals for all 5 metrics."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        samples.append(all_metrics(yt, proba[idx], threshold))
    df = pd.DataFrame(samples)
    out = {}
    for col in df.columns:
        lo, hi = np.percentile(df[col], [2.5, 97.5])
        out[f"{col}_CI_low"] = round(float(lo), 4)
        out[f"{col}_CI_high"] = round(float(hi), 4)
    return out


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Compare two classifiers on the exact same test rows via their disagreements."""
    right_a = pred_a == y_true
    right_b = pred_b == y_true
    b = int(np.sum(right_a & ~right_b))     # A right, B wrong
    c = int(np.sum(~right_a & right_b))     # A wrong, B right
    n_disc = b + c
    if n_disc == 0:
        stat, p_val = 0.0, 1.0
    elif n_disc < 25:
        stat = float("nan")
        p_val = float(binomtest(min(b, c), n_disc, 0.5).pvalue)
    else:
        stat = float((abs(b - c) - 1) ** 2 / n_disc)
        p_val = float(chi2.sf(stat, df=1))
    return {
        "Comparison": f"{label_b} vs {label_a}",
        "A_only_correct": b,
        "B_only_correct": c,
        "Statistic": round(stat, 4) if not np.isnan(stat) else None,
        "p_value": round(p_val, 6),
        "Significant_0.05": p_val < 0.05,
    }


# --------------------------------------------------------------------------- #
# Single locked test evaluation
# --------------------------------------------------------------------------- #
def test_evaluation(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    print(f"\n{'=' * 70}\nSINGLE LOCKED TEST EVALUATION (n={len(X_test):,})\n{'=' * 70}")
    y_arr = np.asarray(y_test)
    rows = []
    probas, preds = {}, {}

    for label, (model, thresh) in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= thresh).astype(int)
        probas[label] = proba
        preds[label] = pred

        m = all_metrics(y_arr, proba, thresh)
        ci = bootstrap_ci(y_arr, proba, thresh)
        brier = brier_score_loss(y_arr, proba)
        row = {"Model": label, "Threshold": thresh, "Brier": round(float(brier), 5)}
        for k, v in m.items():
            row[k] = round(float(v), 4)
        row.update(ci)
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "test_results.csv", index=False)
    show_cols = ["Model", "Threshold", "F1", "F1_CI_low", "F1_CI_high", "FPR", "MCC", "ROC_AUC", "PR_AUC", "Brier"]
    print(results[show_cols].to_string(index=False))

    # McNemar's pairwise tests
    labels = list(models)
    mc_rows = [
        mcnemar_test(y_arr, preds[labels[0]], preds[labels[1]], labels[0], labels[1]),
        mcnemar_test(y_arr, preds[labels[1]], preds[labels[2]], labels[1], labels[2]),
        mcnemar_test(y_arr, preds[labels[0]], preds[labels[2]], labels[0], labels[2]),
    ]
    mc_df = pd.DataFrame(mc_rows)
    print("\nMCNEMAR'S TEST (locked test set):")
    print(mc_df.to_string(index=False))

    # Calibration curve + DET curve
    fig, (ax_cal, ax_det) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for label, proba in probas.items():
        frac_pos, mean_pred = calibration_curve(y_arr, proba, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_arr, proba)
        ax_cal.plot(mean_pred, frac_pos, "o-", label=f"{label} (Brier={brier:.4f})")

        fpr, tpr, _ = roc_curve(y_arr, proba)
        ax_det.plot(np.clip(fpr, 1e-4, 1.0), np.clip(1.0 - tpr, 1e-4, 1.0), label=label)

    ax_cal.plot([0, 1], [0, 1], "k:", lw=0.9, label="Perfect calibration")
    ax_cal.set(title="Calibration curve (test set)", xlabel="Mean predicted probability", ylabel="Fraction of phishing")
    ax_cal.legend(fontsize=8)
    ax_cal.grid(alpha=0.25)

    ax_det.set(
        title="DET curve (lower-left is better)",
        xlabel="False positive rate (log)",
        ylabel="False negative rate (log)",
        xscale="log",
        yscale="log",
    )
    ax_det.legend(fontsize=8)
    ax_det.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration_and_det.png", dpi=150)
    plt.close(fig)
    print(f"Saved calibration & DET plot -> {OUT_DIR / 'calibration_and_det.png'}")
    return {"probas": probas, "preds": preds}


# --------------------------------------------------------------------------- #
# Robustness, stress testing & error analysis
# --------------------------------------------------------------------------- #
def robustness(
    meta: dict,
    ensemble: EnsembleModel,
    X_train_full: pd.DataFrame,
    y_train: pd.Series,
    X_test_full: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    print(f"\n{'=' * 70}\nROBUSTNESS, STRESS TESTS & ERROR ANALYSIS\n{'=' * 70}")
    feat_cols = meta["feature_names"]
    X_test = X_test_full[feat_cols]
    y_arr = np.asarray(y_test)
    ens_thresh = meta["ensemble_threshold"]
    robust_rows = []

    # 12a. Full vs URL-only deployable model
    url_cols = [c for c in feat_cols if c not in EXTERNAL_FEATURES]
    ext_cols = [c for c in feat_cols if c in EXTERNAL_FEATURES]
    print(f"Feature split: {len(url_cols)} URL-only lexical features | {len(ext_cols)} external lookup features")

    full_proba = ensemble.predict_proba(X_test)[:, 1]
    robust_rows.append({"Condition": f"Full ensemble ({len(feat_cols)} features)",
                        **all_metrics(y_arr, full_proba, ens_thresh)})

    # Retrain best single model on URL-only features for instant, lookup-free GUI deployment
    best_name = meta["best_single_model"]
    best_params = meta["best_single_params"]
    print(f"Training URL-only deployable model ({best_name}) on {len(url_cols)} lexical features ...")
    url_pipe = prob_pipeline(best_name, best_params)
    url_oof = cross_val_predict(url_pipe, X_train_full[url_cols], y_train, cv=make_cv(), method="predict_proba")[:, 1]
    url_thresh = tune_threshold(y_train, url_oof)
    url_model = prob_pipeline(best_name, best_params).fit(X_train_full[url_cols], y_train)

    url_proba = url_model.predict_proba(X_test_full[url_cols])[:, 1]
    url_m = all_metrics(y_arr, url_proba, url_thresh)
    robust_rows.append({"Condition": f"URL-only {best_name} ({len(url_cols)} features)", **url_m})

    url_meta = {
        "model_name": best_name,
        "feature_names": url_cols,
        "threshold": url_thresh,
        "test_metrics": {k: round(float(v), 4) for k, v in url_m.items()},
    }

    # 12b. External-lookup timeout stress test (replace external cols with -1)
    rng = np.random.default_rng(RANDOM_SEED)
    if ext_cols:
        for rate in (0.25, 0.50, 1.00):
            X_stress = X_test.copy()
            mask = rng.random(len(X_stress)) < rate
            X_stress.loc[mask, ext_cols] = -1
            p_stress = ensemble.predict_proba(X_stress)[:, 1]
            robust_rows.append({"Condition": f"External lookup failure ({int(rate * 100)}% -> -1)",
                                **all_metrics(y_arr, p_stress, ens_thresh)})

    # 12c. Feature drift / adversarial noise stress test (jitter URL count & length columns)
    drift_cols = [c for c in url_cols if c.startswith("qty_") or "length" in c]
    for noise_std in (0.10, 0.25):
        X_noisy = X_test.copy().astype(float)
        for c in drift_cols:
            col_std = float(X_train_full[c].std())
            if col_std > 0:
                valid = X_noisy[c] != -1
                jitter = rng.normal(0.0, noise_std * col_std, size=int(valid.sum()))
                X_noisy.loc[valid, c] = np.maximum(0.0, X_noisy.loc[valid, c] + jitter)
        p_noisy = ensemble.predict_proba(X_noisy)[:, 1]
        robust_rows.append({"Condition": f"URL drift noise (sigma={noise_std:.2f} std)",
                            **all_metrics(y_arr, p_noisy, ens_thresh)})

    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(OUT_DIR / "robustness_summary.csv", index=False)
    print("\nROBUSTNESS & STRESS TEST SUMMARY (locked test set):")
    print(robust_df.round(4).to_string(index=False))

    # 12d. Subgroup slices on test set
    ens_pred = (full_proba >= ens_thresh).astype(int)
    slice_rows = []
    for label, make_mask in SLICES.items():
        try:
            mask = make_mask(X_test_full)
        except KeyError:
            continue
        if mask.sum() < 50 or y_test[mask].nunique() < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_arr[mask], ens_pred[mask], labels=[0, 1]).ravel()
        slice_rows.append({
            "Slice": label,
            "n": int(mask.sum()),
            "Phish_pct": round(float(y_arr[mask].mean()), 3),
            "F1": round(float(f1_score(y_arr[mask], ens_pred[mask], zero_division=0)), 4),
            "FPR": round(float(fp / (fp + tn)) if (fp + tn) else float("nan"), 4),
            "FNR": round(float(fn / (fn + tp)) if (fn + tp) else float("nan"), 4),
        })
    if slice_rows:
        print("\nSUBGROUP SLICE PERFORMANCE (winning ensemble on test set):")
        print(pd.DataFrame(slice_rows).to_string(index=False))

    # 12e. Misclassification profile on test set
    misclassification_profile("Winning Ensemble (Test)", y_test, ens_pred, X_test, OUT_DIR)
    return url_model, url_meta

# --------------------------------------------------------------------------- #
# Step 13: Interpretability (Grouped Permutation Importance + SHAP)
# --------------------------------------------------------------------------- #
def grouped_permutation_importance(
    ensemble: EnsembleModel, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 5
) -> None:
    """Shuffle each entire URL feature group together to measure F1 drop without collinearity bias."""
    y_arr = np.asarray(y_test)
    base_f1 = f1_score(y_arr, ensemble.predict(X_test))
    groups: dict[str, list[str]] = {}
    for col in X_test.columns:
        groups.setdefault(feature_group(col), []).append(col)

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for grp, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_test))
            X_perm[cols] = X_test[cols].to_numpy()[perm_idx]
            drops.append(base_f1 - f1_score(y_arr, ensemble.predict(X_perm)))
        rows.append({
            "Group": grp,
            "n_features": len(cols),
            "F1_drop_mean": float(np.mean(drops)),
            "F1_drop_std": float(np.std(drops, ddof=1)),
        })

    df = pd.DataFrame(rows).sort_values("F1_drop_mean", ascending=False).reset_index(drop=True)
    print("\nGROUPED PERMUTATION IMPORTANCE (test set F1 drop):")
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    order = df.iloc[::-1]
    ax.barh(order["Group"], order["F1_drop_mean"], xerr=order["F1_drop_std"], color="#4c78a8", alpha=0.85)
    ax.set(title="Grouped permutation importance (F1 drop on test set)", xlabel="Mean F1 drop when group is shuffled")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grouped_permutation_importance.png", dpi=150)
    plt.close(fig)
    print(f"Saved grouped importance plot -> {OUT_DIR / 'grouped_permutation_importance.png'}")


def _extract_tree_pipeline(ensemble: EnsembleModel, tuned_model):
    """Find the strongest fitted tree pipeline (LightGBM or Random Forest) for fast TreeExplainer."""
    for name, model in zip(ensemble.base_names, ensemble.base_models):
        if name in {"LightGBM", "Random Forest"}:
            return name, model
    return "Tuned Model", tuned_model


def shap_analysis(ensemble: EnsembleModel, tuned_model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    try:
        import shap
    except ImportError:
        print("[skip] shap not installed (`pip install shap`) -- skipping SHAP plots.")
        return

    name, pipe = _extract_tree_pipeline(ensemble, tuned_model)
    if not hasattr(pipe, "steps"):
        print(f"[skip] {name} is not a standard pipeline -- skipping TreeExplainer.")
        return

    estimator = pipe.steps[-1][1]
    sample_n = min(N_SHAP_SAMPLES, len(X_test))
    X_sub = X_test.iloc[:sample_n]
    y_sub = np.asarray(y_test.iloc[:sample_n])

    # Transform through all preprocessing steps before the final tree estimator
    X_trans = pipe[:-1].transform(X_sub) if len(pipe.steps) > 1 else X_sub.to_numpy()
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Keep feature names aligned (zero_flag adds missing-indicator columns)
    if X_trans.shape[1] == X_sub.shape[1]:
        feat_names = list(X_sub.columns)
    elif hasattr(pipe[:-1], "get_feature_names_out"):
        feat_names = list(pipe[:-1].get_feature_names_out(X_sub.columns))
    else:
        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
    X_disp = pd.DataFrame(X_trans, columns=feat_names)

    try:
        explainer = shap.TreeExplainer(estimator)
        sv = explainer(X_disp)
    except Exception as exc:
        print(f"[skip] SHAP TreeExplainer failed on {name}: {exc}")
        return

    # Select positive-class (phishing = 1) SHAP values across SHAP versions
    if sv.values.ndim == 3:
        sv_pos = sv[:, :, 1]
    else:
        sv_pos = sv

    # Global beeswarm summary plot
    plt.figure(figsize=(8.5, 5.5))
    shap.plots.beeswarm(sv_pos, max_display=15, show=False)
    plt.title(f"SHAP global feature impact ({name}, n={sample_n})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary -> {OUT_DIR / 'shap_summary.png'}")

    # Local waterfall plots: 1 correctly caught phishing (TP) + 1 error (FP or FN)
    preds = pipe.predict(X_sub)
    tp_idx = np.where((preds == 1) & (y_sub == 1))[0]
    err_idx = np.where(preds != y_sub)[0]

    for tag, idxs in [("tp_phishing", tp_idx), ("misclassified", err_idx)]:
        if len(idxs) == 0:
            continue
        i = int(idxs[0])
        plt.figure(figsize=(8.5, 4.8))
        shap.plots.waterfall(sv_pos[i], max_display=12, show=False)
        plt.title(f"SHAP local explanation: {tag} (true={y_sub[i]}, pred={preds[i]})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"shap_waterfall_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP waterfall -> {OUT_DIR / f'shap_waterfall_{tag}.png'}")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_eval() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle, meta, baseline, tuned, ensemble = load_artifacts()
    feat_cols = meta["feature_names"]

    df = load_dataset()
    X_train_full, _X_val_full, X_test_full, y_train, _y_val, y_test = train_val_test(df, seed=RANDOM_SEED)
    X_test = X_test_full[feat_cols]

    # Save a lightweight 100-row stratified test sample for the Streamlit GUI (Tab 2)
    sample_idx = (
        pd.concat([
            y_test[y_test == 0].head(50),
            y_test[y_test == 1].head(50),
        ]).index
    )
    sample_df = X_test_full.loc[sample_idx, feat_cols].copy()
    sample_df["phishing"] = y_test.loc[sample_idx].to_numpy()
    models = {
        f"Baseline ({meta['best_single_model']})": (baseline, 0.5),
        f"Tuned ({meta['best_single_model']})": (tuned, meta["best_single_threshold"]),
        f"Ensemble ({meta['ensemble_method']})": (ensemble, meta["ensemble_threshold"]),
    }

    test_evaluation(models, X_test, y_test)
    url_model, url_meta = robustness(meta, ensemble, X_train_full, y_train, X_test_full, y_test)

    bundle["url_only_model"] = url_model
    bundle["url_only_meta"] = url_meta
    bundle["test_sample"] = sample_df
    joblib.dump(bundle, ARTIFACT_DIR / "models.joblib")
    print(f"Updated single bundle -> {(ARTIFACT_DIR / 'models.joblib').resolve()}")

    print(f"\n{'=' * 70}\nINTERPRETABILITY\n{'=' * 70}")
    grouped_permutation_importance(ensemble, X_test, y_test)
    shap_analysis(ensemble, tuned, X_test, y_test)

    print(f"\nAll final evaluation outputs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    run_eval()