"""
b_ensemble.py - Part B Step 10: diversity analysis, 11-combination ensembling
(hard voting, soft voting, weighted blending, stacking), and saving fitted artifacts.

Run after b_tuning.py:
    python b_ensemble.py

The held-out test split is NEVER used here.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, log_loss
from sklearn.model_selection import cross_val_predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from b_experiments import all_metrics, make_cv  # noqa: E402
from b_tuning import (  # noqa: E402
    FEATURES_FILE,
    RESULTS_FILE,
    THRESHOLDS,
    get_data,
    params_from_text,
    pipeline_for,
)
from load import RANDOM_SEED  # noqa: E402


OUT_DIR = Path("reports/final")
ARTIFACT_DIR = Path("artifacts")

FAMILIES = {
    "Linear / probabilistic": ["Logistic Regression", "Naive Bayes"],
    "Distance / geometric": ["k-NN", "SVM"],
    "Tree-based": ["Random Forest", "LightGBM"],
    "Neural network": ["MLP", "TabNet"],
}


# --------------------------------------------------------------------------- #
# Reusable wrapper saved to artifacts/full_ensemble.joblib
# --------------------------------------------------------------------------- #
class EnsembleModel:
    """Unified wrapper for hard voting, soft voting, weighted blending, and stacking."""

    def __init__(
        self,
        method: str,
        base_names: list[str],
        base_models: list,
        feature_names: list[str],
        threshold: float = 0.5,
        weights: np.ndarray | None = None,
        meta_model=None,
    ):
        self.method = method
        self.base_names = base_names
        self.base_models = base_models
        self.feature_names = feature_names
        self.threshold = float(threshold)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.meta_model = meta_model

    def _base_matrix(self, X) -> np.ndarray:
        cols = [m.predict_proba(X)[:, 1] for m in self.base_models]
        return np.column_stack(cols)

    def individual_probas(self, X) -> dict[str, np.ndarray]:
        """Return each base model's phishing probability (used by the Streamlit GUI)."""
        return {name: m.predict_proba(X)[:, 1] for name, m in zip(self.base_names, self.base_models)}

    def predict_proba(self, X) -> np.ndarray:
        P = self._base_matrix(X)
        if self.method == "hard":
            p = (P >= 0.5).mean(axis=1)
        elif self.method == "soft":
            p = P.mean(axis=1)
        elif self.method == "weighted":
            p = P @ self.weights
        elif self.method == "stacking":
            p = self.meta_model.predict_proba(P)[:, 1]
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def prob_pipeline(name: str, params: dict | None = None):
    """Build a tuned pipeline guaranteed to support predict_proba (calibrates SVM)."""
    pipe = pipeline_for(name, params)
    if not hasattr(pipe, "predict_proba"):
        return CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
    return pipe


def load_shortlist() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read tuning_results.csv and pick the best tuned model per family by CV F1."""
    if not RESULTS_FILE.exists():
        sys.exit(f"{RESULTS_FILE} not found. Run b_tuning.py first.")
    df = pd.read_csv(RESULTS_FILE)
    family_of = {m: f for f, members in FAMILIES.items() for m in members}
    df["Family"] = df["Model"].map(family_of)
    df = df.sort_values("Tuned_F1", ascending=False).reset_index(drop=True)

    shortlist = df.groupby("Family", as_index=False).first()
    shortlist = shortlist.sort_values("Tuned_F1", ascending=False).reset_index(drop=True)
    if len(shortlist) < 2:
        sys.exit("Need at least 2 tuned model families in tuning_results.csv to build ensembles.")
    return df, shortlist


def fit_blend_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find non-negative weights summing to 1 that minimize log-loss on probability matrix P."""
    k = P.shape[1]

    def loss(w):
        pred = np.clip(P @ w, 1e-7, 1.0 - 1e-7)
        return log_loss(y, pred)

    res = minimize(
        loss,
        x0=np.full(k, 1.0 / k),
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
    )
    return res.x if res.success else np.full(k, 1.0 / k)


def tune_threshold(y_true, proba: np.ndarray) -> float:
    """Pick the probability threshold in THRESHOLDS that maximizes F1 on OOF predictions."""
    scores = {float(t): all_metrics(y_true, proba, float(t))["F1"] for t in THRESHOLDS}
    return max(scores, key=scores.get)


# --------------------------------------------------------------------------- #
# Diversity analysis (error correlation + Cohen's Kappa)
# --------------------------------------------------------------------------- #
def diversity_analysis(oof_df: pd.DataFrame, y_true: pd.Series, out_dir: Path) -> None:
    names = list(oof_df.columns)
    y_arr = np.asarray(y_true)
    preds = (oof_df >= 0.5).astype(int)
    errors = preds.ne(y_arr, axis=0).astype(int)

    err_corr = errors.corr().fillna(0.0)
    kappa = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            kappa.loc[a, b] = cohen_kappa_score(preds[a], preds[b])

    print("\nDIVERSITY CHECK: Error correlation (lower = more complementary mistakes)")
    print(err_corr.round(4).to_string())
    print("\nDIVERSITY CHECK: Cohen's Kappa agreement (lower = higher diversity)")
    print(kappa.round(4).to_string())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, mat, title, vmin in [
        (ax1, err_corr, "Error correlation (lower = more diverse)", 0.0),
        (ax2, kappa, "Cohen's Kappa (lower = more diverse)", 0.5),
    ]:
        im = ax.imshow(mat.to_numpy(), cmap="Blues", vmin=vmin, vmax=1.0)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{mat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved diversity plot -> {out_dir / 'diversity_heatmap.png'}")


# --------------------------------------------------------------------------- #
# Evaluate all combinations x 4 strategies using leak-free OOF predictions
# --------------------------------------------------------------------------- #
def evaluate_combinations(oof_df: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict]:
    names = list(oof_df.columns)
    y_arr = np.asarray(y_train)
    splits = list(make_cv().split(oof_df, y_arr))

    rows = []
    oof_store: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    combos = [c for r in range(2, len(names) + 1) for c in itertools.combinations(names, r)]
    print(f"\nEvaluating {len(combos)} combinations x 4 methods ({len(combos) * 4} ensembles) on OOF predictions...")

    for combo in combos:
        P = oof_df[list(combo)].to_numpy()
        label = " + ".join(combo)

        # 1. Hard voting (majority class; tie >= 0.5 goes to phishing)
        hard_score = (P >= 0.5).mean(axis=1)
        m_hard = all_metrics(y_arr, hard_score, threshold=0.5)
        rows.append({"Method": "hard", "Size": len(combo), "Models": label, **m_hard})
        oof_store[("hard", combo)] = hard_score

        # 2. Soft voting (equal probability average)
        soft_score = P.mean(axis=1)
        m_soft = all_metrics(y_arr, soft_score, threshold=0.5)
        rows.append({"Method": "soft", "Size": len(combo), "Models": label, **m_soft})
        oof_store[("soft", combo)] = soft_score

        # 3. Weighted blending (weights learned inside each CV fold -> zero leakage)
        blend_oof = np.zeros(len(y_arr), dtype=float)
        for tr, te in splits:
            w = fit_blend_weights(P[tr], y_arr[tr])
            blend_oof[te] = P[te] @ w
        m_blend = all_metrics(y_arr, blend_oof, threshold=0.5)
        rows.append({"Method": "weighted", "Size": len(combo), "Models": label, **m_blend})
        oof_store[("weighted", combo)] = blend_oof

        # 4. Stacking (L2 Logistic Regression meta-learner via OOF CV -> zero leakage)
        meta = LogisticRegression(random_state=RANDOM_SEED)
        stack_oof = cross_val_predict(meta, P, y_arr, cv=make_cv(), method="predict_proba")[:, 1]
        m_stack = all_metrics(y_arr, stack_oof, threshold=0.5)
        rows.append({"Method": "stacking", "Size": len(combo), "Models": label, **m_stack})
        oof_store[("stacking", combo)] = stack_oof

    results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return results, oof_store


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_ensemble() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val = get_data()
    feature_names = list(X_train.columns)
    all_tuned, shortlist = load_shortlist()

    print("SHORTLISTED FAMILY WINNERS (from tuning_results.csv):")
    print(shortlist[["Family", "Model", "Tuned_F1", "Tuned_FPR", "Threshold"]].round(4).to_string(index=False))

    # 1. Generate 5-fold OOF probabilities for the shortlisted models
    oof_dict = {}
    params_by_name = {}
    for _, row in shortlist.iterrows():
        name = row["Model"]
        params = params_from_text(row["Best_params"])
        params_by_name[name] = params
        print(f"  computing 5-fold OOF probabilities for {name} ...")
        pipe = prob_pipeline(name, params)
        oof_dict[name] = cross_val_predict(pipe, X_train, y_train, cv=make_cv(), method="predict_proba")[:, 1]

    oof_df = pd.DataFrame(oof_dict, index=X_train.index)
    diversity_analysis(oof_df, y_train, OUT_DIR)

    # 2. Evaluate all 11 combinations x 4 strategies
    results, oof_store = evaluate_combinations(oof_df, y_train)
    results.to_csv(OUT_DIR / "ensemble_results.csv", index=False)

    print(f"\nTOP 10 ENSEMBLES (saved full table -> {OUT_DIR / 'ensemble_results.csv'})")
    print(results.head(10).round(4).to_string(index=False))

    # 3. Pick winning ensemble & tune decision threshold on OOF predictions
    winner = results.iloc[0]
    win_method = winner["Method"]
    win_combo = tuple(winner["Models"].split(" + "))
    win_oof = oof_store[(win_method, win_combo)]
    win_threshold = 0.5 if win_method == "hard" else tune_threshold(y_train, win_oof)
    print(f"\nWINNING ENSEMBLE: {win_method.upper()} ({' + '.join(win_combo)}) | threshold={win_threshold}")

    # 4. Fit winning ensemble base models on full X_train
    print("Fitting final base models on full X_train ...")
    fitted_bases = [prob_pipeline(n, params_by_name[n]).fit(X_train, y_train) for n in win_combo]

    P_win_oof = oof_df[list(win_combo)].to_numpy()
    weights = fit_blend_weights(P_win_oof, np.asarray(y_train)) if win_method == "weighted" else None
    meta_model = (
        LogisticRegression(random_state=RANDOM_SEED).fit(P_win_oof, y_train)
        if win_method == "stacking"
        else None
    )
    if weights is not None:
        print("  learned blend weights:", {n: round(float(w), 4) for n, w in zip(win_combo, weights)})

    ensemble = EnsembleModel(
        method=win_method,
        base_names=list(win_combo),
        base_models=fitted_bases,
        feature_names=feature_names,
        threshold=win_threshold,
        weights=weights,
        meta_model=meta_model,
    )

    # 5. One validation check on X_val
    val_proba = ensemble.predict_proba(X_val)[:, 1]
    val_metrics = all_metrics(y_val, val_proba, threshold=win_threshold)
    print("Validation check (winning ensemble):", {k: round(v, 4) for k, v in val_metrics.items()})

    # 6. Also fit & save Baseline (default) and Best Tuned Single model for Step 11 test comparison
    best_row = all_tuned.iloc[0]
    best_single_name = best_row["Model"]
    best_single_params = params_from_text(best_row["Best_params"])
    best_single_thresh = 0.5 if pd.isna(best_row["Threshold"]) else float(best_row["Threshold"])

    print(f"Fitting baseline & tuned single models ({best_single_name}) on full X_train ...")
    baseline_model = prob_pipeline(best_single_name, params=None).fit(X_train, y_train)
    tuned_model = prob_pipeline(best_single_name, params=best_single_params).fit(X_train, y_train)

    metadata = {
        "features_file": FEATURES_FILE,
        "feature_names": feature_names,
        "best_single_model": best_single_name,
        "best_single_params": best_single_params,
        "best_single_threshold": best_single_thresh,
        "shortlist": shortlist[["Family", "Model", "Tuned_F1"]].to_dict(orient="records"),
        "ensemble_method": win_method,
        "ensemble_models": list(win_combo),
        "ensemble_threshold": win_threshold,
        "ensemble_weights": None if weights is None else [round(float(w), 4) for w in weights],
        "ensemble_val_metrics": {k: round(float(v), 4) for k, v in val_metrics.items()},
    }
    bundle = {
        "baseline": baseline_model,
        "tuned": tuned_model,
        "ensemble": ensemble,
        "meta": metadata,
    }
    joblib.dump(bundle, ARTIFACT_DIR / "models.joblib")
    print(f"\nSaved all models + metadata -> {(ARTIFACT_DIR / 'models.joblib').resolve()}")

if __name__ == "__main__":
    run_ensemble()