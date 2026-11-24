"""Learning curves to the FULL training size (including TabNet) and iteration curves.

Why this exists
- evaluation.learning_curves() uses sklearn's learning_curve with 3-fold CV, so the x-axis
  stops at 2/3 of the training rows, and k-NN/SVM stop at 10,000 because of the 15,000-row
  subsample. TabNet is skipped because its registry entry is None (not sklearn-clonable).
- Here every model is trained on growing subsets of the training split and scored on the
  VALIDATION split, so the x-axis reaches 100% of the training rows. (SVM is still capped by
  SVM_MAX_TRAIN_ROWS inside fit_predict; its x-values show the rows it really used.)
- iteration_curves() shows when the iterative models stop improving, which tells you
  whether the default iteration budget is the limit (RF trees, LightGBM trees, MLP epochs,
  TabNet epochs) and gives starting ranges for Part B tuning.

Run after a_baseline.py:   python curves.py
The held-out test split is never used.

Outputs (reports/curves/): learning_curves_full.csv/.png, iteration_curves.png, convergence_table.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from load import RANDOM_SEED, load_dataset, train_val_test
from baseline import CPU_JOBS, SVM_MAX_TRAIN_ROWS, build_models, fit_predict

OUT_DIR = Path("./reports/curves")


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df, val_size=0.2, test_size=0.2, seed=RANDOM_SEED)
    print(f"Train: {len(X_train):,} | Validation: {len(X_val):,} (test split not used)")

    models = build_models()
    for step, fn in [
        ("iteration curves", lambda: iteration_curves(X_train, y_train, X_val, y_val, OUT_DIR)),
        ("learning curves", lambda: learning_curves_full(models, X_train, y_train, X_val, y_val, OUT_DIR)),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"\n[error] {step} failed: {type(exc).__name__}: {exc}")
    print(f"\nCurve outputs saved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()