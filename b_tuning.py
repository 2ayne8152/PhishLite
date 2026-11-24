"""
b_tuning.py - tune the models on the final feature set, in ONE run.

    python b_tuning.py              # every model
    python b_tuning.py svm mlp      # only these (so the work can be shared between team members)

Model names: logistic_regression, naive_bayes, knn, svm, random_forest, lightgbm, mlp, tabnet

For each model it does, in order:
  1. default settings, 5-fold CV                               (the "before")
  2. a search for better settings                              (grid, random or Optuna: see SEARCHES)
  3. the winner re-checked with 5-fold CV on ALL train rows    (the "after")
  4. the decision threshold, from out-of-fold predictions
  5. ONE look at the validation split
  6. a validation curve for one key parameter                  (only the models listed in CURVES)
and then adds one row to reports/tuning/tuning_results.csv.

A model that already has a row is skipped, so a stopped run can simply be started again
(delete a model's row from the CSV to redo it).

The test split is never used here. Needs load.py and b_experiments.py in the same folder.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, cross_val_predict,
                                     cross_validate, learning_curve, train_test_split,
                                     validation_curve)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

matplotlib.use("Agg")                      # save pictures to a file instead of opening a window
import matplotlib.pyplot as plt            # noqa: E402

from b_experiments import SCORING, SENTINEL_CHOICE, all_metrics, build, cv_scores, make_cv  # noqa: E402
from load import RANDOM_SEED, load_dataset, train_val_test                                   # noqa: E402

# --------------------------------------------------------------------------- #
# SETTINGS (every team member must use exactly the same values)
# --------------------------------------------------------------------------- #
FEATURES_FILE = "reports/features_final.txt"
OUT_DIR = Path("reports/tuning")
RESULTS_FILE = OUT_DIR / "tuning_results.csv"      # one row per model
CONFIG_FILE = OUT_DIR / "config.csv"               # the configuration table for the report

# ONE tuning budget: every random or Optuna search tries at most this many settings.
# (A grid search tries every combination, so keep each grid at N_TRIALS combinations or fewer.)
N_TRIALS = 10

ORDER = ["Logistic Regression", "Naive Bayes", "k-NN", "SVM",
         "Random Forest", "LightGBM", "MLP", "TabNet"]

# What to try for each model.
#   method  "grid"   = try every combination (small spaces)
#           "random" = try N_TRIALS random combinations
#           "optuna" = try N_TRIALS combinations chosen by Optuna's TPE sampler, which learns from earlier
#                      trials which regions work well (a "smarter" search for the slower / bigger models)
#   rows    slow models are SEARCHED on a random subsample of the train rows to save time. The winner is
#           then re-checked with 5-fold CV on ALL train rows, so the final comparison stays fair.
SEARCHES = {
    "Logistic Regression": dict(method="grid", space={
        "C": [0.01, 0.1, 1, 10, 100],                  # smaller C = stronger penalty
        "class_weight": [None, "balanced"]}),          # "balanced" = care more about the rarer class
    "Naive Bayes": dict(method="grid", space={
        "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]}),
    "k-NN": dict(method="grid", space={
        "n_neighbors": [3, 5, 9, 15, 21],
        "weights": ["uniform", "distance"]}),
    "SVM": dict(method="random", rows=15000, space={
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.001, 0.01, 0.1],
        "class_weight": [None, "balanced"]}),
    "Random Forest": dict(method="random", space={
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.3],
        "class_weight": [None, "balanced"]}),
    "LightGBM": dict(method="optuna", space={
        "n_estimators": [100, 200, 400, 800],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [10, 20, 50],
        "subsample": [0.7, 0.85, 1.0], "subsample_freq": [1],      # subsample only works with freq=1
        "colsample_bytree": [0.5, 0.7, 1.0],
        "class_weight": [None, "balanced"]}),
    "MLP": dict(method="optuna", rows=20000, space={
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [0.001, 0.003],
        "early_stopping": [True], "n_iter_no_change": [10]}),      # early stopping: stop when it stops improving
    "TabNet": dict(method="optuna", rows=20000, space={
        "n_d": [8, 16, 32],                            # width of the network
        "n_steps": [3, 4, 5],
        "gamma": [1.0, 1.3, 1.5],
        "lambda_sparse": [1e-4, 1e-3, 1e-2],
        "lr": [0.01, 0.02, 0.05],
        "batch_size": [1024, 2048]}),                  # TabNet stops early inside fit() (see TabNetSklearn)
}

# (param, values, log_x, invert_x, max_rows, cv_folds)
# invert_x=True flips parameters where smaller values = higher complexity, so left -> right is always low -> high complexity.
CURVES = {
    "Logistic Regression": ("C", [0.001, 0.01, 0.1, 1, 10, 100], True, False, None, 5),
    "Naive Bayes":         ("var_smoothing", [1e-9, 1e-7, 1e-5, 1e-3, 1e-1], True, True, None, 5),
    "k-NN":                ("n_neighbors", [1, 3, 5, 9, 15, 21, 31], False, True, None, 5),
    "SVM":                 ("C", [0.01, 0.1, 1, 10, 100], True, False, 15000, 3),
    "Random Forest":       ("max_depth", [3, 5, 10, 15, 20, 30], False, False, None, 5),
    "LightGBM":            ("num_leaves", [7, 15, 31, 63, 127, 255], False, False, None, 5),
    "MLP":                 ("alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], True, True, 20000, 3),
    "TabNet":              ("n_d", [4, 8, 16, 32], False, False, 15000, 3),
}

LC_FRACTIONS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
LC_SLOW = {"SVM": (15000, 3), "MLP": (20000, 3), "TabNet": (15000, 3)}

THRESHOLDS = np.unique(np.concatenate([
    np.round(np.arange(0.05, 0.96, 0.05), 2),
    np.logspace(-6, -1, 11),
    1 - np.logspace(-6, -1, 11),
]))

# --------------------------------------------------------------------------- #
# The models. TabNet gets a small wrapper so it can be treated like the others.
# --------------------------------------------------------------------------- #
class TabNetSklearn(ClassifierMixin, BaseEstimator):     # mixin first, or sklearn will not see a classifier
    """Minimal scikit-learn wrapper around pytorch-tabnet.
    Early stopping: 10% of the training rows are held back inside fit() and training stops when the
    log-loss on them has not improved for `patience` epochs (the best epoch is kept)."""

    def __init__(self, n_d=16, n_steps=3, gamma=1.3, lambda_sparse=1e-3, lr=0.02,
                 batch_size=1024, max_epochs=100, patience=10, seed=RANDOM_SEED):
        self.n_d = n_d
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetClassifier
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        X_fit, X_stop, y_fit, y_stop = train_test_split(X, y, test_size=0.1, stratify=y,
                                                        random_state=self.seed)
        self.model_ = TabNetClassifier(n_d=self.n_d, n_a=self.n_d, n_steps=self.n_steps, gamma=self.gamma,
                                       lambda_sparse=self.lambda_sparse, optimizer_params={"lr": self.lr},
                                       seed=self.seed, verbose=0)
        self.model_.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], eval_metric=["logloss"],
                        max_epochs=self.max_epochs, patience=self.patience,
                        batch_size=self.batch_size, virtual_batch_size=128)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=np.float32))


_TOLD = set()          # so each "[skip]" message is printed only once


def make_models():
    """Fresh default models (same defaults as the baseline). LightGBM / TabNet only if installed."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Naive Bayes": GaussianNB(),
        "k-NN": KNeighborsClassifier(n_jobs=-1),
        "SVM": SVC(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "MLP": MLPClassifier(max_iter=300, random_state=RANDOM_SEED),
    }
    for module, name, make in [("lightgbm", "LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
                                   random_state=RANDOM_SEED, verbosity=-1)),
                               ("pytorch_tabnet", "TabNet", lambda: TabNetSklearn())]:
        if importlib.util.find_spec(module):
            models[name] = make()
        elif name not in _TOLD:
            _TOLD.add(name)
            print(f"[skip] {name} is not installed (module '{module}')")
    return {name: models[name] for name in ORDER if name in models}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def slug(text):
    return text.lower().replace("-", "").replace(" ", "_")


def pick_models(args):
    """Model names from the command line, or every available model if none were given."""
    available = list(make_models())
    if not args:
        return available
    lookup = {slug(n): n for n in available}
    chosen = []
    for a in args:
        if slug(a) not in lookup:
            sys.exit(f"Unknown model '{a}'. Available: {', '.join(slug(n) for n in available)}")
        chosen.append(lookup[slug(a)])
    return chosen


def get_data():
    """X_train, y_train, X_val, y_val, with only the final feature list."""
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df, seed=RANDOM_SEED)   # X_test is NOT used
    if not Path(FEATURES_FILE).exists():
        sys.exit(f"{FEATURES_FILE} not found. Run b_experiments.py reduction first.")
    columns = Path(FEATURES_FILE).read_text().split()
    return X_train[columns], y_train, X_val[columns], y_val


def with_prefix(pipeline, params):
    """Inside a pipeline a parameter is called <laststep>__<name>. Add that prefix."""
    step = pipeline.steps[-1][0]
    return {f"{step}__{k}": v for k, v in params.items()}


def param_name(pipeline, name):
    """Full pipeline name of one parameter, e.g. 'C' -> 'logisticregression__C'."""
    return f"{pipeline.steps[-1][0]}__{name}"


def pipeline_for(name, params=None):
    """Pipeline (fix -1 -> scale -> model), with the given settings if there are any."""
    pipeline = build(make_models()[name])
    if params:
        pipeline.set_params(**with_prefix(pipeline, params))
    return pipeline


def params_from_text(text):
    """The Best_params column (JSON text) -> dict. JSON turns tuples into lists, sklearn wants tuples."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(text).items()}


def get_score(pipeline, X):
    """(score, default threshold): P(phishing) and 0.5, or for the SVM its decision value and 0."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1], 0.5
    return pipeline.decision_function(X), 0.0


def save_row(path, row):
    """Add (or replace) one model's row in a CSV, so models can be run one at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[old[table.columns[0]] != row[table.columns[0]]], table], ignore_index=True)
    table.to_csv(path, index=False)


def summary(prefix, scores, seconds):
    """Mean of every metric, plus the F1 std and the mean fold time, with a column-name prefix."""
    out = {f"{prefix}_{m}": v.mean() for m, v in scores.items()}
    out[f"{prefix}_F1_std"] = scores["F1"].std(ddof=1)
    out[f"{prefix}_Time_Sec"] = round(seconds / 5, 1)
    return out


# --------------------------------------------------------------------------- #
# Step 2: the search (grid / random / Optuna all return the same two things)
# --------------------------------------------------------------------------- #
def optuna_search(name, X, y, space, n_trials):
    """Optuna (TPE): every trial is a 5-fold CV of one setting; later trials learn from earlier ones."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    labels = {k: {str(v): v for v in values} for k, values in space.items()}    # Optuna wants simple labels
    rows = []

    def objective(trial):
        params = {k: labels[k][trial.suggest_categorical(k, list(labels[k]))] for k in space}
        scores = cross_validate(pipeline_for(name, params), X, y, cv=make_cv(), scoring=SCORING)
        rows.append({"settings": ", ".join(f"{k}={v}" for k, v in params.items()), "params": params,
                     "F1": scores["test_F1"].mean(), "F1_std": scores["test_F1"].std(ddof=1),
                     "FPR": scores["test_FPR"].mean(), "MCC": scores["test_MCC"].mean()})
        return rows[-1]["F1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials)
    history = pd.DataFrame(rows)
    return history.loc[history["F1"].idxmax(), "params"], history.drop(columns="params")


def run_search(name, X, y):
    """Returns (best settings, table of everything tried, number of rows searched on)."""
    spec = SEARCHES[name]
    rows = spec.get("rows")
    if rows and rows < len(X):
        X, _, y, _ = train_test_split(X, y, train_size=rows, stratify=y, random_state=RANDOM_SEED)
        print(f"   searching on a stratified subsample of {rows:,} rows (the winner is re-checked on all rows)")

    if spec["method"] == "optuna":
        best, history = optuna_search(name, X, y, spec["space"], N_TRIALS)
    else:
        pipeline = pipeline_for(name)
        space = with_prefix(pipeline, spec["space"])
        common = dict(scoring=SCORING, refit="F1", cv=make_cv())          # the winner = best mean F1
        if spec["method"] == "grid":
            search = GridSearchCV(pipeline, space, **common)
        else:
            search = RandomizedSearchCV(pipeline, space, n_iter=N_TRIALS, random_state=RANDOM_SEED, **common)
        search.fit(X, y)
        res = pd.DataFrame(search.cv_results_)
        history = pd.DataFrame({
            "settings": [", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in p.items()) for p in res["params"]],
            "F1": res["mean_test_F1"], "F1_std": res["std_test_F1"],
            "FPR": res["mean_test_FPR"], "MCC": res["mean_test_MCC"]})
        best = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}

    print("   top 5 settings by F1:")
    print(history.sort_values("F1", ascending=False).head(5).round(4).to_string(index=False))
    return best, history, len(X)


# --------------------------------------------------------------------------- #
# Steps 4-6: threshold, validation split, validation curve
# --------------------------------------------------------------------------- #
def scan_threshold(pipeline, X, y):
    """Best decision threshold by F1, from out-of-fold probabilities. None for the SVM (no probabilities)."""
    if not hasattr(pipeline, "predict_proba"):
        print("   no probabilities (SVM): the threshold stays at the default")
        return None
    proba = cross_val_predict(pipeline, X, y, cv=make_cv(), method="predict_proba")[:, 1]
    table = pd.DataFrame([{"threshold": t, **all_metrics(y, proba, t)} for t in THRESHOLDS]).set_index("threshold")
    best = float(table["F1"].idxmax())
    print(table.loc[sorted({0.3, 0.4, 0.5, 0.6, 0.7, best}), ["F1", "FPR", "MCC"]].round(4).to_string())
    print(f"   best F1 at threshold {best} (the default is 0.5)")
    return best


def validation_check(pipeline, X, y, X_val, y_val, threshold):
    """Fit on TRAIN only, score the validation split once. Look, do not tune on it."""
    pipeline.fit(X, y)
    score, default_t = get_score(pipeline, X_val)
    out = {f"Val_{m}": v for m, v in all_metrics(y_val, score, default_t).items()}
    if threshold is not None:
        tuned = all_metrics(y_val, score, threshold)
        out.update({f"Val_{m}_at_tuned_threshold": tuned[m] for m in ("F1", "FPR", "MCC")})
    print(f"   validation: F1 = {out['Val_F1']:.4f} | FPR = {out['Val_FPR']:.4f} | MCC = {out['Val_MCC']:.4f}")
    return out


def _subsample(X, y, max_rows):
    if max_rows and max_rows < len(X):
        X_sub, _, y_sub, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=RANDOM_SEED)
        return X_sub, y_sub
    return X, y


def plot_bias_variance_grid(X, y):
    """Part B Curve 1: Error (1 - F1) vs. model complexity across all 8 models (2x4 grid)."""
    from sklearn.model_selection import StratifiedKFold

    available = [m for m in ORDER if m in make_models() and m in CURVES]
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.6), squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, available):
        param, values, log_x, invert_x, max_rows, n_splits = CURVES[name]
        print(f"[bias-variance] {name}: sweeping {param} ...")
        X_use, y_use = _subsample(X, y, max_rows)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

        base_params = {"early_stopping": True, "n_iter_no_change": 10} if name == "MLP" else None
        pipe = pipeline_for(name, base_params)

        tr_scores, va_scores = validation_curve(
            pipe, X_use, y_use,
            param_name=param_name(pipe, param),
            param_range=values,
            cv=cv, scoring="f1", n_jobs=1,
        )
        tr_err, va_err = 1.0 - tr_scores, 1.0 - va_scores

        for err, label, color in [(tr_err, "train", "#4c78a8"), (va_err, "validation", "#e45756")]:
            m, s = err.mean(axis=1), err.std(axis=1)
            ax.plot(values, m, "o-", color=color, label=label)
            ax.fill_between(values, m - s, m + s, color=color, alpha=0.15)

        # Mark the optimal complexity point (minimum mean validation error)
        va_mean = va_err.mean(axis=1)
        best_idx = int(np.argmin(va_mean))
        best_x, best_y = values[best_idx], float(va_mean[best_idx])

        ax.axvline(best_x, color="#2ca02c", ls="--", lw=1.2, alpha=0.85, label="optimal")
        ax.scatter([best_x], [best_y], color="#2ca02c", s=65, zorder=5, edgecolor="white", linewidth=0.9)

        if log_x:
            ax.set_xscale("log")
        if invert_x:
            ax.invert_xaxis()

        note = f" (n={max_rows // 1000}k)" if max_rows else ""
        ax.set_title(f"{name}{note}\nopt: {param}={best_x:g} (err={best_y:.4f})", fontsize=9)
        ax.set_xlabel(f"{param} (→ higher complexity)", fontsize=8)
        ax.set_ylabel("Error (1 - F1)", fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(available):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Part B — Bias-Variance Curves (Error vs. Model Complexity, 30 Features)", fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bias_variance_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path.resolve()}")


def plot_tuned_learning_grid(X, y):
    """Part B Curve 2: Error (1 - F1) vs. training rows for all 8 TUNED models (2x4 grid)."""
    from sklearn.model_selection import StratifiedKFold

    if not RESULTS_FILE.exists():
        sys.exit(f"{RESULTS_FILE} not found. Run b_tuning.py first.")
    res_df = pd.read_csv(RESULTS_FILE).set_index("Model")
    available = [m for m in ORDER if m in make_models() and m in res_df.index]

    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.6), squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, available):
        max_rows, n_splits = LC_SLOW.get(name, (None, 5))
        print(f"[learning-curve] {name} (tuned) ...")
        X_use, y_use = _subsample(X, y, max_rows)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

        best_params = params_from_text(res_df.loc[name, "Best_params"])
        pipe = pipeline_for(name, best_params)

        sizes, tr_scores, va_scores = learning_curve(
            pipe, X_use, y_use,
            train_sizes=LC_FRACTIONS,
            cv=cv, scoring="f1", n_jobs=1,
            shuffle=True, random_state=RANDOM_SEED,
        )
        tr_err, va_err = 1.0 - tr_scores, 1.0 - va_scores

        for err, label, color in [(tr_err, "train", "#4c78a8"), (va_err, "validation", "#e45756")]:
            m, s = err.mean(axis=1), err.std(axis=1)
            ax.plot(sizes, m, "o-", color=color, label=label)
            ax.fill_between(sizes, m - s, m + s, color=color, alpha=0.15)

        # Mark the lowest validation error point
        va_mean = va_err.mean(axis=1)
        best_idx = int(np.argmin(va_mean))
        best_n, best_y = int(sizes[best_idx]), float(va_mean[best_idx])

        ax.axvline(best_n, color="#2ca02c", ls="--", lw=1.2, alpha=0.85, label="optimal")
        ax.scatter([best_n], [best_y], color="#2ca02c", s=65, zorder=5, edgecolor="white", linewidth=0.9)

        note = f" (n={max_rows // 1000}k)" if max_rows else ""
        ax.set_title(f"{name} (tuned){note}\nmin val err={best_y:.4f} @ n={best_n:,}", fontsize=9)
        ax.set_xlabel("Training rows", fontsize=8)
        ax.set_ylabel("Error (1 - F1)", fontsize=8)
        ax.grid(alpha=0.25)

    for ax in axes[len(available):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Part B — Learning Curves of Tuned Models (30 Features)", fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "tuned_learning_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path.resolve()}")

# --------------------------------------------------------------------------- #
# One model, start to finish
# --------------------------------------------------------------------------- #
def tune_one(name, X, y, X_val, y_val):
    method = SEARCHES[name]["method"]
    print(f"\n{'=' * 70}\n{name}  ({method} search, {N_TRIALS} settings at most)\n{'=' * 70}")

    print("1/5 default settings, 5-fold CV ...")
    start = time.time()
    default = cv_scores(pipeline_for(name), X, y)
    default_seconds = time.time() - start

    print("2/5 search ...")
    best, history, rows_used = run_search(name, X, y)
    print(f"   best settings: {best}")

    print("3/5 winner re-checked with 5-fold CV on all train rows ...")
    start = time.time()
    tuned = cv_scores(pipeline_for(name, best), X, y)
    tuned_seconds = time.time() - start
    gain = tuned["F1"].mean() - default["F1"].mean()
    print(f"   F1 {default['F1'].mean():.4f} -> {tuned['F1'].mean():.4f}  (gain {gain:+.4f}, "
          f"fold std {tuned['F1'].std(ddof=1):.4f})")

    print("4/5 decision threshold ...")
    threshold = scan_threshold(pipeline_for(name, best), X, y)

    print("5/5 validation split ...")
    val = validation_check(pipeline_for(name, best), X, y, X_val, y_val, threshold)
    plot_bias_variance_grid(X, y)
    plot_tuned_learning_grid(X, y)

    row = {"Model": name, "Method": method, "Trials": len(history), "Rows_searched": rows_used,
           "Best_params": json.dumps(best, default=str), "Threshold": threshold}
    row.update(summary("Default", default, default_seconds))
    row.update(summary("Tuned", tuned, tuned_seconds))
    row.update(val)
    save_row(RESULTS_FILE, row)
    print(f"   saved -> {RESULTS_FILE}")


# --------------------------------------------------------------------------- #
# The configuration table
# --------------------------------------------------------------------------- #
def print_config(names, X):
    rows = [("features_file", FEATURES_FILE), ("n_features", X.shape[1]), ("train_rows", len(X)),
            ("sentinel_choice", SENTINEL_CHOICE), ("scaler", "StandardScaler"),
            ("cv", f"StratifiedKFold(5, shuffle=True, seed={RANDOM_SEED})"), ("seed", RANDOM_SEED),
            ("trials_per_search", N_TRIALS), ("scikit-learn", sklearn.__version__)]
    for name in names:
        spec = SEARCHES[name]
        size = int(np.prod([len(v) for v in spec["space"].values()]))
        rows.append((name, f"{spec['method']} | rows={spec.get('rows', 'all')} | space size={size} | "
                           f"{json.dumps(spec['space'], default=str)}"))
        if spec["method"] == "grid" and size > N_TRIALS:
            print(f"[warn] {name}: the grid has {size} combinations, more than N_TRIALS = {N_TRIALS}")
    config = pd.DataFrame(rows, columns=["setting", "value"])
    print("CONFIGURATION (the same for everyone)")
    print(config.to_string(index=False, max_colwidth=90))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():                                    # keep the lines of models run earlier
        old = pd.read_csv(CONFIG_FILE)
        config = pd.concat([old[~old["setting"].isin(config["setting"])], config], ignore_index=True)
    config.to_csv(CONFIG_FILE, index=False)


# --------------------------------------------------------------------------- #
def main(args):
    X, y, X_val, y_val = get_data()
    if "--curves" in args:
        plot_bias_variance_grid(X, y)
        plot_tuned_learning_grid(X, y)
        return
    names = pick_models(args)
    print_config(names, X)
    done = set(pd.read_csv(RESULTS_FILE)["Model"]) if RESULTS_FILE.exists() else set()
    for name in names:
        if name in done:
            print(f"\n[skip] {name} already has a row in {RESULTS_FILE.name}")
            continue
        try:                                                    # one failing model must not stop the others
            tune_one(name, X, y, X_val, y_val)
        except Exception as exc:
            print(f"\n[error] {name} failed: {type(exc).__name__}: {exc}")
    print(f"\nDone. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main(sys.argv[1:])