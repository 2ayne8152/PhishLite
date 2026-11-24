import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# skip the convergence warning

"""
b_experiments.py - small experiments on the phishing data. Run ONE at a time:

    python b_experiments.py sentinel      # 1. what should we do with the -1 values?
    python b_experiments.py scaling       # 2. which scaler works best?
    python b_experiments.py collinearity  # 3. how bad is the collinearity, and does it matter?
                                          #    (also saves reports/features_cluster.txt)
    python b_experiments.py reduction     # 4. how many features do we really need?
                                          #    (saves ONE majority-vote list: reports/features_final.txt)

Needs load.py in the same folder (it loads the data and makes the train/validation/test split).
Every result comes from 5-fold cross-validation on the TRAIN split only.
The validation and test splits are never touched here.

Order: run 1 -> look at the table -> set SENTINEL_CHOICE below -> run 2 -> run 3 -> run 4.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, make_scorer,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, RobustScaler, StandardScaler

from load import RANDOM_SEED, load_dataset, train_val_test

# --------------------------------------------------------------------------- #
# SETTINGS
# Decide to keep, zero_flag, or replace the '-1' with median of each column.
# --------------------------------------------------------------------------- #
# After running step 1, pick the winner here: "keep", "zero_flag" or "median"
SENTINEL_CHOICE = "keep"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def get_train():
    """Load the data and return ONLY the training part (X_train, y_train)."""
    df = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(df, seed=RANDOM_SEED)
    print(f"Training rows: {len(X_train):,} | features: {X_train.shape[1]}")
    return X_train, y_train


def false_positive_rate(y_true, y_pred):
    """FPR = legit sites wrongly flagged as phishing / all legit sites."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn)


# The five metrics we agreed on
SCORING = {
    "F1": "f1",
    "FPR": make_scorer(false_positive_rate),
    "MCC": "matthews_corrcoef",
    "ROC_AUC": "roc_auc",
    "PR_AUC": "average_precision",
}


def sentinel_step(choice):
    """Return the pipeline step that handles the -1 values.

    keep      : leave -1 as it is
    zero_flag : turn -1 into 0 AND add a 0/1 column "this part was missing" for each feature with -1
    median    : turn -1 into the median of that feature (no flag column)
    """
    if choice == "keep":
        return "passthrough"
    if choice == "zero_flag":
        return SimpleImputer(missing_values=-1, strategy="constant", fill_value=0, add_indicator=True)
    if choice == "median":
        return SimpleImputer(missing_values=-1, strategy="median")
    raise ValueError(f"Unknown sentinel choice: {choice}")


def build(model, sentinel=None, scaler=None):
    """Chain: fix -1  ->  scale  ->  model.
    Because they are in ONE pipeline, the medians / means are learned from the training
    folds only, so nothing leaks from the validation fold."""
    sentinel = SENTINEL_CHOICE if sentinel is None else sentinel
    scaler = StandardScaler() if scaler is None else scaler
    return make_pipeline(sentinel_step(sentinel), scaler, model)

def make_cv():
    """The one cross-validation step used everywhere: 5 folds, same seed (42). """
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def cv_scores(pipeline, X, y):
    """5-fold stratified CV. Returns {metric: array of 5 fold scores}."""
    result = cross_validate(pipeline, X, y, cv=make_cv(), scoring=SCORING)
    return {m: result[f"test_{m}"] for m in SCORING}

def all_metrics(y_true, proba, threshold=0.5):
    """The five metrics from predicted probabilities. A site is called phishing if proba >= threshold"""
    pred = (proba >= threshold).astype(int)
    return {
        "F1": f1_score(y_true, pred),
        "FPR": false_positive_rate(y_true, pred),
        "MCC": matthews_corrcoef(y_true, pred),
        "ROC_AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
    }

def show(results):
    """results = {(model, setting): scores}. Print mean of each metric (+ std of F1)."""
    rows = []
    for (model, setting), scores in results.items():
        row = {"model": model, "setting": setting}
        for metric, values in scores.items():
            row[metric] = round(values.mean(), 4)
        row["F1_std"] = round(scores["F1"].std(ddof=1), 4)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


MODELS_TO_RUN = ["Logistic Regression", "k-NN", "Random Forest", "MLP"]
def models():
    """Fresh copies of the four models (one per family). Defaults, no tuning.
       Max_iter is added just to prevent memory out-of-bound errors."""
    everything = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "k-NN": KNeighborsClassifier(n_jobs=-1),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
        "MLP": MLPClassifier(max_iter=300, random_state=RANDOM_SEED),
    }
    return {name: model for name, model in everything.items() if name in MODELS_TO_RUN}


# --------------------------------------------------------------------------- #
# Experiment 1: the -1 values
# --------------------------------------------------------------------------- #
def run_sentinel():
    X, y = get_train()

    # Look first: how common is -1?
    minus_one = (X == -1)
    per_column = minus_one.sum()
    per_column = per_column[per_column > 0].sort_values(ascending=False)
    print(f"\n{len(per_column)} of {X.shape[1]} features contain -1 "
          f"({minus_one.to_numpy().mean():.2%} of all cells).")
    print("Top 5 features by number of -1:")
    print(per_column.head(5).to_string())

    # Compare the three treatments
    results = {}
    for model_name in models():
        for choice in ["keep", "zero_flag", "median"]:
            print(f"  running {model_name} / {choice} ...")
            pipeline = build(models()[model_name], sentinel=choice)
            results[(model_name, choice)] = cv_scores(pipeline, X, y)

    print("\nSENTINEL TREATMENT (mean over 5 folds)")
    show(results)
    print("\nHow to read it: a treatment only 'wins' if its F1 is higher than the others by MORE "
          "than the F1_std. Smaller gaps are noise. If they tie, prefer the lowest FPR, then the simplest.")


# --------------------------------------------------------------------------- #
# Experiment 2: scaling
# --------------------------------------------------------------------------- #
def log_plus_one(X):
    """log(1 + x) squeezes big counts. Negative values (should be none after step 1) become 0."""
    return np.log1p(np.clip(X, 0, None))


def run_scaling():
    X, y = get_train()

    scalers = {
        "none": "passthrough",
        "standard (z-score)": StandardScaler(),
        "robust (median/IQR)": RobustScaler(),
        "quantile -> normal": QuantileTransformer(output_distribution="normal", random_state=RANDOM_SEED),
    }
    if SENTINEL_CHOICE != "keep":     # log1p would turn -1 into 0 and hide the sentinel
        scalers["log1p + standard"] = make_pipeline(FunctionTransformer(log_plus_one), StandardScaler())

    # Only models that care about feature size. Trees split on ORDER, so scaling cannot change them.
    results = {}
    for model_name in [m for m in models() if m != "Random Forest"]:  # trees ignore scaling:
        for scaler_name, scaler in scalers.items():
            print(f"  running {model_name} / {scaler_name} ...")
            pipeline = build(models()[model_name], scaler=scaler)
            results[(model_name, scaler_name)] = cv_scores(pipeline, X, y)

    print(f"\nSCALING (sentinel treatment = {SENTINEL_CHOICE}; mean over 5 folds)")
    show(results)


# --------------------------------------------------------------------------- #
# Experiment 3: collinearity
# --------------------------------------------------------------------------- #
def diagnose(X):
    """Three numbers that describe collinearity, on the data X you pass in."""
    # 1. How many feature pairs are almost the same? (Spearman |rho| >= 0.9)
    corr = X.corr(method="spearman").abs().to_numpy()
    n_pairs = int((np.triu(corr, k=1) >= 0.9).sum())

    # 2. VIF: how well can each feature be predicted from all the others? (>10 is a common red flag)
    #    VIF_j is the j-th diagonal value of the inverse correlation matrix.
    pearson = np.corrcoef(X.to_numpy(), rowvar=False)
    vif = np.diag(np.linalg.pinv(pearson))

    # 3. Condition number of the standardised data (>30 = worrying, >1000 = severe)
    Xs = StandardScaler().fit_transform(X)
    cond = np.linalg.cond(Xs)

    return {"features": X.shape[1], "pairs_rho>=0.9": n_pairs,
            "features_VIF>10": int((vif > 10).sum()), "max_VIF": round(float(vif.max()), 1),
            "condition_number": round(float(cond), 1)}, pd.Series(vif, index=X.columns)

def treated_copy(X):
    """Copy of X where -1 is handled like SENTINEL_CHOICE (only used for correlation work,
    otherwise -1 distorts the correlations)."""
    Xd = X.copy()
    if SENTINEL_CHOICE == "zero_flag":
        Xd = Xd.replace(-1, 0)
    elif SENTINEL_CHOICE == "median":
        Xd = Xd.replace(-1, np.nan)
        Xd = Xd.fillna(Xd.median())
    return Xd


def one_per_cluster(Xd, show=True):
    """Group features whose Spearman |rho| >= 0.9 and keep ONE per group.
    No labels are used here, so nothing can leak from y."""
    corr = Xd.corr(method="spearman").abs()
    distance = (1 - corr).clip(lower=0).to_numpy()        # distance = 1 - |rho|
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    cluster_id = pd.Series(fcluster(tree, t=0.1, criterion="distance"), index=Xd.columns)

    keep = []
    if show:
        print("\nCLUSTERS of near-identical features (we keep ONE per cluster, marked with *):")
    for cid, members in cluster_id.groupby(cluster_id):
        members = members.index.tolist()
        if len(members) == 1:
            keep.append(members[0])
            continue
        # keep the "most typical" member: highest average |rho| with the others in its cluster
        typical = corr.loc[members, members].mean().idxmax()
        keep.append(typical)
        if show:
            print("  " + ", ".join(f"{m}*" if m == typical else m for m in members))
    return keep


def run_collinearity():
    X, y = get_train()
    Xd = treated_copy(X)

    # Constant columns have no variation: they break VIF and correlation, so drop them first.
    constant = Xd.columns[Xd.nunique() <= 1].tolist()
    print(f"Constant columns dropped ({len(constant)}): {constant}")
    Xd = Xd.drop(columns=constant)
    X = X.drop(columns=constant)

    keep = one_per_cluster(Xd)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/features_cluster.txt").write_text("\n".join(keep))   # fallback list for tuning

    # How bad is it? Before vs after dropping the duplicates
    before, vif_before = diagnose(Xd)
    after, _ = diagnose(Xd[keep])
    print("\nCOLLINEARITY DIAGNOSTICS (training data only)")
    print(pd.DataFrame({"all features": before, "one per cluster": after}).to_string())
    print("\nTop 10 VIF (all features):")
    print(vif_before.sort_values(ascending=False).head(10).round(1).to_string())

    # Does it hurt each model family? Same CV, all features vs one per cluster.
    results = {}
    for model_name in models():
        for label, cols in [("all features", list(X.columns)), ("one per cluster", keep)]:
            print(f"  running {model_name} / {label} ...")
            results[(model_name, label)] = cv_scores(build(models()[model_name]), X[cols], y)
    print("\nDOES IT MATTER? (mean over 5 folds)")
    show(results)

# --------------------------------------------------------------------------- #
# Experiment 4: feature reduction (how many features do we really need?)
# --------------------------------------------------------------------------- #
# We start from the "one per cluster" features, then rank them in 3 ways and keep only the top k.
# The ranking is done INSIDE every CV fold (on that fold's training rows only), so the validation
# fold never influences which features are chosen.
METHODS = ["mutual info", "L1 logistic", "RF importance"]
K_LIST = [5, 10, 15, 20, 30]          # sizes to try; "all" (the full cluster set) is added automatically


def l1_logistic(C):
    """Logistic regression with an L1 penalty (pushes useless coefficients to exactly 0)."""
    major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):          # newer scikit-learn: penalty= is deprecated
        return LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=1000, random_state=RANDOM_SEED)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, random_state=RANDOM_SEED)


def rank_features(method, X_tr, y_tr):
    """Return the column names of X_tr ordered from most to least useful."""
    if method == "mutual info":          # filter: how much does the feature tell us about the label?
        scores = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_SEED)
    elif method == "L1 logistic":        # embedded: size of the coefficient after an L1 penalty
        scaled = StandardScaler().fit_transform(X_tr)
        scores = np.abs(l1_logistic(C=0.1).fit(scaled, y_tr).coef_[0])
    elif method == "RF importance":      # embedded: how much a Random Forest uses the feature
        forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        scores = forest.fit(X_tr, y_tr).feature_importances_
    else:
        raise ValueError(method)
    return list(X_tr.columns[np.argsort(-scores, kind="stable")])


def one_se_choice(table):
    """table: rows = k, columns = 'mean' and 'std' of F1 over folds.
    One-standard-error rule: the SMALLEST k whose mean F1 is within 1 standard error of the best."""
    best = table["mean"].idxmax()
    se = table.loc[best, "std"] / np.sqrt(5)                # 5 folds
    ok = table[table["mean"] >= table.loc[best, "mean"] - se]
    return int(ok.index.min())


def run_reduction():
    X, y = get_train()
    Xd = treated_copy(X)
    constant = Xd.columns[Xd.nunique() <= 1].tolist()
    X, Xd = X.drop(columns=constant), Xd.drop(columns=constant)
    pool = one_per_cluster(Xd, show=False)
    X = X[pool]
    ks = [k for k in K_LIST if k < len(pool)] + [len(pool)]
    print(f"Starting pool: {len(pool)} features (one per cluster). Trying k = {ks}")

    rows = []
    for fold, (tr, te) in enumerate(make_cv().split(X, y), start=1):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        for method in METHODS:
            ranking = rank_features(method, X_tr, y_tr)         # uses this fold's TRAIN rows only
            for k in ks:
                cols = ranking[:k]
                for model_name in models():
                    pipeline = build(models()[model_name]).fit(X_tr[cols], y_tr)
                    scores = all_metrics(y_te, pipeline.predict_proba(X_te[cols])[:, 1])
                    rows.append({"fold": fold, "method": method, "model": model_name, "k": k, **scores})
        print(f"  fold {fold}/5 done")

    results = pd.DataFrame(rows)

    # F1 by number of features, one table per model
    chosen = {}
    for model_name in models():
        part = results[results["model"] == model_name]
        print(f"\n{model_name}: mean F1 by number of features (last column = all {len(pool)})")
        print(part.pivot_table(index="method", columns="k", values="F1").round(4).to_string())
        for method in METHODS:
            f1 = part[part["method"] == method].groupby("k")["F1"].agg(["mean", "std"])
            chosen[(model_name, method)] = one_se_choice(f1)

    print("\nSMALLEST k within one standard error of the best F1:")
    table = pd.Series(chosen).unstack()
    print(table.to_string())

    # ONE shared k for every model: take each model's typical (median) k over the 3 ranking methods,
    # then the largest of those, so that no model is hurt. (The median stops one noisy cell in the
    # table above from pushing k up to "all features".)
    k = int(table.median(axis=1).max())
    print(f"\nShared k = {k}")

    # Majority vote. Rank ALL features with each method (full training set, no validation/test rows)
    position = pd.DataFrame({m: pd.Series(range(len(pool)), index=rank_features(m, X, y))
                             for m in METHODS})                  # position 0 = most useful
    print("\nDo the 3 rankings agree? (Spearman correlation between their positions)")
    print(position.corr(method="spearman").round(2).to_string())

    # A feature gets one vote for every method that puts it in its top k
    vote_table = pd.DataFrame({"votes": (position < k).sum(axis=1), "mean_position": position.mean(axis=1)})
    final = vote_table.sort_values(["votes", "mean_position"], ascending=[False, True]).head(k)
    print(f"\nFeatures by number of votes (out of {len(METHODS)}):")
    print(vote_table["votes"].value_counts().sort_index(ascending=False).to_string())
    borderline = final[final["votes"] < len(METHODS)]
    print("Elected without a unanimous vote:",
          ", ".join(f"{f} ({v}/{len(METHODS)} votes)" for f, v in borderline["votes"].items()) or "none")
    print("Left out:", ", ".join(vote_table.index.difference(final.index)) or "none")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/features_final.txt").write_text("\n".join(final.index))
    print(f"\nSaved the {len(final)} elected features -> reports/features_final.txt")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    steps = {"sentinel": run_sentinel, "scaling": run_scaling,
             "collinearity": run_collinearity, "reduction": run_reduction}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        sys.exit("usage: python b_experiments.py [sentinel|scaling|collinearity|reduction]")
    steps[sys.argv[1]]()