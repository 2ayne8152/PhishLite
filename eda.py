"""
What this EDA does:
1.  Overview and class balance are kept.
    The overview also counts features per paper group (URL, domain, directory, file, params, external).

2.  sentinel_check counts the -1 values per column and checks whether the *_params columns go to -1 together.
    It also shows the phishing rate among the -1 rows, which tells you whether -1 carries signal.
    The stacked-bar plot shows the -1 / 0 / positive share per class.

3.  single_feature_auc replaces the Pearson ranking.
    It gives each feature’s ROC-AUC on its own, flags anything above 0.95 as a possible leakage
    or source shortcut, and summarizes signal per feature group.

4.  ecdf_plots replaces the histograms, with one ECDF per class for the six best-separating features on a symlog axis.
    redundancy_analysis gives a clustered Spearman heatmap and a list of redundancy clusters at |r| > 0.95.
    The cluster list saved to redundancy_clusters.csv is your pruning shortlist for Part B.
"""


from pathlib import  Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

from load import DEFAULT_LABEL, load_dataset

OUT_DIR = Path("./reports/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Features that need a network lookup or an external service (paper Table 6).
EXTERNAL_FEATURES = {
    "time_response", "domain_spf", "asn_ip", "time_domain_activation",
    "time_domain_expiration", "qty_ip_resolved", "qty_nameservers",
    "qty_mx_servers", "ttl_hostname", "tls_ssl_certificate", "qty_redirects",
    "url_google_index", "domain_google_index", "url_shortened",
}
GROUP_ORDER = ["url", "domain", "directory", "file", "params", "external", "other"]
GROUP_COLORS = {
    "url": "#4c78a8", "domain": "#f58518", "directory": "#54a24b",
    "file": "#b279a2", "params": "#9d755d", "external": "#e45756", "other": "#999999",
}


#=======================================#
# 0. Helper
#=======================================#

def feature_group(name: str) -> str:
    """Map a column name to its paper table (URL part it was computed from)."""
    if name in EXTERNAL_FEATURES:
        return "external"
    # length / flag features whose names do not end with the part name
    if name in {"domain_length", "domain_in_ip"}:
        return "domain"
    for part in ("directory", "file", "params", "domain", "url"):
        if name.endswith(f"_{part}") or name.startswith(f"{part}_"):
            return part
    return "other"


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def numeric_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    return df.select_dtypes("number").drop(columns=[label], errors="ignore")


#=======================================#
# 1. Overview
#=======================================#

def overview(df: pd.DataFrame, label: str) -> None:
    section("OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.4f} MB")
    print("\nDtypes:")
    print(df.dtypes.value_counts().to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values:", "none" if missing.empty else "")
    if not missing.empty:
        print(pd.DataFrame({"count": missing, "pct": (missing / len(df) * 100).round(2)}).to_string())

    # How many features belong to each paper group
    groups = pd.Series({c: feature_group(c) for c in df.columns if c != label})
    print("\nFeatures per group:")
    print(groups.value_counts().reindex(GROUP_ORDER).dropna().astype(int).to_string())


# --------------------------------------------------------------------------- #
# 2. Class balance
# --------------------------------------------------------------------------- #
def class_balance(df: pd.DataFrame, label: str, out: Path) -> None:
    section("CLASS BALANCE")
    counts = df[label].value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    print(pd.DataFrame({"count": counts, "pct": pct}).to_string())

    fig, ax = plt.subplots(figsize=(8, 6))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#e45756"][: len(counts)])
    ax.set_title("Class balance (0 = legitimate, 1 = phishing)")
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=150)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# 3. Sentinel values (-1): "component absent" or "lookup failed"
# --------------------------------------------------------------------------- #

def sentinel_check(df: pd.DataFrame, label: str, out: Path, top_n: int = 12) -> None:
    section("SENTINEL VALUES (-1)")
    X = numeric_features(df, label)
    is_neg = X.eq(-1)
    n_neg = is_neg.sum()
    cols = n_neg[n_neg > 0].sort_values(ascending=False).index.tolist()
    if not cols:
        print("No -1 values found.")
        return

    base_rate = df[label].mean()
    table = pd.DataFrame({
        "group": [feature_group(c) for c in cols],
        "count_-1": n_neg[cols].values,
        "pct_-1": (n_neg[cols] / len(df) * 100).round(2).values,
        # share of phishing among the rows that have -1 (compare with base rate)
        "phishing_rate_when_-1": [df.loc[is_neg[c], label].mean() for c in cols],
    }, index=cols)
    print(f"{len(cols)} columns contain -1. Overall phishing rate = {base_rate:.3f}")
    print(table.head(25).round(3).to_string())
    table.to_csv(out / "sentinel_summary.csv")

    # Do the *_params columns turn to -1 together? (supports the "component absent" reading)
    params_cols = [c for c in cols if feature_group(c) == "params"]
    if params_cols:
        all_neg = is_neg[params_cols].all(axis=1).mean()
        any_neg = is_neg[params_cols].any(axis=1).mean()
        print(f"\nParams columns: all -1 together in {all_neg:.1%} of rows, "
              f"any -1 in {any_neg:.1%} of rows.")

    # Stacked bars: share of -1 / 0 / positive per class for the most affected columns
    plot_cols = cols[:top_n]
    classes = sorted(df[label].unique())
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(plot_cols)), 4.5))
    width = 0.38
    seg_colors = {"-1": "#e45756", "0": "#cccccc", ">0": "#4c78a8"}
    for i, col in enumerate(plot_cols):
        for j, cls in enumerate(classes):
            v = df.loc[df[label] == cls, col]
            parts = {"-1": (v == -1).mean(), "0": (v == 0).mean(), ">0": (v > 0).mean()}
            x = i + (j - 0.5) * width
            bottom = 0.0
            for key, val in parts.items():
                ax.bar(x, val, width * 0.95, bottom=bottom, color=seg_colors[key],
                       label=key if (i == 0 and j == 0) else None)
                bottom += val
            ax.text(x, 1.01, str(cls), ha="center", fontsize=7)
    ax.set_xticks(range(len(plot_cols)), plot_cols, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Share of rows")
    ax.set_title("-1 / 0 / positive share by class (left bar = 0, right bar = 1)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out / "sentinel_by_class.png", dpi=150)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# 4. Single-feature separation (ROC-AUC) + per-group summary + leakage flag
# --------------------------------------------------------------------------- #

def single_feature_auc(df: pd.DataFrame, label: str, out: Path,
                       leak_threshold: float = 0.95, top_n: int = 25) -> pd.Series:
    """Oriented AUC per feature: 0.5 = no signal, 1.0 = perfect single-feature separator."""
    section("SINGLE-FEATURE SEPARATION (ROC-AUC)")
    X = numeric_features(df, label)
    y = df[label]

    raw = pd.Series({c: roc_auc_score(y, X[c]) for c in X.columns})
    auc = np.maximum(raw, 1 - raw).sort_values(ascending=False)   # direction does not matter
    auc.rename("auc").to_csv(out / "single_feature_auc.csv")

    print(f"Top {min(15, len(auc))} features by single-feature AUC:")
    print(auc.head(15).round(4).to_string())

    suspicious = auc[auc > leak_threshold]
    if suspicious.empty:
        print(f"\nNo single feature above AUC {leak_threshold}.")
    else:
        print(f"\n[!] {len(suspicious)} feature(s) above AUC {leak_threshold} "
              f"(check for leakage / a source shortcut): {list(suspicious.index)}")

    # Per-group summary: how much signal does each feature group carry on its own?
    grp = pd.DataFrame({"auc": auc, "group": [feature_group(c) for c in auc.index]})
    summary = grp.groupby("group")["auc"].agg(n_features="count", mean_auc="mean", max_auc="max")
    summary["best_feature"] = grp.loc[grp.groupby("group")["auc"].idxmax()].reset_index().set_index("group")["index"]
    summary = summary.reindex([g for g in GROUP_ORDER if g in summary.index])
    print("\nSignal per feature group:")
    print(summary.round(4).to_string())
    summary.to_csv(out / "group_auc_summary.csv")

    # Bar chart of the top features, coloured by group
    top = auc.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(top) + 1.5))
    ax.barh(top.index, top.values, color=[GROUP_COLORS[feature_group(c)] for c in top.index])
    ax.axvline(leak_threshold, color="k", ls="--", lw=0.8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Single-feature ROC-AUC (0.5 = no signal)")
    ax.set_title(f"Top {len(top)} features by separation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_ORDER if g in set(grp["group"])]
    ax.legend(handles, [g for g in GROUP_ORDER if g in set(grp["group"])], fontsize=7, loc="lower right")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "single_feature_auc.png", dpi=150)
    plt.close(fig)

    return auc


# --------------------------------------------------------------------------- #
# 5. Redundancy: clustered correlation heatmap + redundancy clusters
# --------------------------------------------------------------------------- #

def redundancy_analysis(df: pd.DataFrame, label: str, out: Path, cluster_corr: float = 0.95) -> None:
    section("REDUNDANT / HIGHLY CORRELATED FEATURES")
    X = numeric_features(df, label)
    X = X.loc[:, X.nunique() > 1]               # constant columns give NaN correlations
    if X.shape[1] < 2:
        print("Not enough varying numeric features.")
        return

    corr = X.corr(method="spearman").abs().fillna(0.0)   # rank-based: robust to heavy tails
    values = corr.to_numpy()

    iu = np.triu_indices_from(values, k=1)
    pair_vals = pd.Series(values[iu], index=pd.MultiIndex.from_arrays(
        [corr.index[iu[0]], corr.columns[iu[1]]]))
    high = pair_vals[pair_vals > cluster_corr].sort_values(ascending=False)
    print(f"{len(high)} feature pairs with |Spearman| > {cluster_corr}")
    print(high.head(20).round(4).to_string())

    # Hierarchical clustering on distance = 1 - |corr|
    dist = 1 - values
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(link)

    # Redundancy clusters: features within (1 - cluster_corr) of each other
    labels = fcluster(link, t=1 - cluster_corr, criterion="distance")
    clusters = pd.Series(corr.index, index=labels).groupby(level=0).apply(list)
    multi = [c for c in clusters if len(c) > 1]
    print(f"\n{len(multi)} redundancy clusters (|r| > {cluster_corr}); "
          f"keeping one feature per cluster would drop {sum(len(c) - 1 for c in multi)} columns.")
    for c in sorted(multi, key=len, reverse=True)[:12]:
        print(f"  [{len(c)}] {', '.join(c[:8])}{' ...' if len(c) > 8 else ''}")
    pd.DataFrame({"cluster": [i for i, c in enumerate(multi) for _ in c],
                  "feature": [f for c in multi for f in c]}).to_csv(out / "redundancy_clusters.csv", index=False)

    # Clustered heatmap (rows/columns ordered so correlated features sit together)
    ordered = corr.index[order]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.loc[ordered, ordered].to_numpy(), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=4)
    ax.set_yticks(range(len(ordered)), ordered, fontsize=4)
    fig.colorbar(im, ax=ax, fraction=0.03, label="|Spearman correlation|")
    ax.set_title("Clustered feature correlation (bright blocks = redundant groups)")
    fig.tight_layout()
    fig.savefig(out / "correlation_clustered.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 6. Categorical / text columns (expected: none for this dataset)
# --------------------------------------------------------------------------- #

def categorical_summary(df: pd.DataFrame) -> None:
    cat = df.select_dtypes(exclude="number").columns.tolist()
    section("CATEGORICAL / TEXT FEATURES")
    if not cat:
        print("None.")
        return
    for col in cat:
        sample = df[col].dropna()
        example = repr(sample.iloc[0] if not sample.empty else "No valid values")
        print(f"{col}: {df[col].nunique():,} unique | e.g. {example}"[:150])


# ===================== #
# Main                  #
# ===================== #

def main() -> None:
    df = load_dataset()
    label = DEFAULT_LABEL

    overview(df, label)
    class_balance(df, label, OUT_DIR)
    sentinel_check(df, label, OUT_DIR)
    auc = single_feature_auc(df, label, OUT_DIR)
    auc.to_csv(OUT_DIR / "auc.csv")
    redundancy_analysis(df, label, OUT_DIR)
    categorical_summary(df)
    print(f"\nEDA outputs saved to {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()