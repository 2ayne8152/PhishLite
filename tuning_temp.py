from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from b_experiments import cv_scores, get_train
X, y = get_train()
cols = open("reports/features_final.txt").read().split()
for fs, Xs in [("111 features", X), ("30 selected", X[cols])]:
    for sc, est in [("unscaled", GaussianNB()), ("scaled", make_pipeline(StandardScaler(), GaussianNB()))]:
        s = cv_scores(est, Xs, y); print(f"{fs:13} {sc:9} F1 {s['F1'].mean():.4f}  FPR {s['FPR'].mean():.4f}")