"""Training-only, five-fold robustness check for the twelve declared candidates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
if ROOT.name == "work":
    ROOT = ROOT.parent
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=os.environ.get("IN6227_DATA_DIR", "dataset"))
DATA = Path(parser.parse_args().data_dir)
if not (DATA / "train.csv").exists() and (DATA / "dataset" / "train.csv").exists():
    DATA = DATA / "dataset"
SEED = 42

train = pd.read_csv(DATA / "train.csv").dropna(subset=["label"]).copy()
features = [column for column in train if column != "label"]
numeric = train[features].select_dtypes(include="number").columns.tolist()
categorical = [column for column in features if column not in numeric]
train[categorical] = train[categorical].astype(object).where(pd.notna(train[categorical]), np.nan)
X = train[features]
y = (train["label"] == "yes").astype(int)


def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


candidates = {}
for weight in (None, "balanced"):
    for c in (0.1, 1, 10):
        candidates[f"logistic_C_{c:g}_weight_{weight or 'none'}"] = LogisticRegression(
            C=c, class_weight=weight, max_iter=1000, solver="lbfgs", random_state=SEED
        )
for depth in (12, None):
    for leaf in (2, 5, 10):
        candidates[f"forest_depth_{depth or 'none'}_leaf_{leaf}"] = RandomForestClassifier(
            n_estimators=150, max_depth=depth, min_samples_leaf=leaf,
            n_jobs=-1, random_state=SEED,
        )

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, classifier in candidates.items():
    for fold, (fit_index, val_index) in enumerate(folds.split(X, y), start=1):
        pipeline = Pipeline([("prep", preprocessing()), ("classifier", classifier)])
        pipeline.fit(X.iloc[fit_index], y.iloc[fit_index])
        probabilities = pipeline.predict_proba(X.iloc[val_index])[:, 1]
        truth = y.iloc[val_index]
        rows.append({
            "candidate": name,
            "family": "logistic_regression" if name.startswith("logistic") else "random_forest",
            "fold": fold,
            "average_precision": average_precision_score(truth, probabilities),
            "roc_auc": roc_auc_score(truth, probabilities),
            "recall_at_0_5": recall_score(truth, probabilities >= 0.5),
        })
    print(f"Completed {name}", flush=True)

fold_results = pd.DataFrame(rows)
(ROOT / "outputs").mkdir(exist_ok=True)
fold_results.to_csv(ROOT / "outputs" / "IN6227-cross-validation-folds.csv", index=False)
summary = fold_results.groupby(["candidate", "family"]).agg(
    cv_ap_mean=("average_precision", "mean"),
    cv_ap_sd=("average_precision", "std"),
    cv_roc_auc_mean=("roc_auc", "mean"),
    cv_recall_mean=("recall_at_0_5", "mean"),
).reset_index().sort_values(["family", "cv_ap_mean"], ascending=[True, False])
summary.to_csv(ROOT / "outputs" / "IN6227-cross-validation.csv", index=False)
print(summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
