"""Reproducible IN6227 Assignment 1, Variant 1 classification comparison.

Usage: python assignment1_analysis.py --data-dir PATH --output PATH
Supplied test labels are excluded from validation-based model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


def summarize(y, p):
    pred = p >= 0.5
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision_yes": precision_score(y, pred, zero_division=0),
        "recall_yes": recall_score(y, pred, zero_division=0),
        "f1_yes": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "confusion_matrix_no_yes": confusion_matrix(y, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    assert train.columns.tolist() == test.columns.tolist()
    feature_columns = [c for c in train.columns if c != "label"]
    numeric = train[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]

    train_clean = train.dropna(subset=["label"]).copy()
    test_clean = test.dropna(subset=["label"]).copy()
    assert set(train_clean.label) == {"no", "yes"}
    assert set(test_clean.label) == {"no", "yes"}
    # Object columns avoid pandas StringDtype's pd.NA incompatibility with sklearn.
    for frame in (train_clean, test_clean):
        frame[categorical] = frame[categorical].astype(object)
        frame[categorical] = frame[categorical].where(pd.notna(frame[categorical]), np.nan)

    X = train_clean[feature_columns]
    y = (train_clean.label == "yes").astype(int)
    X_test = test_clean[feature_columns]
    y_test = (test_clean.label == "yes").astype(int)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

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

    candidates = {
        "logistic_C_0.1": LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=SEED),
        "logistic_C_1": LogisticRegression(C=1, max_iter=1000, solver="lbfgs", random_state=SEED),
        "logistic_C_10": LogisticRegression(C=10, max_iter=1000, solver="lbfgs", random_state=SEED),
        "forest_depth_12": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "forest_depth_none": RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    }
    validation = {}
    for name, classifier in candidates.items():
        pipe = Pipeline([("prep", preprocessing()), ("classifier", classifier)])
        pipe.fit(X_fit, y_fit)
        validation[name] = summarize(y_val, pipe.predict_proba(X_val)[:, 1])
        print(name, "validation AP", round(validation[name]["average_precision"], 4), flush=True)

    selections = {
        family: max((name for name in candidates if name.startswith(prefix)),
                    key=lambda name: validation[name]["average_precision"])
        for family, prefix in [("logistic_regression", "logistic"), ("random_forest", "forest")]
    }
    held_out = {}
    test_probabilities = {}
    for family, name in selections.items():
        pipe = Pipeline([("prep", preprocessing()), ("classifier", candidates[name])])
        pipe.fit(X, y)
        probabilities = pipe.predict_proba(X_test)[:, 1]
        test_probabilities[family] = probabilities
        held_out[family] = {"selected_configuration": name,
                            **summarize(y_test, probabilities)}

    # A paired bootstrap describes sampling uncertainty in the observed AP gap.
    # It is descriptive only: the test set was not used to choose models.
    rng = np.random.default_rng(SEED)
    paired_differences = []
    y_array = y_test.to_numpy()
    for _ in range(1000):
        sample = rng.integers(0, len(y_array), len(y_array))
        if len(np.unique(y_array[sample])) < 2:
            continue
        paired_differences.append(
            average_precision_score(y_array[sample], test_probabilities["random_forest"][sample])
            - average_precision_score(y_array[sample], test_probabilities["logistic_regression"][sample])
        )

    report = {
        "dataset": {
            "train_rows_raw": len(train), "test_rows_raw": len(test),
            "train_rows_labeled": len(train_clean), "test_rows_labeled": len(test_clean),
            "numeric_features": numeric, "categorical_features": categorical,
            "train_missing_feature_cells": int(train[feature_columns].isna().sum().sum()),
            "test_missing_feature_cells": int(test[feature_columns].isna().sum().sum()),
            "train_duplicate_rows": int(train.duplicated().sum()),
            "test_duplicate_rows": int(test.duplicated().sum()),
            "train_yes_rate": float(y.mean()), "test_yes_rate": float(y_test.mean()),
            "validation_yes_rate": float(y_val.mean()),
        },
        "validation": validation,
        "selections": selections,
        "held_out_test": held_out,
        "test_ap_difference_forest_minus_logistic_95pct_bootstrap_interval":
            np.quantile(paired_differences, [0.025, 0.975]).tolist(),
        "majority_no_test_accuracy": float((y_test == 0).mean()),
        "library_versions": {"pandas": pd.__version__, "numpy": np.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selections": selections, "held_out_test": held_out}, indent=2))


if __name__ == "__main__":
    main()

