#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted training script from the Jupyter notebook:
- Uses the notebook's cleaned features and preprocessing.
- Applies the selected/retained features (if a file from the notebook exists).
- Uses the notebook's "best model" (if recorded); otherwise defaults to XGBoost.
- Trains on the FULL dataset (no validation split), as requested.
- Logs to MLflow and saves the fitted pipeline.

Assumptions:
- Data CSV path: ../data/raw/survey.csv (same as the notebook).
- Optional files produced by the notebook (if you ran it):
    - feature_selection_results.txt  (contains "Retained features" list)
    - best_model_name.txt            (contains a single line with model name, e.g., "XGBoost")
    - best_params.json               (JSON object of tuned hyperparameters for that model)

You can override via CLI flags:
    python adapted_notebook_train.py --data ../data/raw/survey.csv --mlflow_uri http://localhost:5000 \\
        --experiment mental-health-tech-prediction --model XGBoost
"""

import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Helpers mirroring the notebook
# -----------------------------

def clean_gender(gen: object) -> str:
    """Normalize gender categories exactly like the notebook."""
    s = str(gen).strip().lower()
    s = re.sub(r"[\\W_]+", " ", s).strip()
    if s in {"m", "male", "man", "make", "mal", "malr", "msle", "masc", "mail", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "femake", "femail", "femme", "girl"}:
        return "Female"
    return "Other"


def read_retained_features(path: str) -> list:
    """
    Parse feature_selection_results.txt (if present) to extract retained features.
    Expected format sections:
        Retained features:
        - col_a
        - col_b
    """
    if not os.path.exists(path):
        return []
    retained = []
    in_section = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\\n")
            if line.strip().lower().startswith("retained features"):
                in_section = True
                continue
            if in_section:
                if not line.strip():
                    break
                if line.strip().startswith("- "):
                    retained.append(line.strip()[2:])
                else:
                    # stop if section changed
                    break
    return retained


def maybe_read_best_model(path: str) -> str:
    """Read best model name from a tiny text file if it exists; else return empty string."""
    if os.path.exists(path):
        val = open(path, "r", encoding="utf-8").read().strip()
        return val
    return ""


def maybe_read_best_params(path: str) -> dict:
    """Read best hyperparameters JSON if it exists; else return {}."""
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """ColumnTransformer mirroring the notebook: median+scale for numeric, most_frequent+OHE for categorical."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")
    return preprocessor


def build_model(model_name: str, params: dict):
    """Construct the classifier specified by model_name with params (if any)."""
    name = model_name.lower()
    if name in {"xgb", "xgboost"}:
        default = dict(
            random_state=42, n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, reg_lambda=1.0, objective="binary:logistic",
            eval_metric="logloss"
        )
        default.update(params or {})
        return XGBClassifier(**default)
    if name in {"rf", "randomforest", "random_forest"}:
        default = dict(
            n_estimators=400, max_depth=None, min_samples_split=2,
            class_weight=None, random_state=42, n_jobs=-1
        )
        default.update(params or {})
        return RandomForestClassifier(**default)
    if name in {"logreg", "logistic", "logisticregression"}:
        default = dict(max_iter=2000, solver="lbfgs", random_state=42)
        default.update(params or {})
        return LogisticRegression(**default)
    raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/raw/survey.csv", help="Path to survey.csv (same as notebook)")
    parser.add_argument("--mlflow_uri", default="http://localhost:5000", help="MLflow Tracking URI")
    parser.add_argument("--experiment", default="mental-health-tech-prediction", help="MLflow experiment name")
    parser.add_argument("--model", default="", help="Force model name (e.g., XGBoost, RandomForest, LogisticRegression)")
    parser.add_argument("--best_model_file", default="best_model_name.txt", help="Optional file with best model name")
    parser.add_argument("--best_params_file", default="best_params.json", help="Optional JSON file with best hyperparams")
    parser.add_argument("--features_file", default="feature_selection_results.txt", help="Optional retained-features file")
    parser.add_argument("--output_runid", default="run_id.txt", help="Where to save MLflow run_id")
    args = parser.parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}", file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.data)
    print(f"Loaded data: {df.shape} from {args.data}")

    # -----------------------------
    # Target & feature prep (as in notebook)
    # -----------------------------
    target_col = "treatment"
    if target_col not in df.columns:
        print(f"[ERROR] Expected target column '{target_col}' not found in data.", file=sys.stderr)
        sys.exit(3)

    # Binary mapping Yes/No -> 1/0
    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    features_to_drop = ["Timestamp", "Country", "state", "comments", target_col]
    existing_to_drop = [c for c in features_to_drop if c in df.columns]
    X = df.drop(columns=existing_to_drop)

    # Gender cleaning
    if "Gender" in X.columns:
        X["Gender"] = X["Gender"].apply(clean_gender)

    # Use retained features if provided
    retained = read_retained_features(args.features_file)
    if retained:
        # Keep intersection to avoid KeyError
        retained = [c for c in retained if c in X.columns]
        if not retained:
            print("[WARN] Retained-features list was empty after intersecting with columns; falling back to all features.")
        else:
            X = X[retained]
            print(f"Using {len(retained)} retained features from {args.features_file}")

    # Separate features by dtype (same approach as the notebook)
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric features: {len(numeric_features)} | Categorical features: {len(categorical_features)}")

    # -----------------------------
    # Preprocessor & model
    # -----------------------------
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Resolve best model
    chosen_model_name = args.model or maybe_read_best_model(args.best_model_file) or "XGBoost"
    best_params = maybe_read_best_params(args.best_params_file)
    model = build_model(chosen_model_name, best_params)

    # Full pipeline
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # -----------------------------
    # MLflow logging & training (NO validation split; fit on all data)
    # -----------------------------
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

        # Log basic params
        mlflow.log_params({
            "model_name": chosen_model_name,
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "n_rows": X.shape[0],
            "n_cols": X.shape[1],
        })
        if best_params:
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

        # Fit on FULL data
        pipeline.fit(X, y)

        # Log artifact: retained features (if any) and schema
        schema = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "retained_features": retained or "ALL_COLUMNS",
            "target": target_col
        }
        with open("training_schema.json", "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2)
        mlflow.log_artifact("training_schema.json")

        # Log the full sklearn pipeline
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # Persist run_id for downstream use
        with open(args.output_runid, "w", encoding="utf-8") as fh:
            fh.write(run_id)

        print("Training complete. Model and artifacts logged to MLflow.")
        print("Run ID saved to", args.output_runid)


if __name__ == "__main__":
    main()
