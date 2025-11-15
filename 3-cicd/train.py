# -*- coding: utf-8 -*-
"""
Mental Health Training Script — with Kaggle download support

Features:
- Load data from a local CSV (default ../data/raw/survey.csv) OR directly from Kaggle using --kaggle_slug.
- Applies the same preprocessing used in your notebook/training (Gender cleaning + ColumnTransformer).
- Uses selected/retained features if feature_selection_results.txt exists (optional).
- Trains on FULL data (no validation) with your chosen/best model and logs to MLflow.
- Saves training_schema.json and run_id.txt for serving.

Kaggle usage examples:
    pip install kaggle
    # set MLFLOW_TRACKING_URI as you prefer (server or file store)
    python train.py --kaggle_slug osmi/mental-health-in-tech-2016 --kaggle_file survey.csv \
        --experiment mental-health-tech-prediction

Local file:
    python train.py --data ../data/raw/survey.csv --experiment mental-health-tech-prediction
"""

import os
import re
import sys
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import argparse
import mlflow
import mlflow.sklearn


# -----------------------------
# Helpers (notebook parity)
# -----------------------------
def clean_gender(gen: object) -> str:
    s = str(gen).strip().lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    if s in {"m", "male", "man", "make", "mal", "malr", "msle", "masc", "mail", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "femake", "femail", "femme", "girl"}:
        return "Female"
    return "Other"


def read_retained_features(path: str) -> list:
    if not os.path.exists(path):
        return []
    retained = []
    in_section = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip().lower().startswith("retained features"):
                in_section = True
                continue
            if in_section:
                if not line.strip():
                    break
                if line.strip().startswith("- "):
                    retained.append(line.strip()[2:])
                else:
                    break
    return retained


def maybe_read_best_model(path: str) -> str:
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read().strip()
    return ""


def maybe_read_best_params(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
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
    name = (model_name or "").lower()
    if name in {"xgb", "xgboost"} or not name:
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
        default = dict(max_iter=2000, solver="lbfgs")
        default.update(params or {})
        return LogisticRegression(**default)
    raise ValueError(f"Unknown model name: {model_name}")


# -----------------------------
# Kaggle download utilities
# -----------------------------
def _have_kaggle_cli() -> bool:
    from shutil import which
    return which("kaggle") is not None


def _kaggle_download(slug: str, out_dir: Path) -> Path:
    """
    Download a Kaggle dataset to out_dir using the Kaggle CLI.
    Returns the path to the first CSV found (preferring *survey*.csv).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "-q"]
    subprocess.check_call(cmd)

    csvs = []
    for z in out_dir.glob("*.zip"):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_dir)
        z.unlink()
    csvs.extend(out_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found after downloading Kaggle dataset: {slug}")
    for cand in csvs:
        if "survey" in cand.name.lower():
            return cand
    return csvs[0]


def load_from_kaggle(slug: str, preferred_file: str | None = None) -> pd.DataFrame:
    if not _have_kaggle_cli():
        raise RuntimeError("Kaggle CLI not found. Install with `pip install kaggle` and configure your API token.")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        csv_path = _kaggle_download(slug, tmpdir)
        if preferred_file:
            cand = tmpdir / preferred_file
            if not cand.exists():
                matches = list(tmpdir.rglob(preferred_file))
                if not matches:
                    raise FileNotFoundError(
                        f"Requested file '{preferred_file}' not found in dataset {slug}. "
                        f"Available: {[p.name for p in tmpdir.rglob('*.csv')]}"
                    )
                csv_path = matches[0]
            else:
                csv_path = cand
        print(f"[KAGGLE] Using file: {csv_path.name}")
        return pd.read_csv(csv_path)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/raw/survey.csv", help="Path to survey CSV (if not using Kaggle)")
    parser.add_argument("--mlflow_uri", default="http://127.0.0.1:5000", help="MLflow Tracking URI")
    parser.add_argument("--experiment", default="mental-health-tech-prediction", help="MLflow experiment name")
    parser.add_argument("--model", default="", help="Force model (XGBoost/RandomForest/LogisticRegression)")
    parser.add_argument("--best_model_file", default="best_model_name.txt", help="Optional file with best model name")
    parser.add_argument("--best_params_file", default="best_params.json", help="Optional JSON with tuned hyperparams")
    parser.add_argument("--features_file", default="feature_selection_results.txt", help="Optional retained-features file")
    parser.add_argument("--output_runid", default="run_id.txt", help="Where to write MLflow run_id")
    # Kaggle flags
    parser.add_argument("--kaggle_slug", default="", help="Kaggle dataset slug like 'osmi/mental-health-in-tech-2016'")
    parser.add_argument("--kaggle_file", default="", help="Optional CSV filename inside the Kaggle dataset")
    args = parser.parse_args()

    # -----------------------------
    # Load data (Kaggle or local)
    # -----------------------------
    if args.kaggle_slug:
        df = load_from_kaggle(args.kaggle_slug, args.kaggle_file or None)
        print(f"Loaded data from Kaggle: {df.shape}")
    else:
        if not os.path.exists(args.data):
            print(f"[ERROR] Data not found: {args.data}", file=sys.stderr)
            sys.exit(2)
        df = pd.read_csv(args.data)
        print(f"Loaded data: {df.shape} from {args.data}")

    # -----------------------------
    # Target & feature prep
    # -----------------------------
    target_col = "treatment"
    if target_col not in df.columns:
        print(f"[ERROR] Expected target column '{target_col}' not found in data.", file=sys.stderr)
        sys.exit(3)

    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    features_to_drop = ["Timestamp", "Country", "state", "comments", target_col]
    existing_to_drop = [c for c in features_to_drop if c in df.columns]
    X = df.drop(columns=existing_to_drop)

    if "Gender" in X.columns:
        X["Gender"] = X["Gender"].apply(clean_gender)

    retained = read_retained_features(args.features_file)
    if retained:
        retained = [c for c in retained if c in X.columns]
        if retained:
            X = X[retained]
            print(f"Using {len(retained)} retained features from {args.features_file}")

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Numeric: {len(numeric_features)} | Categorical: {len(categorical_features)}")

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    chosen_model_name = args.model or maybe_read_best_model(args.best_model_file) or "XGBoost"
    best_params = maybe_read_best_params(args.best_params_file)
    model = build_model(chosen_model_name, best_params)

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    # -----------------------------
    # MLflow logging (NO validation; fit on full data)
    # -----------------------------
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

        mlflow.log_params({
            "model_name": chosen_model_name,
            "n_numeric_features": len(numeric_features),
            "n_categorical_features": len(categorical_features),
            "n_rows": X.shape[0],
            "n_cols": X.shape[1],
        })
        if best_params:
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

        pipeline.fit(X, y)

        schema = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "retained_features": retained or "ALL_COLUMNS",
            "target": target_col
        }
        with open("training_schema.json", "w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2)
        mlflow.log_artifact("training_schema.json")

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        with open(args.output_runid, "w", encoding="utf-8") as fh:
            fh.write(run_id)

        print("Training complete. Model and artifacts logged to MLflow.")
        print("Run ID saved to", args.output_runid)


if __name__ == "__main__":
    main()
