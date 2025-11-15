# -*- coding: utf-8 -*-
"""
Mental Health Training Script — Kaggle-only (robust CSV selection)

Highlights:
- Downloads ONLY from Kaggle (no local CSV fallback).
- Defaults to slug: osmi/mental-health-in-tech-survey (override with --kaggle_slug).
- If multiple CSVs exist, it **prefers a file containing 'survey'**; otherwise picks the largest CSV.
- If --kaggle_file is given, it tries exact match, then case-insensitive suffix match,
  then case-insensitive substring match; otherwise it raises with a helpful list.
- Uses TemporaryDirectory and **deletes** all downloaded files afterward.
- Preprocessing: Gender cleaning + ColumnTransformer (impute+scale numeric, impute+one-hot categorical).
- Trains on FULL data (no validation), logs sklearn Pipeline to MLflow, and writes:
  - training_schema.json
  - run_id.txt

Usage example:
    pip install kaggle
    # Put kaggle.json in ~/.kaggle/ (macOS/Linux) or C:\\Users\\<you>\\.kaggle\\ (Windows)
    # or export KAGGLE_USERNAME / KAGGLE_KEY
    python train.py --kaggle_file survey.csv \
        --experiment mental-health-tech-prediction \
        --mlflow_uri http://127.0.0.1:5000
"""

import os
import re
import sys
import json
import zipfile
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
# Kaggle download utilities (robust) with explicit cleanup
# -----------------------------
def _have_kaggle_cli() -> bool:
    from shutil import which
    return which("kaggle") is not None


def _ensure_kaggle_credentials():
    cfg_path = Path.home() / ".kaggle" / "kaggle.json"
    has_file = cfg_path.exists()
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if not (has_file or has_env):
        raise RuntimeError(
            "Kaggle credentials not found.\n"
            "Add C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows) or ~/.kaggle/kaggle.json (macOS/Linux), "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )


def _choose_csv(candidates: List[Path], preferred_file: Optional[str]) -> Tuple[Path, str]:
    """
    Choose a CSV from candidates using robust rules:
    - If preferred_file is given:
        1) exact name match,
        2) case-insensitive suffix match,
        3) case-insensitive substring match,
       otherwise raise with list of available files.
    - If no preferred_file:
        1) any file whose name contains 'survey' (case-insensitive); if multiple, pick the largest,
        2) else pick the largest CSV by size.
    """
    if not candidates:
        raise FileNotFoundError("No CSV files found in the Kaggle dataset contents.")

    if preferred_file:
        pref = preferred_file.lower()
        # exact
        exact = [p for p in candidates if p.name == preferred_file]
        if exact:
            chosen = exact[0]
            return chosen, chosen.name
        # suffix (useful when files live in folders or have prefixes)
        suffix = [p for p in candidates if p.name.lower().endswith(pref)]
        if suffix:
            # if multiple, largest wins
            chosen = max(suffix, key=lambda p: p.stat().st_size)
            return chosen, chosen.name
        # substring
        substr = [p for p in candidates if pref in p.name.lower()]
        if substr:
            chosen = max(substr, key=lambda p: p.stat().st_size)
            return chosen, chosen.name

        raise FileNotFoundError(
            f"Requested file '{preferred_file}' not found. "
            f"Available: {[p.name for p in candidates]}"
        )

    # No preferred file: prefer names with 'survey'
    surveyish = [p for p in candidates if "survey" in p.name.lower()]
    if surveyish:
        chosen = max(surveyish, key=lambda p: p.stat().st_size)
        return chosen, chosen.name

    # Otherwise pick the largest CSV available
    chosen = max(candidates, key=lambda p: p.stat().st_size)
    return chosen, chosen.name


def _kaggle_download_to_tmp(slug: str, preferred_file: Optional[str]) -> Tuple[pd.DataFrame, str]:
    """
    Download the Kaggle dataset into a TemporaryDirectory and return (DataFrame, chosen_filename).
    Ensures all temporary files are cleaned after reading.
    """
    if not _have_kaggle_cli():
        raise RuntimeError("Kaggle CLI not found. Install with `pip install kaggle` and configure your API token.")
    _ensure_kaggle_credentials()

    tmpdir_obj = tempfile.TemporaryDirectory()
    tmpdir = Path(tmpdir_obj.name)
    try:
        # download
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(tmpdir), "-q"]
        subprocess.check_call(cmd)

        # unzip everything
        for z in tmpdir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(tmpdir)
            z.unlink()

        # collect CSVs (from root or subfolders)
        candidates = list(tmpdir.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found after downloading Kaggle dataset: {slug}")

        chosen_path, chosen_name = _choose_csv(candidates, preferred_file)
        df = pd.read_csv(chosen_path)
        return df, chosen_name
    finally:
        # cleanup
        try:
            tmpdir_obj.cleanup()
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)


# -----------------------------
# Main (Kaggle-only)
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", default="http://127.0.0.1:5000", help="MLflow Tracking URI")
    parser.add_argument("--experiment", default="mental-health-tech-prediction", help="MLflow experiment name")
    parser.add_argument("--model", default="", help="Force model (XGBoost/RandomForest/LogisticRegression)")
    parser.add_argument("--best_model_file", default="best_model_name.txt", help="Optional file with best model name")
    parser.add_argument("--best_params_file", default="best_params.json", help="Optional JSON with tuned hyperparams")
    parser.add_argument("--features_file", default="feature_selection_results.txt", help="Optional retained-features file")
    parser.add_argument("--output_runid", default="run_id.txt", help="Where to write MLflow run_id")

    # Default to your preferred dataset (you can override via CLI)
    parser.add_argument(
        "--kaggle_slug",
        default="osmi/mental-health-in-tech-survey",
        required=False,
        help="Kaggle dataset slug (default: osmi/mental-health-in-tech-survey)"
    )
    parser.add_argument("--kaggle_file", default="", help="Optional CSV filename (robust matching enabled)")
    args = parser.parse_args()

    # -----------------------------
    # Load data FROM KAGGLE (only; auto-clean temporary files)
    # -----------------------------
    df, chosen_name = _kaggle_download_to_tmp(args.kaggle_slug, args.kaggle_file or None)
    print(f"[KAGGLE] Using file: {chosen_name}  |  shape={df.shape}")

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
