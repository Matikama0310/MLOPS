# -*- coding: utf-8 -*-
"""
Simplified training script for OSMI Mental Health survey.

What it does (only):
  1) Downloads the dataset from Kaggle (osmi/mental-health-in-tech-survey).
  2) Minimal preprocessing + feature engineering.
  3) Train/validation split.
  4) Train a model with the SAME core hyperparameters as the original train.py
     (max_depth=8, learning_rate=0.1, n_estimators=200, subsample=0.8,
      colsample_bytree=0.8, random_state=42), adapted to XGBClassifier.
  5) Prints evaluation metrics at the end.

Requirements:
  - Kaggle CLI installed and configured (kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
  - xgboost, scikit-learn, pandas, numpy.

Run:
  python train.py
"""

from __future__ import annotations

import os
import re
import zipfile
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)
from xgboost import XGBClassifier

# -------------------- Config --------------------
KAGGLE_DATASET = "osmi/mental-health-in-tech-survey"
PREFERRED_FILE = None  # if you know the exact CSV name, put it here; otherwise we auto-pick
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Keep EXACTLY these from the original train.py (adapted to classifier)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=6,
    learning_rate=0.01,
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=1.0,
    random_state=42,
)

# Columns to retain before preprocessing (subset of original features)
RETAINED_FEATURES = [
    "Gender",
    "self_employed",
    "family_history",
    "work_interfere",
    "no_employees",
    "remote_work",
    "benefits",
    "care_options",
    "wellness_program",
    "seek_help",
    "leave",
    "mental_health_consequence",
    "phys_health_consequence",
    "coworkers",
    "supervisor",
    "mental_health_interview",
    "phys_health_interview",
    "mental_vs_physical",
    "obs_consequence",
]

# -------------------- Kaggle download --------------------

def _have_kaggle_cli() -> bool:
    from shutil import which
    return which("kaggle") is not None


def _ensure_kaggle_credentials():
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return
    raise RuntimeError(
        "Kaggle credentials not found. Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME/KAGGLE_KEY."
    )


def _choose_csv(cands: List[Path], preferred: str | None) -> Path:
    if not cands:
        raise FileNotFoundError("No CSV files found in the Kaggle dataset archive.")
    if preferred:
        # direct match or suffix/substring match, otherwise error
        exact = [p for p in cands if p.name == preferred]
        if exact:
            return exact[0]
        suf = [p for p in cands if p.name.lower().endswith(preferred.lower())]
        if suf:
            return max(suf, key=lambda p: p.stat().st_size)
        sub = [p for p in cands if preferred.lower() in p.name.lower()]
        if sub:
            return max(sub, key=lambda p: p.stat().st_size)
        raise FileNotFoundError(
            f"Requested '{preferred}' not found. Available: {[p.name for p in cands]}"
        )
    # Prefer anything with 'survey' in the name, else the largest CSV
    surveyish = [p for p in cands if "survey" in p.name.lower()]
    if surveyish:
        return max(surveyish, key=lambda p: p.stat().st_size)
    return max(cands, key=lambda p: p.stat().st_size)


def load_from_kaggle() -> Tuple[pd.DataFrame, str]:
    if not _have_kaggle_cli():
        raise RuntimeError("Install and configure Kaggle CLI: pip install kaggle")
    _ensure_kaggle_credentials()

    tmp = tempfile.TemporaryDirectory()
    tmpdir = Path(tmp.name)
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(tmpdir), "-q"
        ])
        for z in tmpdir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(tmpdir)
            z.unlink()
        csvs = list(tmpdir.rglob("*.csv"))
        path = _choose_csv(csvs, PREFERRED_FILE)
        df = pd.read_csv(path)
        return df.copy(), path.name
    finally:
        try:
            tmp.cleanup()
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)


# -------------------- Preprocessing --------------------

def clean_gender(x: object) -> str:
    s = re.sub(r"[\W_]+", " ", str(x).strip().lower())
    if s in {"m", "male", "man", "make", "mal", "malr", "msle", "masc", "mail", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "femake", "femail", "femme", "girl"}:
        return "Female"
    return "Other"


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = "treatment"
    if target not in df.columns:
        raise KeyError(f"Expected target column '{target}' not found in data.")

    y = df[target].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[c for c in ["Timestamp", "Country", "state", "comments", target] if c in df.columns])

    # Basic cleaning
    if "Gender" in X.columns:
        X["Gender"] = X["Gender"].apply(clean_gender)

    # Keep only the requested features that exist in the data
    available = [c for c in RETAINED_FEATURES if c in X.columns]
    missing = [c for c in RETAINED_FEATURES if c not in X.columns]
    if missing:
        print(f"[Info] Missing requested columns (will be skipped): {missing}")
    if not available:
        raise ValueError("None of the requested RETAINED_FEATURES exist in the dataset.")
    X = X[available].copy()

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


# -------------------- Train & Evaluate --------------------

def train_and_eval(X: pd.DataFrame, y: pd.Series) -> None:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y
    )

    model = XGBClassifier(**XGB_PARAMS)
    pre = build_preprocessor(Xtr)
    pipe = Pipeline([
        ("preprocess", pre),
        ("model", model),
    ])

    pipe.fit(Xtr, ytr)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        yproba = pipe.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, yproba))
    else:
        yproba = None
        auc = float("nan")

    ypred = pipe.predict(Xte)
    acc = float(accuracy_score(yte, ypred))
    pre = float(precision_score(yte, ypred, zero_division=0))
    rec = float(recall_score(yte, ypred, zero_division=0))
    f1  = float(f1_score(yte, ypred, zero_division=0))

    print("\n=== Validation Metrics ===")
    if not np.isnan(auc):
        print(f"AUC:        {auc:.4f}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {pre:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-score:   {f1:.4f}")


# -------------------- Main --------------------

def main() -> None:
    print("Downloading dataset from Kaggle …")
    df, chosen = load_from_kaggle()
    print(f"Loaded '{chosen}' with shape: {df.shape}")

    print("Preprocessing …")
    X, y = preprocess(df)
    print(f"Features after filtering: {X.shape[1]} | Rows: {X.shape[0]}")

    print("Training & evaluating …")
    train_and_eval(X, y)


if __name__ == "__main__":
    main()
