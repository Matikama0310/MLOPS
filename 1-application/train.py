"""
Trainable Python application adapted from test.ipynb.

Usage (from project folder):
    python train.py

Adjust DATA_PATH and MLFLOW_TRACKING_URI as needed.
"""
from __future__ import annotations

import os
import re
import json
import inspect  # Add this import
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve
)

import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "mental-health-tech-prediction"
DATA_PATH = os.path.join("..", "data", "raw", "survey.csv")
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data shape: {df.shape}")
    return df


def clean_gender(gen: Any) -> str:
    s = str(gen).strip().lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    male = {"m", "male", "man", "make", "mal", "malr", "msle", "masc", "mail", "boy"}
    female = {"f", "female", "woman", "femake", "femail", "femme", "girl"}
    if s in male:
        return "Male"
    if s in female:
        return "Female"
    return "Other"


def prepare_features(df: pd.DataFrame, target_col: str = "treatment") -> Tuple[pd.DataFrame, pd.Series]:
    features_to_drop = ['Timestamp', 'Country', 'state', 'comments', target_col]
    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[c for c in features_to_drop if c in df.columns])
    if "Gender" in X.columns:
        X["Gender"] = X["Gender"].apply(clean_gender)
    return X, y


def make_onehot_encoder(**kwargs):
    """Create OneHotEncoder with consistent sparse behavior across sklearn versions."""
    try:
        # Try newer sklearn first
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        # Fallback for older sklearn
        return OneHotEncoder(sparse=False, **kwargs)


def get_feature_names_out_compat(preproc: ColumnTransformer, input_columns):
    """Try the sklearn-provided get_feature_names_out, otherwise build names from transformers_."""
    try:
        # try to use built-in method (works on recent sklearn)
        return preproc.get_feature_names_out(input_columns)
    except Exception:
        names = []
        # preproc.transformers_ is a list of (name, transformer, columns)
        for name, trans, cols in getattr(preproc, "transformers_", []):
            if trans == 'drop' or trans == 'passthrough':
                continue
            # handle Pipeline wrapper
            transformer = trans
            if hasattr(trans, 'named_steps'):
                # assume last step produces feature names (e.g., OneHotEncoder)
                last_step = list(trans.named_steps.values())[-1]
                transformer = last_step
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    # some encoders accept input_features param, some don't
                    out = transformer.get_feature_names_out(cols)
                    names.extend(out.tolist() if hasattr(out, 'tolist') else list(out))
                    continue
                except Exception:
                    pass
            # try to build names for OneHotEncoder-like objects with categories_
            if hasattr(transformer, "categories_"):
                for col, cats in zip(cols, transformer.categories_):
                    for cat in cats:
                        names.append(f"{col}_{cat}")
                continue
            # fallback: keep original column names
            if isinstance(cols, (list, tuple, np.ndarray)):
                names.extend([c for c in cols])
            else:
                names.append(str(cols))
        return np.array(names)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_onehot_encoder(handle_unknown='ignore'))
    ])
    
    # Add verbose_feature_names_out=False to get simpler feature names
    preprocessor = ColumnTransformer(
        [
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor


def get_model():
    """Return the best model configuration (XGBoost) based on notebook results"""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )


def train_model(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer) -> Pipeline:
    """Train the model without validation"""
    model = get_model()
    pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])
    print("Training model...")
    pipeline.fit(X, y)
    return pipeline


def main(args):
    df = load_data(args.data_path)
    X, y = prepare_features(df, target_col=args.target)
    preproc = build_preprocessor(X)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="final_model") as run:
        pipeline = train_model(X, y, preproc)
        
        # Evaluate on holdout
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        metrics = {
            'final_accuracy': float(accuracy_score(y_te, y_pred)),
            'final_precision': float(precision_score(y_te, y_pred, zero_division=0)),
            'final_recall': float(recall_score(y_te, y_pred, zero_division=0)),
            'final_f1': float(f1_score(y_te, y_pred, zero_division=0)),
            'final_roc_auc': float(roc_auc_score(y_te, y_prob))
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "final_model")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC={metrics['final_roc_auc']:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC - Final Model")
        plt.legend()
        roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)

        print(f"Final run logged: {run.info.run_id}")
        print("Done. Artifacts in:", ARTIFACTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=DATA_PATH)
    parser.add_argument("--target", type=str, default="treatment")
    args = parser.parse_args()
    main(args)