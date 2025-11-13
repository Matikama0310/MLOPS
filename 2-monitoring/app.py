# -*- coding: utf-8 -*-
"""
FastAPI inference service for the Mental Health survey classifier.

Fixes included:
- Replace pandas NA with numpy.nan before inference.
- Add missing expected columns with np.nan (not pd.NA).
- Coerce numeric features to numeric dtype; convert nullable booleans to float.
- Load numeric/categorical feature lists from training_schema.json.
"""

import os
import re
import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts

# -----------------------------
# Config (env overridable)
# -----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RUN_ID_FILE = os.getenv("RUN_ID_FILE", "run_id.txt")
RUN_ID_ENV = os.getenv("RUN_ID", "").strip()

# Globals loaded at startup
RUN_ID: str = ""
model = None
schema: Dict[str, Any] = {}  # training_schema.json contents
expected_columns: List[str] = []  # columns the pipeline expects
numeric_features_g: List[str] = []
categorical_features_g: List[str] = []
target_col: str = "treatment"


# -----------------------------
# Helpers (mirror training)
# -----------------------------
def clean_gender(gen: object) -> str:
    s = str(gen).strip().lower()
    s = re.sub(r"[\\W_]+", " ", s).strip()
    if s in {"m", "male", "man", "make", "mal", "malr", "msle", "masc", "mail", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "femake", "femail", "femme", "girl"}:
        return "Female"
    return "Other"


def _load_run_id() -> str:
    if RUN_ID_ENV:
        return RUN_ID_ENV
    if not os.path.exists(RUN_ID_FILE):
        raise FileNotFoundError(f"RUN ID file not found: {RUN_ID_FILE}")
    return open(RUN_ID_FILE, "r", encoding="utf-8").read().strip()


def _load_schema(run_id: str) -> Dict[str, Any]:
    """Download and parse training_schema.json from the run; return empty dict if unavailable."""
    try:
        local_path = download_artifacts(artifact_uri=f"runs:/{run_id}/training_schema.json")
        with open(local_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _compute_expected_columns(schema_json: Dict[str, Any]) -> List[str]:
    if not schema_json:
        return []
    retained = schema_json.get("retained_features", "ALL_COLUMNS")
    if isinstance(retained, list) and retained:
        return retained
    # Fallback: union of numeric + categorical features logged at train time
    num = schema_json.get("numeric_features", []) or []
    cat = schema_json.get("categorical_features", []) or []
    return list(dict.fromkeys([*num, *cat]))


def _extract_num_cat(schema_json: Dict[str, Any]) -> (List[str], List[str]):
    if not schema_json:
        return [], []
    num = schema_json.get("numeric_features") or []
    cat = schema_json.get("categorical_features") or []
    num = list(num) if isinstance(num, (list, tuple)) else []
    cat = list(cat) if isinstance(cat, (list, tuple)) else []
    return num, cat


def _preprocess_payload(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Convert incoming JSON to a DataFrame, clean Gender, select expected columns, and drop target if present."""
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be an object or a list of objects.")

    df = pd.DataFrame(rows)

    # Normalize missing values: replace pandas NA with numpy.nan
    df = df.replace({pd.NA: np.nan})

    # Remove target if present
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Apply same gender normalization
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(clean_gender)

    # Add any missing expected columns as np.nan so imputers can handle them
    if expected_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        # Order columns for consistency
        df = df[expected_columns]

    # Coerce numeric features to numeric dtype (avoid nullable boolean issues)
    num_cols = set(numeric_features_g or [])
    if num_cols:
        for c in (set(df.columns) & num_cols):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Convert pandas nullable boolean to float
            if str(df[c].dtype) == "boolean":
                df[c] = df[c].astype("float")

    # Ensure any remaining pd.NA are numpy.nan to avoid ambiguous truth values
    df = df.replace({pd.NA: np.nan})

    return df


# -----------------------------
# Pydantic models
# -----------------------------
class PredictResponse(BaseModel):
    model_version: str = Field(..., description="MLflow run id")
    predictions: List[int] = Field(..., description="Predicted class labels (1 = Yes, 0 = No)")
    probabilities: List[float] = Field(..., description="Probability for class 1 (treatment=Yes)")


# -----------------------------
# App lifecycle
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model, schema, expected_columns, numeric_features_g, categorical_features_g

    # 1) read run id & set mlflow
    RUN_ID = _load_run_id()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # 2) load model once
    model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/model")

    # 3) load schema to know expected columns and feature types
    schema = _load_schema(RUN_ID)
    expected_columns = _compute_expected_columns(schema)
    numeric_features_g, categorical_features_g = _extract_num_cat(schema)
    if expected_columns:
        print(f"[startup] Using {len(expected_columns)} expected columns from training_schema.json")
    else:
        print("[startup] No schema found; using columns provided by requests")
    if numeric_features_g or categorical_features_g:
        print(f"[startup] numeric_features: {len(numeric_features_g)} | categorical_features: {len(categorical_features_g)}")

    yield

# Create app
app = FastAPI(
    title="Mental Health Treatment Predictor",
    description="Predicts treatment need (Yes/No) from the mental health survey using the trained sklearn Pipeline logged to MLflow.",
    version="1.1.0",
    lifespan=lifespan,
)


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Mental Health Prediction API. See /docs for usage.", "run_id": RUN_ID}


@app.get("/health")
def health():
    ok = model is not None
    return {"status": "ok" if ok else "not_ready", "run_id": RUN_ID, "has_schema": bool(schema)}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: Dict[str, Any] = Body(..., example={
    "Age": 33,
    "Gender": "Male",
    "self_employed": "No",
    "family_history": "No",
    "benefits": "Yes",
    "work_interfere": "Sometimes",
    "remote_work": "Yes",
    "no_employees": "6-25",
    "tech_company": "Yes",
    "anonymity": "Yes"
})):
    """Single prediction: JSON object with survey fields (except 'treatment')."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    df = _preprocess_payload(payload)
    try:
        y_prob = getattr(model, "predict_proba", None)
        if y_prob is not None:
            proba = model.predict_proba(df)[:, 1].tolist()
            preds = (pd.Series(proba) >= 0.5).astype(int).tolist()
        else:
            preds = model.predict(df).tolist()
            proba = [float(p) for p in preds]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return PredictResponse(model_version=RUN_ID, predictions=preds, probabilities=proba)


@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(payload: List[Dict[str, Any]] = Body(..., example=[
    {"Age": 28, "Gender": "F", "remote_work": "No", "tech_company": "Yes"},
    {"Age": 41, "Gender": "Other", "remote_work": "Yes", "tech_company": "No"}
])):
    """Batch prediction: JSON array of objects with survey fields (except 'treatment')."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    df = _preprocess_payload(payload)
    try:
        y_prob = getattr(model, "predict_proba", None)
        if y_prob is not None:
            proba = model.predict_proba(df)[:, 1].tolist()
            preds = (pd.Series(proba) >= 0.5).astype(int).tolist()
        else:
            preds = model.predict(df).tolist()
            proba = [float(p) for p in preds]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return PredictResponse(model_version=RUN_ID, predictions=preds, probabilities=proba)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
