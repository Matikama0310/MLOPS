# -*- coding: utf-8 -*-
"""
FastAPI inference service for the Mental Health survey classifier.

- Prefer LOCAL model directory (MODEL_URI, default "./model") baked into the image.
- Fall back to MLflow only if RUN_ID is set.
- If no model is found, load a DUMMY model (disable with ALLOW_DUMMY=0).
- Startup never crashes; /health reports readiness as 'ok' or 'not_ready'.
"""

import os
import re
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union, Optional, Tuple

import pandas as pd
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field

import mlflow
import mlflow.pyfunc
from mlflow.artifacts import download_artifacts

# -----------------------------
# Tiny dummy model (for tests / no-artifacts mode)
# -----------------------------
class _DummyModel:
    """Minimal model that always predicts 0 with prob 0.0.
    Used only when no real model is available, so tests can exercise endpoints."""
    def predict_proba(self, X):
        import numpy as _np
        n = len(X)
        return _np.c_[_np.ones(n, dtype=float), _np.zeros(n, dtype=float)]
    def predict(self, X):
        import numpy as _np
        return _np.zeros(len(X), dtype=int)

# -----------------------------
# Config (env overridable)
# -----------------------------
PORT = int(os.getenv("PORT", "8000"))

MODEL_URI = os.getenv("MODEL_URI", "./model")
SCHEMA_PATH = os.getenv("SCHEMA_PATH", "./training_schema.json")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
RUN_ID_FILE = os.getenv("RUN_ID_FILE", "run_id.txt")
RUN_ID_ENV = os.getenv("RUN_ID", "").strip()

ALLOW_DUMMY = os.getenv("ALLOW_DUMMY", "1")  # set to "0" to disable dummy fallback

# Globals
RUN_ID: str = ""
model = None
schema: Dict[str, Any] = {}
expected_columns: List[str] = []
numeric_features_g: List[str] = []
categorical_features_g: List[str] = []
target_col: str = "treatment"

# -----------------------------
# Helpers (mirror training)
# -----------------------------
def clean_gender(gen: object) -> str:
    s = str(gen).strip().lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    if s in {"m", "male", "man", "make", "mal", "malr", "msle", "masc", "mail", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "femake", "femail", "femme", "girl"}:
        return "Female"
    return "Other"

def _load_run_id() -> str:
    if RUN_ID_ENV:
        return RUN_ID_ENV
    p = Path(RUN_ID_FILE)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""

def _load_schema_from_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _load_schema_from_mlflow(run_id: str) -> Dict[str, Any]:
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
    num = schema_json.get("numeric_features", []) or []
    cat = schema_json.get("categorical_features", []) or []
    return list(dict.fromkeys([*num, *cat]))

def _extract_num_cat(schema_json: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    if not schema_json:
        return [], []
    num = schema_json.get("numeric_features") or []
    cat = schema_json.get("categorical_features") or []
    num = list(num) if isinstance(num, (list, tuple)) else []
    cat = list(cat) if isinstance(cat, (list, tuple)) else []
    return num, cat

def _preprocess_payload(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise HTTPException(status_code=400, detail="Payload must be an object or a list of objects.")

    df = pd.DataFrame(rows)

    # Normalize missing values / booleans
    df = df.replace({pd.NA: np.nan})
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(clean_gender)

    if expected_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df = df[expected_columns]

    num_cols = set(numeric_features_g or [])
    if num_cols:
        for c in (set(df.columns) & num_cols):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if str(df[c].dtype) == "boolean":
                df[c] = df[c].astype("float")

    df = df.replace({pd.NA: np.nan})
    return df

# -----------------------------
# Pydantic models
# -----------------------------
class PredictResponse(BaseModel):
    model_version: str = Field(..., description="Model version (RUN_ID or 'local'/'dummy')")
    predictions: List[int] = Field(..., description="Predicted class labels (1 = Yes, 0 = No)")
    probabilities: List[float] = Field(..., description="Probability for class 1 (treatment=Yes)")

# -----------------------------
# Model loading strategies
# -----------------------------
def _try_load_local_model() -> Optional[mlflow.pyfunc.PyFuncModel]:
    path = Path(MODEL_URI)
    if not path.exists():
        return None
    try:
        m = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"[startup] Loaded LOCAL model from {MODEL_URI}")
        return m
    except Exception as e:
        print(f"[startup] Local model load failed from {MODEL_URI}: {e}")
        return None

def _try_load_mlflow_model(run_id: str) -> Optional[mlflow.pyfunc.PyFuncModel]:
    if not run_id:
        return None
    try:
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        m = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        print(f"[startup] Loaded MLflow model from run {run_id}")
        return m
    except Exception as e:
        print(f"[startup] MLflow model load failed for run {run_id}: {e}")
        return None

def _load_schema() -> Dict[str, Any]:
    s = _load_schema_from_file(SCHEMA_PATH)
    if s:
        print(f"[startup] Loaded LOCAL schema from {SCHEMA_PATH}")
        return s
    if RUN_ID:
        s = _load_schema_from_mlflow(RUN_ID)
        if s:
            print(f"[startup] Loaded schema from MLflow run {RUN_ID}")
            return s
    print("[startup] No schema found; API will accept provided columns")
    return {}

# -----------------------------
# App lifecycle
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model, schema, expected_columns, numeric_features_g, categorical_features_g

    RUN_ID = _load_run_id()

    # Try local -> MLflow (if RUN_ID) -> Dummy
    model = _try_load_local_model()
    if model is None and RUN_ID:
        model = _try_load_mlflow_model(RUN_ID)
    if model is None and ALLOW_DUMMY == "1":
        model = _DummyModel()
        RUN_ID_local = RUN_ID or "dummy"
        print(f"[startup] Using DUMMY model (set ALLOW_DUMMY=0 to disable), run_id={RUN_ID_local}")

    schema = _load_schema()
    expected_columns = _compute_expected_columns(schema)
    numeric_features_g, categorical_features_g = _extract_num_cat(schema)
    if expected_columns:
        print(f"[startup] Using {len(expected_columns)} expected columns from schema")
    if numeric_features_g or categorical_features_g:
        print(f"[startup] numeric_features: {len(numeric_features_g)} | categorical_features: {len(categorical_features_g)}")
    yield

app = FastAPI(
    title="Mental Health Treatment Predictor",
    description="Predicts treatment need (Yes/No) from the mental health survey using a trained sklearn Pipeline.",
    version="1.3.0",
    lifespan=lifespan,
)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "Mental Health Prediction API. See /docs for usage.",
        "run_id": RUN_ID or ("dummy" if isinstance(model, _DummyModel) else "local"),
        "model_uri": MODEL_URI,
        "has_schema": bool(schema),
    }

@app.get("/health")
def health():
    ok = model is not None
    return {"status": "ok" if ok else "not_ready",
            "run_id": RUN_ID or ("dummy" if isinstance(model, _DummyModel) else "local"),
            "has_schema": bool(schema)}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: Dict[str, Any] = Body(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    df = _preprocess_payload(payload)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1].tolist()
            preds = (pd.Series(proba) >= 0.5).astype(int).tolist()
        else:
            preds = list(map(int, model.predict(df).tolist()))
            proba = [float(p) for p in preds]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return PredictResponse(model_version=RUN_ID or ("dummy" if isinstance(model, _DummyModel) else "local"),
                           predictions=preds, probabilities=proba)

@app.post("/predict_batch", response_model=PredictResponse)
def predict_batch(payload: List[Dict[str, Any]] = Body(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    df = _preprocess_payload(payload)
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[:, 1].tolist()
            preds = (pd.Series(proba) >= 0.5).astype(int).tolist()
        else:
            preds = list(map(int, model.predict(df).tolist()))
            proba = [float(p) for p in preds]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return PredictResponse(model_version=RUN_ID or ("dummy" if isinstance(model, _DummyModel) else "local"),
                           predictions=preds, probabilities=proba)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
