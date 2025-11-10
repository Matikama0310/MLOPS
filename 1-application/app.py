from typing import List, Union, Any
from pathlib import Path
from contextlib import asynccontextmanager
import logging
import os  # <- added

import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

# Read run_id from artifacts/run_id.txt only
BASE_DIR = Path(__file__).resolve().parent
RUN_ID_PATH = BASE_DIR / "artifacts" / "run_id.txt"

# Prefer MLFLOW_TRACKING_URI from env, otherwise same default used in train.py
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# add logger
logger = logging.getLogger("ml_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

def read_run_id(path: Path) -> str:
    """Read a single run_id file; fail loudly if missing or empty."""
    if not path.exists():
        raise RuntimeError(f"run_id file not found at expected location: {path}")
    run_id = path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise RuntimeError(f"run_id file {path} is empty")
    logger.info(f"Using run_id from: {path}")
    return run_id

def _try_load_model_from_run(run_id: str):
    """
    Try multiple likely artifact names for the model saved in the run.
    Returns tuple (model, model_uri) or (None, None) if none succeeded.
    """
    candidate_artifact_names = [
        "model",
        "final_model",
        "model/model",
        "final_model/model",
        "artifacts/model",
        "artifacts/final_model",
    ]
    for name in candidate_artifact_names:
        model_uri = f"runs:/{run_id}/{name}"
        try:
            logger.info(f"Attempting to load model from {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model from {model_uri}")
            return model, model_uri
        except Exception as e:
            logger.debug(f"Could not load model from {model_uri}: {e}")
    return None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure app uses same MLflow tracking server as training
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to {mlflow.get_tracking_uri()}")
    except Exception as e:
        logger.exception(f"Could not set MLflow tracking URI: {e}")

    # Load model once at startup using MLflow run id (from artifacts/run_id.txt)
    try:
        run_id = read_run_id(RUN_ID_PATH)
        model, model_uri = _try_load_model_from_run(run_id)
        if model is not None:
            app.state.model = model
            app.state.model_version = run_id
            app.state.model_uri = model_uri
        else:
            logger.warning(f"No model artifact found for run {run_id}; checked common artifact names.")
            app.state.model = None
            app.state.model_version = "unknown"
    except Exception as e:
        logger.exception(f"Failed to read run_id or load model: {e}")
        app.state.model = None
        app.state.model_version = "unknown"

    yield
    # optional cleanup
    app.state.model = None

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    # Accept either a single feature list [f1, f2,...] or a batch [[...], [...]]
    features: Any

    @validator("features", pre=True)
    def ensure_2d(cls, v):
        if v is None:
            raise ValueError("features must be provided and non-empty")
        if isinstance(v, dict):
            raise ValueError("features must be a list or list of lists, not an object/dict")
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError("features must be a non-empty list")
            first = v[0]
            # batch of samples
            if isinstance(first, (list, tuple)):
                # convert all values to floats where possible
                converted = []
                for row in v:
                    if not isinstance(row, (list, tuple)):
                        raise ValueError("All rows must be lists of numeric values")
                    try:
                        converted.append([float(x) for x in row])
                    except Exception as e:
                        raise ValueError(f"Could not convert row to floats: {row} ({e})")
                return converted
            # single sample -> wrap
            try:
                return [[float(x) for x in v]]
            except Exception as e:
                raise ValueError(f"Could not convert features to floats: {v} ({e})")
        raise ValueError("features must be a list or list of lists")

class PredictResponse(BaseModel):
    predictions: List[Union[float, str]]
    model_version: str

@app.get("/")
def root():
    return {"message": "Model prediction API. Use POST /predict with JSON {\"features\": [...]}"}

@app.get("/health")
def health():
    model_loaded = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": model_loaded, "model_version": getattr(app.state, "model_version", "unknown")}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to numpy array (2D)
    try:
        X = np.array(request.features, dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature format: {e}")

    try:
        preds = model.predict(X)
    except Exception as e:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Ensure list of native types, try casting to float otherwise string-ify
    preds_list = []
    for p in np.array(preds).ravel().tolist():
        try:
            preds_list.append(float(p))
        except Exception:
            preds_list.append(str(p))

    return PredictResponse(predictions=preds_list, model_version=getattr(app.state, "model_version", "unknown"))

if __name__ == "__main__":
    # Run locally with Uvicorn on port 9696
    uvicorn.run(app, host="0.0.0.0", port=9696, reload=False)