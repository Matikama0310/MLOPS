from typing import List, Union
from pathlib import Path
from contextlib import asynccontextmanager

import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator

# Read run_id from run_id.txt next to this file
RUN_ID_PATH = Path(__file__).resolve().parent / "run_id.txt"

def read_run_id(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read run_id from {path}: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup using MLflow run id
    run_id = read_run_id(RUN_ID_PATH)
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_uri}: {e}")
    app.state.model = model
    app.state.model_version = run_id
    yield
    # optional cleanup
    app.state.model = None

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    # Accept either a single feature list [f1, f2,...] or a batch [[...], [...]]
    features: Union[List[float], List[List[float]]]

    @validator("features", pre=True)
    def ensure_2d(cls, v):
        if not v:
            raise ValueError("features must be a non-empty list")
        # if first element is a list/tuple, assume batch already
        if isinstance(v[0], (list, tuple)):
            return v
        # otherwise wrap single sample into a batch
        return [v]

class PredictResponse(BaseModel):
    predictions: List[float]
    model_version: str

@app.get("/")
def root():
    return {"message": "Model prediction API. Use POST /predict with JSON {\"features\": [...]}"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to numpy array (2D)
    try:
        X = np.array(request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature format: {e}")

    try:
        preds = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Ensure list of native types
    preds_list = np.array(preds).ravel().tolist()
    return PredictResponse(predictions=preds_list, model_version=getattr(app.state, "model_version", "unknown"))

if __name__ == "__main__":
    # Run locally with Uvicorn on port 9696
    uvicorn.run("c:\Users\dell\OneDrive\Documentos\MLOPS\1-application\app:app", host="0.0.0.0", port=9696, reload=False)   