# 1-application: Training & Inference Service

This stage turns the experimental results into a production-ready model and exposes it through a FastAPI service.

## Contents
- `train.py`: Trains the final model using the configuration exported from `0-try/experiment.ipynb`, logs artifacts to MLflow, and writes the resulting `run_id.txt` and schema files.
- `best_config.json`: Expected configuration produced by the experimentation step (consumed by `train.py`).
- `artifacts/run_id.txt`: Example run identifier output after training.
- `app.py`: FastAPI application that loads the trained model (locally or from MLflow), validates payloads against the training schema, and serves `/predict`, `/health`, and example endpoints.
- `test_api.py`: Simple test client for hitting the inference endpoints.

## How to use
1. Ensure `best_config.json` exists (created by the notebook in `0-try`).
2. Train the production model:
   ```bash
   cd 1-application
   python train.py
   ```
3. Start the inference API (after training or providing a `RUN_ID`/`MODEL_URI`):
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
4. Exercise the endpoints locally using `test_api.py` or any HTTP client.
