# Mental Health Treatment Predictor

Interactive MLOps demo that trains a mental-health treatment classifier, serves it via a FastAPI backend, and exposes a Streamlit UI. The project is organized so you can iterate on notebooks locally, automate training and tracking with MLflow, deploy inference to Render, and point a managed Streamlit app at the hosted API.

## Project Layout

```
.
├── 0-try/              # Experiment notebooks + exploratory code
├── 1-application/      # Local app + artifacts from early iterations
├── 2-monitoring/       # Monitoring scripts/dashboards
├── 3-cicd/             # Production-grade training, FastAPI service, Streamlit UI
├── data/               # Raw survey data (Kaggle)
└── requirements.txt    # Convenience meta-requirements
```

Day-to-day work happens in `3-cicd/`, which contains:

- `train.py` – trains the production pipeline, logs to MLflow, exports artifacts to `3-cicd/models/model` and captures the latest `run_id`.
- `app.py` – FastAPI inference service that loads the saved pipeline (prefers local artifacts, falls back to MLflow or a dummy model).
- `streamlit_app.py` – end-user interface; calls the FastAPI `/predict` endpoint.

## Prerequisites

- Python 3.10+
- Kaggle CLI configured with API credentials (for training – dataset `osmi/mental-health-in-tech-survey`).
- Optional: MLflow tracking server. By default the scripts use `file:./mlruns`.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

The root `requirements.txt` simply aggregates each sub-project’s pinned dependencies; you can also install the per-folder `requirements.txt` if you want a minimal environment.

## Training Pipeline (optional before deployment)

1. Generate `3-cicd/best_config.json` by running the notebook in `0-try/experiment.ipynb` or by supplying your own hyperparameters.
2. (Optional but recommended) store your Kaggle credentials as GitHub secrets so CI/CD workflows can download the dataset:
   - In your GitHub repository, navigate to **Settings → Secrets and variables → Actions → New repository secret**.
   - Create secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY` using the values from your local `~/.kaggle/kaggle.json`.
   - Reference them in your workflow file as `${{ secrets.KAGGLE_USERNAME }}` and `${{ secrets.KAGGLE_KEY }}` when running `python train.py`.
3. Start running the MLFlow server
```powershell
mlflow server --host 127.0.0.1 --port 5000
```
Open http://localhost:5000 (leave running).
4. From `3-cicd/`, run:
   ```bash
   python train.py
   ```
   - Downloads Kaggle data, cleans it, trains the configured model, logs metrics/artifacts to MLflow.
   - Writes `run_id.txt`, `training_schema.json`, and copies the exported model to `3-cicd/models/model` (an MLflow pyfunc directory).
5. Keep `models/model` under version control (or publish it to object storage) so the inference service can load it without MLflow connectivity.

## Running Locally

### FastAPI service

```bash
py app.py
```

- Uses the local `models/model` artifact by default. Override with `MODEL_URI` or `RUN_ID` if you want to pull directly from MLflow.
- Health check: `GET http://localhost:8000/health`
- Prediction: `POST http://localhost:8000/predict`

### Streamlit UI

```bash
cd 3-cicd
streamlit run streamlit_app.py
```

The UI expects the API at `API_URL = "http://localhost:8000"` when running locally. Update the constant or set an environment variable/Streamlit secret as described below when you deploy.

## Deploying the FastAPI Backend to Render

1. Push this repository to GitHub (or another git provider Render supports).
2. In Render, create a **New Web Service** and connect it to the repo.
3. Configure the service:
   - **Root directory**: `3-cicd`
   - **Environment**: Python 3.10+
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn app:app --host 0.0.0.0 --port 10000`
4. Environment variables (adjust as needed):
   - `PORT=10000` (Render injects this automatically; match the start command)
   - `MODEL_URI=./model` (if you committed `models/model` inside `3-cicd` and added it to the Docker/Render context)
   - `RUN_ID` or `MLFLOW_TRACKING_URI` if you prefer loading directly from MLflow
   - `ALLOW_DUMMY=0` to fail fast if the real model can’t load
5. Include the `models/model` directory in the repo or fetch it during build (e.g., via `aws s3 cp ...` if you’re storing artifacts externally).
6. Deploy; the public URL will look like `https://your-service.onrender.com`. Test `/health` before pointing clients at it.

> **Tip:** If your model artifacts are large, enable the Render background worker or use persistent disks; otherwise keep the artifact small enough to store in the repo.

## Deploying the Streamlit App

You can keep the Streamlit UI in the same repo and host it on [Streamlit Community Cloud](https://streamlit.io/cloud) or any container platform.

### Streamlit Community Cloud

1. Sign in at share.streamlit.io and create a new app.
2. Select the same GitHub repo and point the app to `3-cicd/streamlit_app.py` (branch `main` or whichever branch you deploy from).
3. Under **Advanced settings → Secrets**, add:
   ```toml
   API_URL = "https://your-service.onrender.com"
   ```
   Then modify `streamlit_app.py` to read `st.secrets.get("API_URL", API_URL)` if you want to override the default without editing code.
4. Deploy. The Streamlit app will call the Render backend for predictions.

### Custom Streamlit Hosting

If you use another platform (Docker, Render, Azure, etc.), run:

```bash
streamlit run 3-cicd/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Set an environment variable before launch:

```bash
export API_URL=https://your-service.onrender.com
```

Update the code to read `API_URL = os.getenv("API_URL", "http://localhost:8000")` if you need runtime configurability.

## Linking Streamlit to Render

1. Deploy the FastAPI backend on Render and verify the public URL.
2. Configure the Streamlit app (local or cloud) so that `API_URL` matches the Render URL.
3. Exercise the UI: submit the form, watch the “Connected to prediction service” banner, and confirm you receive probability outputs rather than only binary labels.

## Monitoring (optional)

`2-monitoring/` contains scripts (`monitor.py`, `simulate.py`, etc.) that can replay predictions, log drifts, and export reports. Update the `API_URL` inside those scripts to the Render endpoint before running them in production.

## Troubleshooting

- **API not ready:** Check `/health` to confirm the model loaded. If you see `"run_id": "dummy"`, ensure `models/model` exists or set `RUN_ID` for MLflow retrieval.
- **100 % / 0 % probabilities:** Make sure the saved pipeline’s final estimator implements `predict_proba`. The provided LogisticRegression/RandomForest/XGBoost configs do.
- **Streamlit errors about CORS or HTTPS:** Render serves HTTPS by default; Streamlit Cloud also uses HTTPS. No extra configuration is needed, but ensure `API_URL` starts with `https://` when pointing to Render.

Happy predicting!

