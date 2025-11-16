# 3-cicd: Packaging & Deployment

This folder contains assets for containerizing and automating the deployment of the training and inference services.

## Contents
- `Dockerfile`: Builds an image that includes the training and FastAPI inference components.
- `requirements.txt`: Python dependencies needed by the containerized app.
- `app.py`: Inference service entrypoint (FastAPI) bundled for deployment.
- `train.py`: Training entrypoint used inside CI/CD jobs or during container builds.
- `training_schema.json`, `best_config.json`, `run_id.txt`: Artifacts consumed by the training/inference steps.
- `test_api.py`: Smoke test script for the deployed API.

## How to use
1. Ensure `best_config.json` and any MLflow credentials/run IDs are available before building.
2. Build the image from the repository root:
   ```bash
   docker build -f 3-cicd/Dockerfile -t mental-health-mlops .
   ```
3. Run the container locally (exposing the FastAPI service):
   ```bash
   docker run -p 8000:8000 -e RUN_ID=<mlflow_run_id> mental-health-mlops
   ```
4. Integrate these steps in your CI/CD pipeline to retrain and redeploy automatically when configurations or code change.
