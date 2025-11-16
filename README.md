# MLOps Mental Health Survey Pipeline

This repository walks through an end-to-end MLOps workflow for a mental health treatment classifier. It contains four stages that build on each other: experimentation, application training/inference, monitoring, and CI/CD packaging.

## Repository layout
- **0-try** – Initial experiment notebook and helper script for generating the best model configuration.
- **1-application** – Production training pipeline and FastAPI inference service for serving the model.
- **2-monitoring** – Monitoring utilities for model drift/health checks and simulation helpers.
- **3-cicd** – Containerization and CI/CD-ready assets (Dockerfile, requirements, training/inference entrypoints).

## Quickstart
1. Run the notebook in `0-try/experiment.ipynb` (or the helper `create_experiment_notebook.py`) to evaluate models and export `best_config.json`.
2. Train the production model from `1-application/train.py`; the resulting `run_id.txt` and artifacts are used by the inference API (`1-application/app.py`).
3. Use `2-monitoring/monitor.py` and `simulate.py` to validate data, generate monitoring reports, and exercise the deployed API.
4. Package the service with the assets in `3-cicd` (Dockerfile plus entrypoints) for automated builds and deployments.

Each subfolder includes a README that explains its files and how to run them.
