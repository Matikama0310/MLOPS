# 2-monitoring: Model Health & Data Quality

This folder provides monitoring tools to validate data quality, track model health, and simulate traffic against the deployed inference API.

## Contents
- `train.py`: Re-trains the model with monitoring-focused instrumentation, mirroring the production training pipeline.
- `monitor.py`: Generates monitoring reports (e.g., data drift/health checks) using the training schema and current data.
- `simulate.py`: Sends sample requests to a running API instance to exercise prediction and logging flows.
- `app.py`: Monitoring-ready FastAPI app variant that loads the trained model and exposes inference plus health endpoints.
- `training_schema.json`: Schema produced during training that lists expected features and types.
- `health_monitoring_report.html`: Example output from running `monitor.py`.
- `best_config.json`, `run_id.txt`, `test_api.py`: Supporting artifacts and utilities shared with the training/inference flow.

## How to use
1. Train or reuse a model so that `run_id.txt` and `training_schema.json` are present.
2. Start the monitoring-ready API (similar to the application stage) if you need a local endpoint:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
3. Generate a monitoring report from local data (for example, a CSV with current predictions):
   ```bash
   python monitor.py --input <path_to_data.csv> --output health_monitoring_report.html
   ```
4. Use `simulate.py` or `test_api.py` to send sample payloads and verify API health.
