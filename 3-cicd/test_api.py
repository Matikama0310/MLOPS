"""
External API tests for the Mental Health Prediction FastAPI service.

Prereqs:
- The server is running (e.g., `python app_mental.py` on port 9696).
- The model run_id.txt points to a trained run with a logged sklearn Pipeline and training_schema.json.

Run with:
    python -m pytest -q test_api_mental.py
or simply:
    python test_api_mental.py
"""

import sys
import requests

BASE_URL = "http://localhost:9696"


def _assert_probabilities(proba):
    assert isinstance(proba, list), "probabilities should be a list"
    assert len(proba) >= 1, "probabilities should not be empty"
    for p in proba:
        assert isinstance(p, (float, int)), "each probability must be numeric"
        assert 0.0 <= float(p) <= 1.0, f"probability out of bounds: {p}"


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text}"
    data = resp.json()
    assert data.get("status") in {"ok", "not_ready"}
    # If ready, run_id should be a non-trivial string
    if data.get("status") == "ok":
        assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5


def test_predict_single():
    # Minimal payload (server will add missing expected columns as NA and impute)
    payload = {
        "Age": 33,
        "Gender": "Male",
        "remote_work": "Yes",
        "tech_company": "Yes"
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text}"
    data = resp.json()

    # Validate structure
    assert "model_version" in data
    assert "predictions" in data and "probabilities" in data

    # Sanity checks
    preds = data["predictions"]
    proba = data["probabilities"]
    assert isinstance(preds, list) and len(preds) == 1
    assert preds[0] in (0, 1), "prediction must be 0 or 1"
    _assert_probabilities(proba)


def test_predict_batch():
    payload = [
        {"Age": 28, "Gender": "F", "remote_work": "No", "tech_company": "Yes"},
        {"Age": 41, "Gender": "Other", "remote_work": "Yes", "tech_company": "No"},
        {"Age": 22, "Gender": "Male", "remote_work": "No", "tech_company": "Yes"}
    ]
    resp = requests.post(f"{BASE_URL}/predict_batch", json=payload)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} body={resp.text}"
    data = resp.json()

    # Validate structure
    assert "model_version" in data
    assert "predictions" in data and "probabilities" in data

    # Sanity checks
    preds = data["predictions"]
    proba = data["probabilities"]
    assert isinstance(preds, list) and len(preds) == len(payload)
    assert all(p in (0, 1) for p in preds), "all predictions must be 0 or 1"
    _assert_probabilities(proba)


if __name__ == "__main__":
    # lightweight runner
    failures = 0
    try:
        test_health_endpoint()
        print("✓ /health")
    except AssertionError as e:
        failures += 1
        print("✗ /health:", e, file=sys.stderr)

    try:
        test_predict_single()
        print("✓ /predict (single)")
    except AssertionError as e:
        failures += 1
        print("✗ /predict (single):", e, file=sys.stderr)

    try:
        test_predict_batch()
        print("✓ /predict_batch")
    except AssertionError as e:
        failures += 1
        print("✗ /predict_batch:", e, file=sys.stderr)

    sys.exit(1 if failures else 0)
