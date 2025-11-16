"""
External API tests for the Mental Health Prediction FastAPI service.

Tests use only SELECTED features from best_config.json
"""

import sys
import json
import requests

BASE_URL = "http://localhost:8000"


def load_selected_features():
    """Load selected features from best_config.json"""
    try:
        with open('best_config.json', 'r') as f:
            config = json.load(f)
            return config['features']['selected']
    except:
        # Fallback to features we know are selected
        return [
            "work_interfere", "family_history", "care_options",
            "supervisor", "leave", "Gender", "obs_consequence",
            "mental_health_interview", "no_employees", "coworkers",
            "phys_health_interview", "self_employed",
            "mental_health_consequence"
        ]


def _assert_probabilities(proba):
    assert isinstance(proba, list), "probabilities should be a list"
    assert len(proba) >= 1, "probabilities should not be empty"
    for p in proba:
        assert isinstance(p, (float, int)), "each probability must be numeric"
        assert 0.0 <= float(p) <= 1.0, f"probability out of bounds: {p}"


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Unexpected: {resp.status_code}"
    data = resp.json()
    assert data.get("status") in {"ok", "not_ready"}
    if data.get("status") == "ok":
        assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5


def test_predict_single():
    """Test with only SELECTED features"""
    payload = {
        "Gender": "Male",
        "family_history": "Yes",
        "work_interfere": "Sometimes",
        "care_options": "Yes",
        "supervisor": "Yes"
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    data = resp.json()

    assert "model_version" in data
    assert "predictions" in data and "probabilities" in data

    preds = data["predictions"]
    proba = data["probabilities"]
    assert isinstance(preds, list) and len(preds) == 1
    assert preds[0] in (0, 1), "prediction must be 0 or 1"
    _assert_probabilities(proba)
    
    print(f"✓ Prediction: {preds[0]} (prob: {proba[0]:.3f})")


def test_predict_batch():
    """Test batch with only SELECTED features"""
    payload = [
        {
            "Gender": "Female",
            "family_history": "No",
            "work_interfere": "Rarely",
            "care_options": "No",
            "supervisor": "No"
        },
        {
            "Gender": "Male",
            "family_history": "Yes",
            "work_interfere": "Often",
            "care_options": "Yes",
            "supervisor": "Yes"
        },
        {
            "Gender": "Other",
            "family_history": "No",
            "work_interfere": "Sometimes",
            "care_options": "Don't know",
            "supervisor": "Some of them"
        }
    ]
    resp = requests.post(f"{BASE_URL}/predict_batch", json=payload)
    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    data = resp.json()

    assert "model_version" in data
    assert "predictions" in data and "probabilities" in data

    preds = data["predictions"]
    proba = data["probabilities"]
    assert isinstance(preds, list) and len(preds) == len(payload)
    assert all(p in (0, 1) for p in preds), "all predictions must be 0 or 1"
    _assert_probabilities(proba)
    
    print(f"✓ Batch predictions: {preds}")


def test_with_minimal_features():
    """Test with very minimal payload (server should handle missing features)"""
    payload = {
        "Gender": "Male",
        "family_history": "Yes"
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    data = resp.json()
    assert "predictions" in data
    print(f"✓ Minimal payload works: {data['predictions'][0]}")


if __name__ == "__main__":
    failures = 0
    
    print("\n🧪 Testing Mental Health API...")
    print(f"Target: {BASE_URL}\n")
    
    try:
        test_health_endpoint()
        print("✓ /health")
    except AssertionError as e:
        failures += 1
        print(f"✗ /health: {e}", file=sys.stderr)

    try:
        test_predict_single()
        print("✓ /predict (single)")
    except AssertionError as e:
        failures += 1
        print(f"✗ /predict (single): {e}", file=sys.stderr)

    try:
        test_predict_batch()
        print("✓ /predict_batch")
    except AssertionError as e:
        failures += 1
        print(f"✗ /predict_batch: {e}", file=sys.stderr)
    
    try:
        test_with_minimal_features()
        print("✓ /predict (minimal)")
    except AssertionError as e:
        failures += 1
        print(f"✗ /predict (minimal): {e}", file=sys.stderr)

    if failures == 0:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ {failures} test(s) failed")
    
    sys.exit(1 if failures else 0)