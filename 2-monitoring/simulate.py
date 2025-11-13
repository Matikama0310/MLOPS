"""
Simulate real-time requests to the Mental Health Prediction API and log results.

Usage (server must be running, e.g., `python app_mental.py`):
    python simulate_health.py --data ../data/raw/survey.csv \
                              --log data/health_predictions.csv \
                              --api http://localhost:9696 \
                              --batch 32 --sleep 0.02

What it does:
- Loads rows from the mental-health survey CSV.
- Builds request payloads (excluding the target "treatment").
- Sends either single or batched requests to /predict or /predict_batch.
- Appends predictions (+ probabilities) and optional ground truth to a CSV.

Notes:
- If your CSV contains the target column "treatment" (Yes/No), it will be
  mapped to 1/0 and stored as ground truth for monitoring.
- The script keeps *all* original feature columns (minus the target and a few
  text/meta ones) so drift reports have context.
"""

import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd


# ---------------------------------------------------------------------
# Configuration (defaults can be overridden by CLI args)
# ---------------------------------------------------------------------
DEFAULT_API = "http://localhost:9696"
DEFAULT_LOG = Path("data/health_predictions.csv")
DEFAULT_DATA = Path("../data/raw/survey.csv")

# Meta/target columns to drop from payloads (training removes these too)
DROP_COLS = ["Timestamp", "Country", "state", "comments", "treatment"]


def load_data(path: Path, n_rows: int = 200, shuffle_seed: int = 42) -> pd.DataFrame:
    """Load a slice of the survey and prepare basic target mapping (if present)."""
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")
    df = pd.read_csv(path)

    # Optional ground truth (1/0)
    if "treatment" in df.columns:
        df["treatment"] = df["treatment"].map({"Yes": 1, "No": 0}).astype("Int64")

    # Light cleaning: keep as-is for the server to process (app_mental handles Gender etc.)
    if n_rows and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=shuffle_seed).reset_index(drop=True)
    return df


def to_payload_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Turn DataFrame into a list of JSON-able dicts for POSTing (drop target/meta)."""
    cols = [c for c in df.columns if c not in DROP_COLS]
    # Convert pandas NA to plain None so JSON is valid
    payloads = df[cols].where(pd.notna(df[cols]), None).to_dict(orient="records")
    return payloads


def _post_json(url: str, endpoint: str, json_payload: Any, timeout_s: float = 10.0) -> requests.Response:
    return requests.post(f"{url}{endpoint}", json=json_payload, timeout=timeout_s)


def simulate(api: str, log_path: Path, data_path: Path, n_rows: int, batch_size: int, sleep_s: float) -> pd.DataFrame:
    """Send requests (single or batch) to the API and collect results with timestamps."""
    df = load_data(data_path, n_rows=n_rows)
    payloads = to_payload_rows(df)

    rows = []
    use_batch = batch_size and batch_size > 1
    endpoint = "/predict_batch" if use_batch else "/predict"

    for i in range(0, len(payloads), max(1, batch_size)):
        batch = payloads[i:i+max(1, batch_size)]
        try:
            if use_batch:
                resp = _post_json(api, endpoint, batch)
                resp.raise_for_status()
                out = resp.json()
                preds = out["predictions"]
                prob = out.get("probabilities", [None] * len(preds))
                run_id = out.get("model_version", "")
                ts = pd.Timestamp.utcnow().isoformat()
                # Attach results to original records
                for j, pld in enumerate(batch):
                    row = dict(pld)
                    row.update({
                        "ts": ts,
                        "prediction": preds[j],
                        "probability": float(prob[j]) if prob[j] is not None else None,
                        "model_version": run_id,
                    })
                    # Add ground truth if available
                    if "treatment" in df.columns:
                        row["treatment"] = int(df.iloc[i+j]["treatment"]) if pd.notna(df.iloc[i+j]["treatment"]) else None
                    rows.append(row)
            else:
                pld = batch[0]
                resp = _post_json(api, endpoint, pld)
                resp.raise_for_status()
                out = resp.json()
                preds = out["predictions"] if "predictions" in out else [out]
                prob = out.get("probabilities", [None])
                run_id = out.get("model_version", "")
                ts = pd.Timestamp.utcnow().isoformat()
                row = dict(pld)
                row.update({
                    "ts": ts,
                    "prediction": preds[0] if isinstance(preds, list) else preds,
                    "probability": float(prob[0]) if isinstance(prob, list) and prob else None,
                    "model_version": run_id,
                })
                if "treatment" in df.columns:
                    row["treatment"] = int(df.iloc[i]["treatment"]) if pd.notna(df.iloc[i]["treatment"]) else None
                rows.append(row)

        except Exception as e:
            print(f"⚠️  Request failed at index {i}: {e}")

        if ((i + 1) % (max(1, batch_size) * 5)) == 0:
            print(f"   Progress: {min(i + batch_size, len(payloads))}/{len(payloads)}")
        time.sleep(sleep_s)

    out_df = pd.DataFrame(rows)

    # Append to log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        prev = pd.read_csv(log_path)
        out_df = pd.concat([prev, out_df], ignore_index=True)

    out_df.to_csv(log_path, index=False)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=DEFAULT_API, help="Base API URL (no trailing slash)")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Output CSV for predictions log")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the survey CSV")
    ap.add_argument("--rows", type=int, default=200, help="Number of rows to simulate")
    ap.add_argument("--batch", type=int, default=32, help="Batch size (>1 uses /predict_batch)")
    ap.add_argument("--sleep", type=float, default=0.02, help="Delay between requests/batches (seconds)")
    args = ap.parse_args()

    print("\n🧪 Starting Mental Health simulation...\n")
    out_df = simulate(args.api.rstrip("/"), args.log, args.data, args.rows, args.batch, args.sleep)
    print(f"✅ Logged {len(out_df)} total rows to {args.log.resolve()}")


if __name__ == "__main__":
    main()
