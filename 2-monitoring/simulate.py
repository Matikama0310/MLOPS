"""
Simulate requests using SELECTED features from best_config.json
"""

import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import requests
import pandas as pd


DEFAULT_API = "http://localhost:9696"
DEFAULT_LOG = Path("data/health_predictions.csv")
DEFAULT_DATA = Path("../data/raw/survey.csv")
CONFIG_FILE = Path("best_config.json")

# Will be loaded from config
SELECTED_FEATURES = []
DROP_COLS = ["Timestamp", "Country", "state", "comments", "treatment"]


def load_config():
    """Load best_config.json to get selected features"""
    global SELECTED_FEATURES
    if not CONFIG_FILE.exists():
        print(f"⚠️  {CONFIG_FILE} not found, using default features")
        SELECTED_FEATURES = [
            "work_interfere", "family_history", "care_options",
            "supervisor", "leave", "Gender", "obs_consequence",
            "mental_health_interview", "no_employees", "coworkers",
            "phys_health_interview", "self_employed",
            "mental_health_consequence"
        ]
        return
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            SELECTED_FEATURES = config['features']['selected']
            print(f"✓ Loaded {len(SELECTED_FEATURES)} selected features from config")
    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        SELECTED_FEATURES = []


def load_data(path: Path, n_rows: int = 200, shuffle_seed: int = 42) -> pd.DataFrame:
    """Load survey data"""
    if not path.exists():
        raise FileNotFoundError(f"Input data not found: {path}")
    
    df = pd.read_csv(path)

    # Keep ground truth
    if "treatment" in df.columns:
        df["treatment"] = df["treatment"].map({"Yes": 1, "No": 0}).astype("Int64")

    if n_rows and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=shuffle_seed).reset_index(drop=True)
    
    return df


def to_payload_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert to payload using ONLY selected features"""
    # Use selected features that exist in the dataframe
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    
    if not available_features:
        print("⚠️  No selected features found in data, using all non-meta columns")
        available_features = [c for c in df.columns if c not in DROP_COLS]
    
    print(f"Using {len(available_features)} features for payloads")
    
    # Convert to dict, handling NaN
    payloads = df[available_features].where(
        pd.notna(df[available_features]), None
    ).to_dict(orient="records")
    
    return payloads


def _post_json(url: str, endpoint: str, json_payload: Any, 
               timeout_s: float = 10.0) -> requests.Response:
    return requests.post(
        f"{url}{endpoint}", json=json_payload, timeout=timeout_s
    )


def simulate(api: str, log_path: Path, data_path: Path, 
             n_rows: int, batch_size: int, sleep_s: float) -> pd.DataFrame:
    """Send requests and log results"""
    df = load_data(data_path, n_rows=n_rows)
    payloads = to_payload_rows(df)

    rows = []
    use_batch = batch_size and batch_size > 1
    endpoint = "/predict_batch" if use_batch else "/predict"

    print(f"\n🚀 Starting simulation...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Rows: {len(payloads)}")
    print(f"   Batch size: {batch_size if use_batch else 1}")

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
                
                for j, pld in enumerate(batch):
                    row = dict(pld)
                    row.update({
                        "ts": ts,
                        "prediction": preds[j],
                        "probability": float(prob[j]) if prob[j] is not None else None,
                        "model_version": run_id,
                    })
                    # Add ground truth
                    if "treatment" in df.columns:
                        idx = i + j
                        if idx < len(df):
                            row["treatment"] = int(df.iloc[idx]["treatment"]) \
                                if pd.notna(df.iloc[idx]["treatment"]) else None
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
                if "treatment" in df.columns and i < len(df):
                    row["treatment"] = int(df.iloc[i]["treatment"]) \
                        if pd.notna(df.iloc[i]["treatment"]) else None
                rows.append(row)

        except Exception as e:
            print(f"⚠️  Request failed at index {i}: {e}")

        if ((i + 1) % (max(1, batch_size) * 5)) == 0:
            progress = min(i + batch_size, len(payloads))
            print(f"   Progress: {progress}/{len(payloads)}")
        
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
    ap.add_argument("--api", default=DEFAULT_API, 
                    help="Base API URL")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, 
                    help="Output CSV")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, 
                    help="Survey CSV")
    ap.add_argument("--rows", type=int, default=200, 
                    help="Rows to simulate")
    ap.add_argument("--batch", type=int, default=32, 
                    help="Batch size")
    ap.add_argument("--sleep", type=float, default=0.02, 
                    help="Delay between requests")
    args = ap.parse_args()

    print("\n🧪 Mental Health Simulation\n")
    
    # Load config first
    load_config()
    
    out_df = simulate(
        args.api.rstrip("/"), args.log, args.data, 
        args.rows, args.batch, args.sleep
    )
    
    print(f"\n✅ Logged {len(out_df)} total rows to {args.log.resolve()}")
    print(f"   Columns: {list(out_df.columns)}")
    print(f"\n📊 Summary:")
    print(f"   Predictions: {out_df['prediction'].value_counts().to_dict()}")
    if 'treatment' in out_df.columns:
        print(f"   Ground truth: {out_df['treatment'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()