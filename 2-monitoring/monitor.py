"""
Generate an Evidently HTML report for the Mental Health classifier,
comparing older (reference) vs newer (current) logged predictions.

Usage:
    python health_monitor.py \
        --log data/health_predictions.csv \
        --out reports/health_monitoring_report.html \
        [--schema training_schema.json]

Notes:
- If the true target column ("treatment") is present in the log, the report
  will include performance + data drift (ClassificationPreset).
- If the target is missing, the report will include data drift only.
- Features are taken from the provided schema when available; otherwise they
  are inferred from the log file's dtypes, excluding meta columns.

This script is based on the original monitor.py (taxi regression drift).
"""

import argparse
from pathlib import Path
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_LOG = Path("data/health_predictions.csv")
DEFAULT_OUT = Path("health_monitoring_report.html")
DEFAULT_SCHEMA = Path("training_schema.json")

META_COLS = {"ts", "prediction", "probability", "proba", "run_id", "model_version"}
TARGET_COL = "treatment"         # 1 = Yes, 0 = No
PRED_COL = "prediction"          # predicted label from the API/script


def load_schema(schema_path: Path):
    """Load optional training schema to recover feature lists."""
    import json
    if not schema_path.exists():
        return {}
    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def infer_features(df: pd.DataFrame, schema: dict):
    """Return (num_features, cat_features) based on schema or inference."""
    num = []
    cat = []
    if schema:
        retained = schema.get("retained_features", "ALL_COLUMNS")
        if isinstance(retained, list) and retained:
            cols = [c for c in retained if c in df.columns]
        else:
            cols = list(dict.fromkeys((schema.get("numeric_features") or []) + (schema.get("categorical_features") or [])))
            cols = [c for c in cols if c in df.columns]
    else:
        # Infer from dtypes; drop meta and target/pred columns
        cols = [c for c in df.columns if c not in META_COLS | {TARGET_COL, PRED_COL}]
    # Split by dtype
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help="CSV with logged predictions")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output HTML report path")
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help="Optional training_schema.json")
    args = ap.parse_args()

    print("\n🩺 Starting Mental Health monitoring report...\n")

    if not args.log.exists():
        raise FileNotFoundError(f"❌ No logged predictions found at: {args.log}")

    # Load logged predictions
    df = pd.read_csv(args.log)
    # Parse timestamp if present
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # Ensure required columns for analysis exist
    if PRED_COL not in df.columns:
        raise ValueError(f"Expected column '{PRED_COL}' with predicted labels in {args.log}")

    # Drop rows without predictions; optionally require target if present
    drop_cols = [PRED_COL]
    if TARGET_COL in df.columns:
        drop_cols.append(TARGET_COL)
    before = len(df)
    df = df.dropna(subset=drop_cols)
    print(f"✓ Loaded {before} rows; {len(df)} remain after dropping NAs in {drop_cols}")

    # Sort by time if available; otherwise keep original order
    if "ts" in df.columns:
        df = df.sort_values("ts")

    # Split into reference (older) vs current (recent) halves
    midpoint = len(df) // 2 or 1
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"Reference: {len(reference)}  |  Current: {len(current)}")

    # Build column mapping using schema if present
    schema = load_schema(args.schema)
    num_feats, cat_feats = infer_features(df, schema)

    column_mapping = ColumnMapping(
        target=TARGET_COL if TARGET_COL in df.columns else None,
        prediction=PRED_COL,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )

    # Choose metrics based on availability of target
    metrics = [DataDriftPreset()]
    if TARGET_COL in df.columns:
        metrics.append(ClassificationPreset())

    print("\n🧮 Generating Evidently report...")
    report = Report(metrics=metrics)
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    # Save HTML
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(args.out))
    print(f"✅ Report saved: {args.out.resolve()}")
    print("Open it in your browser to explore drift & performance.\n")


if __name__ == "__main__":
    main()
