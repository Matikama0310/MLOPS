"""
Generate Evidently report for Mental Health classifier
Uses features from training_schema.json automatically
"""

import argparse
from pathlib import Path
import pandas as pd

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# Defaults
DEFAULT_LOG = Path("data/health_predictions.csv")
DEFAULT_OUT = Path("health_monitoring_report.html")
DEFAULT_SCHEMA = Path("training_schema.json")

META_COLS = {"ts", "prediction", "probability", "proba", "run_id", "model_version"}
TARGET_COL = "treatment"
PRED_COL = "prediction"


def load_schema(schema_path: Path):
    """Load training schema"""
    import json
    if not schema_path.exists():
        print(f"⚠️  Schema not found: {schema_path}")
        return {}
    try:
        with open(schema_path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
            print(f"✓ Loaded schema from {schema_path}")
            return schema
    except Exception as e:
        print(f"⚠️  Error loading schema: {e}")
        return {}


def infer_features(df: pd.DataFrame, schema: dict):
    """Get features from schema or infer from data"""
    num = []
    cat = []
    
    if schema and 'retained_features' in schema:
        cols = [c for c in schema['retained_features'] if c in df.columns]
        print(f"✓ Using {len(cols)} features from schema")
    else:
        # Infer: drop meta columns
        cols = [c for c in df.columns 
                if c not in META_COLS | {TARGET_COL, PRED_COL}]
        print(f"⚠️  Inferring {len(cols)} features from data")
    
    # Split by dtype
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    
    print(f"   Numeric: {len(num)}, Categorical: {len(cat)}")
    return num, cat


def main():
    ap = argparse.ArgumentParser(
        description="Generate Evidently monitoring report"
    )
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, 
                    help="CSV with logged predictions")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, 
                    help="Output HTML path")
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, 
                    help="training_schema.json")
    args = ap.parse_args()

    print("\n🩺 Mental Health Monitoring Report\n")

    if not args.log.exists():
        raise FileNotFoundError(f"❌ Log not found: {args.log}")

    # Load data
    df = pd.read_csv(args.log)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    # Validate
    if PRED_COL not in df.columns:
        raise ValueError(f"❌ Missing '{PRED_COL}' column")

    # Drop NaN predictions
    drop_cols = [PRED_COL]
    if TARGET_COL in df.columns:
        drop_cols.append(TARGET_COL)
    
    before = len(df)
    df = df.dropna(subset=drop_cols)
    print(f"✓ Loaded {before} rows, {len(df)} valid after dropping NaN")

    # Sort by time
    if "ts" in df.columns:
        df = df.sort_values("ts")

    # Split reference vs current
    midpoint = len(df) // 2 or 1
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"   Reference: {len(reference)} rows")
    print(f"   Current:   {len(current)} rows")

    # Get features from schema
    schema = load_schema(args.schema)
    num_feats, cat_feats = infer_features(df, schema)

    # Column mapping
    column_mapping = ColumnMapping(
        target=TARGET_COL if TARGET_COL in df.columns else None,
        prediction=PRED_COL,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )

    # Choose metrics
    metrics = [DataDriftPreset()]
    if TARGET_COL in df.columns:
        metrics.append(ClassificationPreset())
        print(f"✓ Including performance metrics (target available)")
    else:
        print(f"⚠️  Target not found, drift only")

    # Generate report
    print("\n🧮 Generating report...")
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference, 
        current_data=current, 
        column_mapping=column_mapping
    )

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(args.out))
    
    print(f"✅ Report saved: {args.out.resolve()}")
    print(f"\n📊 Open it in your browser to view drift analysis\n")


if __name__ == "__main__":
    main()