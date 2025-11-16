# -*- coding: utf-8 -*-
"""
Mental Health Treatment Prediction - Production Training Script.

Loads best configuration from experiment.ipynb and trains final model.
Logs to MLflow and returns run_id for deployment.

Usage:
    1. Run experiment.ipynb to generate best_config.json
    2. Run this script: python train.py
    3. Deploy with run_id from run_id.txt
"""

import os
import re
import json
import zipfile
import shutil
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---- compat OneHotEncoder across sklearn versions ----
try:
    from sklearn.preprocessing import OneHotEncoder  # type: ignore
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
except TypeError:
    from sklearn.preprocessing import OneHotEncoder  # type: ignore
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # sklearn <1.2
# ------------------------------------------------------

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score,
    f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

# ==================== CONFIG ====================
KAGGLE_DATASET = "osmi/mental-health-in-tech-survey"
BEST_CONFIG_FILE = "best_config.json"
MLFLOW_EXPERIMENT_NAME = "mental-health-production"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ==================== HELPERS ====================
def clean_gender(gen):
    """Clean gender values."""
    s = str(gen).strip().lower()
    s = re.sub(r"[\W_]+", " ", s).strip()
    if s in {"m", "male", "man", "make", "mal", "malr",
             "msle", "masc", "mail", "boy"}:
        return "Male"
    if s in {"f", "female", "woman", "femake", "femail",
             "femme", "girl"}:
        return "Female"
    return "Other"

def load_best_config():
    """Load best configuration from experiment."""
    if not os.path.exists(BEST_CONFIG_FILE):
        raise FileNotFoundError(
            f"\n❌ {BEST_CONFIG_FILE} not found!\n"
            "\n📝 Steps to fix:\n"
            "   1. Open and run: 0 try/experiment.ipynb\n"
            "   2. Execute all cells to generate best_config.json\n"
            "   3. The file will be created in: 3-cicd/best_config.json\n"
            "   4. Then run this script again: python train.py\n"
        )

    with open(BEST_CONFIG_FILE, 'r') as f:
        config = json.load(f)

    print(f"\n{'=' * 60}")
    print("LOADED CONFIGURATION FROM EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Source: {BEST_CONFIG_FILE}")
    print(f"Model: {config['model']['name']}")
    print(f"Features: {len(config['features']['selected'])}")
    print(
        f"Expected Recall: "
        f"{config['performance']['cv_recall_mean']:.1%}"
    )
    print(
        f"Expected Precision: "
        f"{config['performance']['cv_precision_mean']:.1%}"
    )
    print(f"{'=' * 60}")

    return config

def create_model(model_name, hyperparameters):
    """Create model instance from config."""
    models = {
        'LogisticRegression': LogisticRegression,
        'RandomForest': RandomForestClassifier,
        'XGBoost': XGBClassifier
    }

    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    return model_class(**hyperparameters, random_state=RANDOM_STATE)

def build_preprocessor(X):
    """Build preprocessing pipeline."""
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    # Use module-level _ohe defined for sklearn compatibility

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", _ohe),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

def load_from_kaggle():
    """Download dataset from Kaggle."""
    from shutil import which

    if not which("kaggle"):
        raise RuntimeError("Install Kaggle CLI: pip install kaggle")

    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists() and not (
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    ):
        raise RuntimeError(
            "Kaggle credentials not found. "
            "Create ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME/KAGGLE_KEY."
        )

    tmp = tempfile.TemporaryDirectory()
    tmpdir = Path(tmp.name)
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
            "-p", str(tmpdir), "-q"
        ])
        for z in tmpdir.glob("*.zip"):
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(tmpdir)
            z.unlink()

        csvs = list(tmpdir.rglob("*.csv"))
        surveyish = [p for p in csvs if "survey" in p.name.lower()]
        if surveyish:
            path = max(surveyish, key=lambda p: p.stat().st_size)
        else:
            path = max(csvs, key=lambda p: p.stat().st_size)

        return pd.read_csv(path)
    finally:
        try:
            tmp.cleanup()
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)

from pathlib import Path
import shutil
import mlflow
import urllib.parse
from mlflow.tracking import MlflowClient
try:
    from mlflow.artifacts import download_artifacts as mlflow_download_artifacts
except Exception:  # pragma: no cover - compatibility across MLflow versions
    mlflow_download_artifacts = None

def copy_model_to_local_dir(run_id):
    """Copy the 'model' artifact from a given MLflow run into ./models/model, version-agnostic.
    Tries, in order:
      1) mlflow.artifacts.download_artifacts with a runs:/ URI (if available)
      2) MlflowClient().download_artifacts(run_id, "model", ...)
      3) Direct filesystem copy via the run's artifact_uri
    """
    print("📦 Copying model to local directory...")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_dest = models_dir / "model"
    if model_dest.exists():
        shutil.rmtree(model_dest)

    # --- Option 1: modern helper using runs:/ URI ---
    if mlflow_download_artifacts is not None:
        try:
            # MLflow >= 2.x signature: (artifact_uri, dst_path=None)
            local_path = mlflow_download_artifacts(artifact_uri=f"runs:/{run_id}/model", dst_path=str(models_dir))
            print(f"Destination: {local_path}")
            return Path(local_path)
        except TypeError:
            # Some versions only accept (artifact_uri) and return a temp dir
            local_path = mlflow_download_artifacts(artifact_uri=f"runs:/{run_id}/model")
            src = Path(local_path)
            if src.exists() and src.is_dir():
                if src.name != "model":
                    shutil.move(str(src), str(model_dest))
                    print(f"Destination: {model_dest}")
                    return model_dest
                if src.parent != models_dir:
                    shutil.move(str(src), str(model_dest))
                    return model_dest
                print(f"Destination: {src}")
                return src
        except Exception as e:
            print(f"download_artifacts not usable, trying client API. Reason: {e}")

    # --- Option 2: client.download_artifacts ---
    try:
        client = MlflowClient()
        local_dir = client.download_artifacts(run_id, "model", dst_path=str(models_dir))
        src = Path(local_dir)
        if src.exists() and src.is_dir():
            if src.name != "model":
                shutil.move(str(src), str(model_dest))
                print(f"Destination: {model_dest}")
                return model_dest
            print(f"Destination: {src}")
            return src
    except Exception as e:
        print(f"MlflowClient().download_artifacts failed, trying direct path. Reason: {e}")

    # --- Option 3: direct filesystem path from artifact_uri ---
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri  # e.g., file:///.../mlruns/<exp>/<run>/artifacts
    parsed = urllib.parse.urlparse(artifact_uri)
    base_path = Path(parsed.path)
    model_source = base_path / "model"
    print(f"Source: {model_source}")
    if not model_source.exists():
        raise FileNotFoundError(f"Model not found at {model_source}")

    shutil.copytree(model_source, model_dest)
    print(f"Destination: {model_dest}")
    print("✅ Model copied successfully")
    return model_dest

def train_production_model(config):
    """Train final production model with MLflow tracking."""

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load data
    print("\nLoading data from Kaggle...")
    df = load_from_kaggle()
    print(f"✅ Data loaded: {df.shape}")

    # Preprocess
    target_col = "treatment"
    features_to_drop = [
        'Timestamp', 'Country', 'state', 'comments', target_col
    ]

    y = df[target_col].map({"Yes": 1, "No": 0}).astype(int)
    X = df.drop(columns=[c for c in features_to_drop if c in df.columns])
    X['Gender'] = X['Gender'].apply(clean_gender)

    # Use selected features from experiment
    selected_features = config['features']['selected']
    available = [c for c in selected_features if c in X.columns]
    X = X[available].copy()

    print(f"✅ Using {len(available)} selected features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")

        # Log config
        mlflow.log_params({
            'model_name': config['model']['name'],
            'n_features': len(available),
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
        })
        mlflow.log_params({
            f"hp_{k}": v
            for k, v in config['model']['hyperparameters'].items()
        })

        # Create and train model
        model = create_model(
            config['model']['name'],
            config['model']['hyperparameters']
        )

        preprocessor = build_preprocessor(X_train)
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

        print("\nTraining model...")
        pipe.fit(X_train, y_train)
        print("✅ Training complete")

        # Evaluate
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        metrics = {
            'test_recall': float(recall_score(y_test, y_pred)),
            'test_precision': float(precision_score(y_test, y_pred)),
            'test_f1': float(f1_score(y_test, y_pred)),
            'test_roc_auc': float(roc_auc_score(y_test, y_proba)),
            'test_tn': int(cm[0, 0]),
            'test_fp': int(cm[0, 1]),
            'test_fn': int(cm[1, 0]),
            'test_tp': int(cm[1, 1]),
        }

        mlflow.log_metrics(metrics)

        print(f"\n{'=' * 60}")
        print("TEST RESULTS")
        print(f"{'=' * 60}")
        print(f"Recall:    {metrics['test_recall']:.3f}")
        print(f"Precision: {metrics['test_precision']:.3f}")
        print(f"F1 Score:  {metrics['test_f1']:.3f}")
        print(f"ROC-AUC:   {metrics['test_roc_auc']:.3f}")
        print("\nConfusion Matrix:")
        print(f"  TN: {metrics['test_tn']:3d}  FP: {metrics['test_fp']:3d}")
        print(f"  FN: {metrics['test_fn']:3d}  TP: {metrics['test_tp']:3d}")

        # Save schema
        schema = {
            'numeric_features': [],
            'categorical_features': available,
            'retained_features': available,
            'target': 'treatment',
        }

        with open('training_schema.json', 'w') as f:
            json.dump(schema, f, indent=2)

        mlflow.log_artifact('training_schema.json')
        mlflow.log_artifact(BEST_CONFIG_FILE)

        # Log model
        mlflow.sklearn.log_model(
            pipe,
            artifact_path='model',
            registered_model_name='mental_health_classifier'
        )

        print("\n✅ Model logged to MLflow")

        # Copy model to local directory for workflow artifact upload
        copy_model_to_local_dir(run_id)

        return run_id, metrics

def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 60)

    # Load experiment results
    config = load_best_config()

    # Train production model
    run_id, metrics = train_production_model(config)

    # Save run_id for deployment
    with open('run_id.txt', 'w') as f:
        f.write(run_id)

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n✅ Run ID: {run_id}")
    print("✅ Run ID saved to: run_id.txt")
    print("✅ Schema saved to: training_schema.json")
    print("✅ Model copied to: models/model/")
    print("\n📊 View in MLflow UI:")
    print(f"   mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("\n🚀 Ready for deployment with app.py")

    return run_id

if __name__ == "__main__":
    main()