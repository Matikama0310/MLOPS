"""Streamlit frontend for the Mental Health treatment predictor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.pyfunc
import pandas as pd
import streamlit as st


##############################
# Configuration
##############################
PROJECT_ROOT = Path(__file__).parent
DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "3-cicd" / "training_schema.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "survey.csv"
RUN_ID_LOCATIONS = [
    PROJECT_ROOT / "1-application" / "run_id.txt",
    PROJECT_ROOT / "1-application" / "artifacts" / "run_id.txt",
    PROJECT_ROOT / "run_id.txt",
]


##############################
# Helpers
##############################
def clean_gender(value: object) -> str:
    s = str(value).strip().lower()
    replacements = {
        "m": "Male",
        "male": "Male",
        "man": "Male",
        "make": "Male",
        "mal": "Male",
        "malr": "Male",
        "msle": "Male",
        "masc": "Male",
        "mail": "Male",
        "boy": "Male",
        "f": "Female",
        "female": "Female",
        "woman": "Female",
        "femake": "Female",
        "femail": "Female",
        "femme": "Female",
        "girl": "Female",
    }
    return replacements.get(s, "Other")


def resolve_run_id() -> str:
    env_run_id = os.getenv("RUN_ID", "").strip()
    if env_run_id:
        return env_run_id
    for path in RUN_ID_LOCATIONS:
        if path.exists():
            run_id = path.read_text(encoding="utf-8").strip()
            if run_id:
                return run_id
    return ""


def load_schema(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_data(show_spinner=False)
def load_feature_catalog(schema_path: Path, data_path: Path) -> Tuple[List[str], Dict[str, Dict]]:
    schema = load_schema(schema_path)
    features = schema.get("retained_features") or schema.get("categorical_features") or []
    catalog: Dict[str, Dict] = {}

    if data_path.exists():
        df = pd.read_csv(data_path)
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].apply(clean_gender)
        for feature in features:
            if feature not in df.columns:
                catalog[feature] = {"options": [], "default": None}
                continue
            series = df[feature].dropna().astype(str).str.strip()
            options = pd.Series(series.unique()).sort_values().tolist()
            default = series.mode().iloc[0] if not series.mode().empty else (options[0] if options else None)
            catalog[feature] = {"options": options, "default": default}
    else:
        for feature in features:
            catalog[feature] = {"options": [], "default": None}

    return features, catalog


@st.cache_resource(show_spinner=True)
def load_model(run_id: str, tracking_uri: str):
    if not run_id:
        raise RuntimeError("Missing run_id. Train a model or set RUN_ID env var.")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/model")


def build_user_payload(features: List[str], catalog: Dict[str, Dict]) -> Dict[str, str]:
    inputs: Dict[str, str] = {}
    for feature in features:
        meta = catalog.get(feature, {"options": [], "default": None})
        options = meta["options"] or ["Unknown"]
        default_value = meta["default"] if meta["default"] in options else options[0]
        selection = st.selectbox(
            label=feature.replace("_", " ").title(),
            options=options,
            index=options.index(default_value),
            key=f"select_{feature}",
        )
        inputs[feature] = selection
    return inputs


##############################
# UI
##############################
def main():
    st.set_page_config(
        page_title="Mental Health Treatment Predictor",
        page_icon="🧠",
        layout="centered",
    )

    st.title("🧠 Mental Health Treatment Predictor")
    st.caption("Interactive frontend powered by the production ML pipeline.")

    run_id = resolve_run_id()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    schema_path = Path(os.getenv("SCHEMA_PATH", str(DEFAULT_SCHEMA_PATH)))
    data_path = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_PATH)))

    try:
        model = load_model(run_id, tracking_uri)
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

    features, catalog = load_feature_catalog(schema_path, data_path)
    if not features:
        st.warning("No feature schema found. Please regenerate training_schema.json.")
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Survey Inputs")
        user_inputs = build_user_payload(features, catalog)
        submitted = st.form_submit_button("Predict Treatment Need", use_container_width=True)

    if not submitted:
        st.info("Select survey values above and click Predict.")
        return

    df = pd.DataFrame([user_inputs])
    try:
        proba = model.predict_proba(df)[:, 1][0]
        label = int(proba >= 0.5)
    except Exception:
        # Fallback to predict if predict_proba is unavailable
        label = int(model.predict(df)[0])
        proba = float(label)

    st.success("Prediction complete!")
    st.metric(
        label="Treatment Recommendation (1 = Needs treatment)",
        value=str(label),
        delta=f"{proba:.1%} probability of treatment",
        delta_color="inverse",
    )

    st.json({"run_id": run_id, "tracking_uri": tracking_uri, "inputs": user_inputs})


if __name__ == "__main__":
    main()

