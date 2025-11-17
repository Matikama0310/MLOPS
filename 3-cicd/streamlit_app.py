# -*- coding: utf-8 -*-
"""
Streamlit Frontend for Mental Health Treatment Predictor

A beautiful, interactive UI for making predictions using the ML model.
"""

import json
from typing import Dict, Any

import streamlit as st
import requests

# Configuration
API_URL = "https://mental-health-w98g.onrender.com"

# Feature options (matching your model's expected values)
FEATURE_OPTIONS = {
    "Gender": ["Male", "Female", "Other"],
    "self_employed": ["Yes", "No"],
    "family_history": ["Yes", "No"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often"],
    "no_employees": ["1-5", "6-25", "26-100", "100-500", "500-1000",
                     "More than 1000"],
    "remote_work": ["Yes", "No"],
    "tech_company": ["Yes", "No"],
    "benefits": ["Yes", "No", "Don't know"],
    "care_options": ["Yes", "No", "Not sure"],
    "wellness_program": ["Yes", "No", "Don't know"],
    "seek_help": ["Yes", "No", "Don't know"],
    "anonymity": ["Yes", "No", "Don't know"],
    "leave": ["Very easy", "Somewhat easy", "Somewhat difficult",
              "Very difficult", "Don't know"],
    "mental_health_consequence": ["Yes", "No", "Maybe"],
    "phys_health_consequence": ["Yes", "No", "Maybe"],
    "coworkers": ["Yes", "No", "Some of them"],
    "supervisor": ["Yes", "No", "Some of them"],
    "mental_health_interview": ["Yes", "No", "Maybe"],
    "phys_health_interview": ["Yes", "No", "Maybe"],
    "mental_vs_physical": ["Yes", "No", "Don't know"],
    "obs_consequence": ["Yes", "No"]
}

# Feature descriptions for better UX
FEATURE_DESCRIPTIONS = {
    "Gender": "Your gender identity",
    "self_employed": "Are you self-employed?",
    "family_history": "Do you have a family history of mental illness?",
    "work_interfere": "How often does mental health interfere with work?",
    "no_employees": "How many employees does your company have?",
    "remote_work": "Do you work remotely?",
    "tech_company": "Is your employer primarily a tech company?",
    "benefits": "Does your employer provide mental health benefits?",
    "care_options": "Do you know the options for mental health care?",
    "wellness_program": "Has your employer discussed mental health in a wellness program?",
    "seek_help": "Does your employer provide resources to learn about mental health?",
    "anonymity": "Is your anonymity protected if you use mental health resources?",
    "leave": "How easy is it to take medical leave for a mental health condition?",
    "mental_health_consequence": "Do you think discussing mental health has negative consequences?",
    "phys_health_consequence": "Do you think discussing physical health has negative consequences?",
    "coworkers": "Would you be willing to discuss mental health with coworkers?",
    "supervisor": "Would you be willing to discuss mental health with your supervisor?",
    "mental_health_interview": "Would you bring up mental health in an interview?",
    "phys_health_interview": "Would you bring up physical health in an interview?",
    "mental_vs_physical": "Do you feel your employer takes mental health as seriously as physical health?",
    "obs_consequence": "Have you heard of or observed negative consequences for mental health discussions?"
}

# Page configuration
st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-yes {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .prediction-no {
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
    }
    h1 {
        color: #1f77b4;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def make_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make a prediction using the API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


def main():
    # Header
    st.title("🧠 Mental Health Treatment Predictor")
    st.markdown("""
    This tool helps predict whether someone might benefit from mental health treatment
    based on workplace and personal factors. Please answer the questions below.
    """)

    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not available. Please make sure the backend service is running.")
        st.info(f"Expected API URL: {API_URL}")
        st.stop()
    else:
        st.success("✅ Connected to prediction service")

    st.markdown("---")

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This predictor uses machine learning to assess whether
        someone might benefit from mental health treatment based on:

        - **Workplace factors**: Company support, benefits, culture
        - **Personal factors**: Family history, current challenges
        - **Attitudes**: Comfort discussing mental health

        **Important**: This is a prediction tool, not a diagnosis.
        Always consult healthcare professionals for mental health concerns.
        """)

        st.markdown("---")
        st.header("📊 Model Info")

        # Get model info from API
        try:
            root_info = requests.get(f"{API_URL}/").json()
            st.write(f"**Model Version**: {root_info.get('run_id', 'N/A')}")
            st.write(f"**Schema Loaded**: {'✅' if root_info.get('has_schema') else '❌'}")
        except Exception:
            st.write("Unable to fetch model info")

    # Main form
    st.header("📝 Survey Questions")

    # Create form
    with st.form("prediction_form"):
        # Organize fields in columns for better layout
        col1, col2 = st.columns(2)

        user_inputs = {}

        # Distribute features across columns
        features = list(FEATURE_OPTIONS.keys())
        mid_point = len(features) // 2

        with col1:
            st.subheader("Personal & Work Information")
            for feature in features[:mid_point]:
                user_inputs[feature] = st.selectbox(
                    label=f"{feature}",
                    options=FEATURE_OPTIONS[feature],
                    help=FEATURE_DESCRIPTIONS.get(feature, ""),
                    key=feature
                )

        with col2:
            st.subheader("Workplace Culture & Support")
            for feature in features[mid_point:]:
                user_inputs[feature] = st.selectbox(
                    label=f"{feature}",
                    options=FEATURE_OPTIONS[feature],
                    help=FEATURE_DESCRIPTIONS.get(feature, ""),
                    key=feature
                )

        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Get Prediction")

    # Process prediction
    if submitted:
        with st.spinner("🔄 Analyzing your responses..."):
            result = make_prediction(user_inputs)

            if result:
                prediction = result["predictions"][0]
                probability = result["probabilities"][0]

                st.markdown("---")
                st.header("📊 Prediction Results")

                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-box prediction-yes">
                        <h2>🟡 Treatment Recommended</h2>
                        <p style="font-size: 20px;">
                            Based on your responses, seeking mental health treatment may be beneficial.
                        </p>
                        <p style="font-size: 16px; color: #666;">
                            Confidence: {probability:.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info("""
                    **What this means:**
                    - Consider talking to a mental health professional
                    - Your workplace factors suggest potential challenges
                    - Treatment can provide valuable support and coping strategies

                    **Next steps:**
                    - Contact your employee assistance program (EAP)
                    - Reach out to your insurance provider for covered therapists
                    - Consider online therapy options if in-person isn't accessible
                    """)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-no">
                        <h2>🟢 Treatment Not Indicated</h2>
                        <p style="font-size: 20px;">
                            Based on your responses, you may not currently need treatment.
                        </p>
                        <p style="font-size: 16px; color: #666;">
                            Confidence: {(1 - probability):.1%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.info("""
                    **What this means:**
                    - Your current situation appears manageable
                    - Continue monitoring your mental health
                    - Maintain healthy coping strategies

                    **Remember:**
                    - It's always okay to seek help if you feel you need it
                    - Prevention is valuable - consider talking to someone even if things seem okay
                    - Your mental health needs may change over time
                    """)

                # Show detailed probability
                st.markdown("---")
                st.subheader("📈 Prediction Details")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Probability of Needing Treatment",
                        value=f"{probability:.1%}"
                    )

                with col2:
                    st.metric(
                        label="Probability of Not Needing Treatment",
                        value=f"{(1 - probability):.1%}"
                    )

                # Progress bar for visual representation
                st.progress(probability)

                # Disclaimer
                st.warning("""
                **Important Disclaimer**: This prediction is based on a machine learning model
                trained on survey data. It should not replace professional medical advice, diagnosis,
                or treatment. If you're experiencing mental health challenges, please consult with
                a qualified healthcare provider.
                """)

                # Option to download results
                with st.expander("📥 Download Results"):
                    results_json = json.dumps({
                        "inputs": user_inputs,
                        "prediction": "Treatment Recommended" if prediction == 1 else "Treatment Not Indicated",
                        "probability": probability,
                        "model_version": result.get("model_version", "unknown")
                    }, indent=2)

                    st.download_button(
                        label="Download JSON",
                        data=results_json,
                        file_name="mental_health_prediction.json",
                        mime="application/json"
                    )


if __name__ == "__main__":
    main()
