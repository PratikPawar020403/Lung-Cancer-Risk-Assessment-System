"""
predict.py

A Streamlit web application that loads a pre-trained lung cancer prediction pipeline,
accepts user input via two assessment forms (Basic and Advanced), preprocesses the input,
and generates predictions with a visually appealing gauge and a color-coded result box.
The Advanced tab also displays personalized recommendations.
"""

import os
from typing import Tuple

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load  # For model persistence
import plotly.graph_objects as go

# ----------------------------- #
#      Page Configuration       #
# ----------------------------- #
st.set_page_config(
    page_title="Risk Prediction - Lung Cancer Risk Assessment",
    page_icon="🔍",
    layout="wide"
)

# Inject custom CSS for improved design and a smaller result box
st.markdown(
    """
    <style>
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    h1, h2, h3 {
        color: #343a40;
    }
    .result-box {
        padding: 0.5rem; 
        border-radius: 4px; 
        text-align: center;
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------- #
#      Utility Functions        #
# ----------------------------- #
@st.cache_resource(show_spinner=True)
def load_pipeline() -> object:
    """
    Load the pre-trained prediction pipeline from a joblib file.
    
    Returns:
        The loaded pipeline.
    """
    pipeline_paths = [
        os.path.join("lung-cancer/best_lung_cancer_model.joblib"),
        "best_lung_cancer_model.joblib"
    ]
    for path in pipeline_paths:
        if os.path.exists(path):
            return load(path)
    raise FileNotFoundError("Pipeline file not found.")

def preprocess_input(user_input: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the user input by adding engineered features and ensuring the input 
    contains all features (in the same order) used during training.
    
    Args:
        user_input (pd.DataFrame): Raw input features.
    
    Returns:
        pd.DataFrame: Processed input DataFrame.
    """
    # Ensure all provided columns are numeric
    user_input = user_input.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # Add missing features with default values if necessary
    defaults = {
        "ENERGY_LEVEL": 59.0,
        "OXYGEN_SATURATION": 96.0
    }
    for col, default in defaults.items():
        if col not in user_input.columns:
            user_input[col] = default

    # Create engineered features (must match training)
    user_input["SMOKING_AGE_Interaction"] = user_input["SMOKING"] * user_input["AGE"]
    user_input["SMOKING_FAMILY_HISTORY_Interaction"] = user_input["SMOKING"] * user_input["FAMILY_HISTORY"]
    user_input["AGE_Squared"] = user_input["AGE"] ** 2
    user_input["Respiratory_Issue_Score"] = (
        user_input["BREATHING_ISSUE"] +
        user_input["CHEST_TIGHTNESS"] +
        user_input["THROAT_DISCOMFORT"]
    )
    
    # Ensure the column order matches what was used during training
    expected_features = [
        "AGE", "GENDER", "SMOKING", "FINGER_DISCOLORATION", "MENTAL_STRESS",
        "EXPOSURE_TO_POLLUTION", "LONG_TERM_ILLNESS", "ENERGY_LEVEL", "IMMUNE_WEAKNESS",
        "BREATHING_ISSUE", "ALCOHOL_CONSUMPTION", "THROAT_DISCOMFORT", "OXYGEN_SATURATION",
        "CHEST_TIGHTNESS", "FAMILY_HISTORY", "SMOKING_FAMILY_HISTORY", "STRESS_IMMUNE",
        "SMOKING_AGE_Interaction", "SMOKING_FAMILY_HISTORY_Interaction",
        "AGE_Squared", "Respiratory_Issue_Score"
    ]
    for col in expected_features:
        if col not in user_input.columns:
            user_input[col] = 0
    user_input = user_input[expected_features]
    
    return user_input

def get_risk_category(probability: float) -> Tuple[str, str]:
    """
    Determine the risk category and corresponding color based on the predicted probability.
    
    Args:
        probability (float): Predicted probability of pulmonary disease.
    
    Returns:
        Tuple[str, str]: Risk category and associated color.
    """
    if probability < 0.2:
        return "Low Risk", "#28a745"      # green
    elif probability < 0.5:
        return "Moderate Risk", "#fd7e14"   # orange
    else:
        return "High Risk", "#dc3545"       # red

# ----------------------------- #
#         Load Pipeline         #
# ----------------------------- #
try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    model_loaded = False

# ----------------------------- #
#          App Title            #
# ----------------------------- #
st.title("🔍 Lung Cancer Risk Assessment Tool")
st.markdown(
    """
### Get Your Personalized Risk Assessment

Complete the appropriate form below to receive your risk score.
**Note:** This tool is for informational purposes only and does not replace professional medical advice.
    """
)

# ----------------------------- #
#        Input Forms            #
# ----------------------------- #
tabs = st.tabs(["Basic Assessment", "Advanced Assessment"])

# --- Basic Assessment Tab ---
with tabs[0]:
    st.subheader("Basic Risk Assessment")
    with st.form("basic_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=18, max_value=100, value=40)
            gender = st.radio("Gender", options=["Female", "Male"], horizontal=True)
            smoking = st.radio("Do you smoke?", options=["No", "Yes"], horizontal=True)
        with col2:
            breathing_issue = st.radio("Breathing issues?", options=["No", "Yes"], horizontal=True)
            family_history = st.radio("Family History of Lung Cancer", options=["No", "Yes"], horizontal=True)
            pollution_exposure = st.radio("Exposure to pollution?", options=["No", "Yes"], horizontal=True)
        basic_submitted = st.form_submit_button("Calculate Risk")
    
    if basic_submitted and model_loaded:
        # Construct input DataFrame for basic assessment (defaults for missing features added in preprocess)
        user_input = pd.DataFrame({
            "AGE": [age],
            "GENDER": [1 if gender == "Male" else 0],
            "SMOKING": [1 if smoking == "Yes" else 0],
            "FINGER_DISCOLORATION": [0],
            "MENTAL_STRESS": [0],
            "EXPOSURE_TO_POLLUTION": [1 if pollution_exposure == "Yes" else 0],
            "LONG_TERM_ILLNESS": [0],
            "IMMUNE_WEAKNESS": [0],
            "BREATHING_ISSUE": [1 if breathing_issue == "Yes" else 0],
            "ALCOHOL_CONSUMPTION": [0],
            "THROAT_DISCOMFORT": [0],
            "CHEST_TIGHTNESS": [0],
            "FAMILY_HISTORY": [1 if family_history == "Yes" else 0],
            "SMOKING_FAMILY_HISTORY": [0],
            "STRESS_IMMUNE": [0]
        })
        processed_input = preprocess_input(user_input)
        prediction_prob = pipeline.predict_proba(processed_input)[0][1]
        risk_category, color = get_risk_category(prediction_prob)
        
        # Display prediction output for Basic tab
        st.header("Your Risk Assessment Results")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Percentage", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': 'lightgreen'},
                        {'range': [20, 50], 'color': 'orange'},
                        {'range': [50, 100], 'color': 'coral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_prob * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(
                f"""<div class="result-box" style="background-color:{color};">
                <h3>Risk: {risk_category}</h3>
                <p>{prediction_prob*100:.1f}%</p>
                </div>""",
                unsafe_allow_html=True
            )

# --- Advanced Assessment Tab ---
with tabs[1]:
    st.subheader("Advanced Risk Assessment")
    with st.form("advanced_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            adv_age = st.number_input("Age", min_value=18, max_value=100, value=40)
            adv_gender = st.radio("Gender", options=["Female", "Male"], key="adv_gender", horizontal=True)
        with col2:
            adv_family_history = st.radio("Family History of Lung Cancer", options=["No", "Yes"], key="adv_family", horizontal=True)
            adv_smoking_family_history = st.radio("Family History of Smoking", options=["No", "Yes"], key="adv_smoking_family", horizontal=True)
        st.markdown("### Behavioral Factors")
        col1, col2 = st.columns(2)
        with col1:
            adv_smoking = st.radio("Do you smoke?", options=["No", "Yes"], key="adv_smoking", horizontal=True)
            adv_alcohol = st.radio("Do you consume alcohol regularly?", options=["No", "Yes"], key="adv_alcohol", horizontal=True)
        with col2:
            adv_pollution_exposure = st.radio("Exposure to pollution?", options=["No", "Yes"], key="adv_pollution", horizontal=True)
            adv_stress = st.radio("Do you experience mental stress?", options=["No", "Yes"], key="adv_stress", horizontal=True)
        st.markdown("### Health Indicators")
        col1, col2 = st.columns(2)
        with col1:
            adv_breathing_issue = st.radio("Breathing issues?", options=["No", "Yes"], key="adv_breathing", horizontal=True)
            adv_chest_tightness = st.radio("Chest tightness?", options=["No", "Yes"], key="adv_chest", horizontal=True)
            adv_throat_discomfort = st.radio("Throat discomfort?", options=["No", "Yes"], key="adv_throat", horizontal=True)
        with col2:
            adv_finger_discoloration = st.radio("Finger discoloration?", options=["No", "Yes"], key="adv_finger", horizontal=True)
            adv_immune_weakness = st.radio("Weakened immune system?", options=["No", "Yes"], key="adv_immune", horizontal=True)
            adv_long_term_illness = st.radio("Long-term illness?", options=["No", "Yes"], key="adv_illness", horizontal=True)
        adv_stress_immune = 1 if adv_stress == "Yes" and adv_immune_weakness == "Yes" else 0
        advanced_submitted = st.form_submit_button("Calculate Detailed Risk")
    
    if advanced_submitted and model_loaded:
        # Construct input DataFrame for advanced assessment
        user_input = pd.DataFrame({
            "AGE": [adv_age],
            "GENDER": [1 if adv_gender == "Male" else 0],
            "SMOKING": [1 if adv_smoking == "Yes" else 0],
            "FINGER_DISCOLORATION": [1 if adv_finger_discoloration == "Yes" else 0],
            "MENTAL_STRESS": [1 if adv_stress == "Yes" else 0],
            "EXPOSURE_TO_POLLUTION": [1 if adv_pollution_exposure == "Yes" else 0],
            "LONG_TERM_ILLNESS": [1 if adv_long_term_illness == "Yes" else 0],
            "IMMUNE_WEAKNESS": [1 if adv_immune_weakness == "Yes" else 0],
            "BREATHING_ISSUE": [1 if adv_breathing_issue == "Yes" else 0],
            "ALCOHOL_CONSUMPTION": [1 if adv_alcohol == "Yes" else 0],
            "THROAT_DISCOMFORT": [1 if adv_throat_discomfort == "Yes" else 0],
            "CHEST_TIGHTNESS": [1 if adv_chest_tightness == "Yes" else 0],
            "FAMILY_HISTORY": [1 if adv_family_history == "Yes" else 0],
            "SMOKING_FAMILY_HISTORY": [1 if adv_smoking_family_history == "Yes" else 0],
            "STRESS_IMMUNE": [adv_stress_immune]
        })
        
        # Preprocess the user input to add engineered features and ensure proper column order
        processed_input = preprocess_input(user_input)
        prediction_prob = pipeline.predict_proba(processed_input)[0][1]
        risk_category, color = get_risk_category(prediction_prob)
        
        # Display prediction output for Advanced tab
        st.header("Your Risk Assessment Results")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Percentage", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 20], 'color': 'lightgreen'},
                        {'range': [20, 50], 'color': 'orange'},
                        {'range': [50, 100], 'color': 'coral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_prob * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown(
                f"""<div class="result-box" style="background-color:{color};">
                <h3>Risk: {risk_category}</h3>
                <p>{prediction_prob*100:.1f}%</p>
                </div>""",
                unsafe_allow_html=True
            )
        
        # Personalized Recommendations for Advanced Assessment
        st.header("Personalized Recommendations")
        rec_tabs = st.tabs(["Prevention", "Screening", "Lifestyle Changes"])
        with rec_tabs[0]:
            st.subheader("Prevention")
            recommendations = []
            if user_input["SMOKING"].iloc[0] == 1:
                recommendations.append(
                    "🚭 **Quit Smoking**: Consider smoking cessation programs and support groups."
                )
            if user_input["EXPOSURE_TO_POLLUTION"].iloc[0] == 1:
                recommendations.append(
                    "🏭 **Reduce Pollution Exposure**: Use air purifiers and wear protective masks."
                )
            if (user_input["BREATHING_ISSUE"].iloc[0] +
                user_input["CHEST_TIGHTNESS"].iloc[0] +
                user_input["THROAT_DISCOMFORT"].iloc[0]) > 0:
                recommendations.append(
                    "🫁 **Respiratory Health**: Consult a pulmonologist for evaluation."
                )
            if recommendations:
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.markdown("### Continue Healthy Practices: Maintain a balanced diet and regular exercise.")
        with rec_tabs[1]:
            st.subheader("Screening")
            high_risk = (
                user_input["AGE"].iloc[0] >= 50 or
                user_input["SMOKING"].iloc[0] == 1 or
                user_input["FAMILY_HISTORY"].iloc[0] == 1 or
                prediction_prob >= 0.3
            )
            if high_risk:
                st.markdown("### Recommended Screening: Consider annual low-dose CT scans and pulmonary tests.")
            else:
                st.markdown("### General Screening: Follow your physician's advice.")
        with rec_tabs[2]:
            st.subheader("Lifestyle Changes")
            st.markdown(
                """
                ### Lifestyle Modifications:
                - **Nutrition:** Emphasize fruits, vegetables, and omega-3s.
                - **Exercise:** Aim for at least 150 minutes of aerobic activity per week.
                - **Stress Management:** Practice mindfulness and ensure adequate sleep.
                """
            )
        
        st.header("Next Steps")
        if risk_category == "High Risk":
            st.markdown(
                """
                ### Urgent Actions for High Risk:
                1. **Medical Consultation:** Schedule an appointment promptly.
                2. **Lung Cancer Screening:** Request a low-dose CT scan.
                """
            )
        elif risk_category == "Moderate Risk":
            st.markdown(
                """
                ### Recommended Actions for Moderate Risk:
                1. **Discuss with Your Doctor:** Address potential risk factors.
                2. **Lifestyle Modifications:** Improve habits to lower risk.
                """
            )
        else:
            st.markdown(
                """
                ### Maintaining Low Risk:
                1. **Preventive Care:** Continue regular check-ups.
                2. **Healthy Lifestyle:** Maintain or enhance your current practices.
                """
            )

# ----------------------------- #
#    Information Section        #
# ----------------------------- #
if not ('user_input' in locals() and model_loaded):
    st.markdown(
        """
    ## How This Assessment Works
    Our model uses a pre-trained machine learning pipeline to assess your risk of pulmonary disease based on your responses.
    After submitting your information, you'll receive:
    1. A personalized risk score with a visual gauge.
    2. Tailored recommendations for prevention, screening, and lifestyle changes.
        """
    )
