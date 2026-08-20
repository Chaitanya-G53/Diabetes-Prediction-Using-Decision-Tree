import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark Theme CSS
st.markdown("""
<style>
    /* Global background and typography */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Card containers */
    .metric-card {
        background: #161f30;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    /* Result cards */
    .result-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        color: #fca5a5;
    }
    
    .result-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.3) 100%);
        border: 1.5px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        color: #6ee7b7;
    }

    /* Headings and labels */
    h1, h2, h3, h4, label {
        color: #f9fafb !important;
        font-weight: 600;
    }
    
    /* Highlight accent */
    .accent-text {
        color: #38bdf8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load the pickle model
@st.cache_resource
def load_model():
    model_path = "Diabetes-Prediction-Project-Model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please place it in the working directory.")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header Section
st.title("🧬 Diabetes Diagnostic Intelligence")
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top:-15px;'>Decision Tree Clinical Decision Support System</p>", unsafe_allow_html=True)
st.markdown("---")

# Feature Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📋 Patient Demographics & Lifestyle")
    
    age = st.slider("Age (Years)", min_value=1, max_value=100, value=45, step=1)
    
    gender = st.selectbox(
        "Biological Gender",
        options=["Female", "Male", "Other"],
        index=0
    )
    
    smoking_history = st.selectbox(
        "Smoking History",
        options=["never", "former", "current", "not current", "ever", "No Info"],
        index=0
    )
    
    st.markdown("### 🫀 Medical Preconditions")
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        hypertension = st.radio("Hypertension", options=["No", "Yes"], horizontal=True)
    with sub_col2:
        heart_disease = st.radio("Heart Disease", options=["No", "Yes"], horizontal=True)

with col2:
    st.markdown("### 🧪 Clinical Biomarkers")
    
    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=70.0,
        value=27.32,
        step=0.1,
        help="Standard healthy range: 18.5 - 24.9"
    )
    
    hba1c = st.slider(
        "HbA1c Level (%)",
        min_value=3.5,
        max_value=12.0,
        value=5.5,
        step=0.1,
        help="Normal: <5.7%, Prediabetes: 5.7%-6.4%, Diabetes: ≥6.5%"
    )
    
    glucose = st.number_input(
        "Blood Glucose Level (mg/dL)",
        min_value=50,
        max_value=350,
        value=130,
        step=1,
        help="Fasting normal: 70-99 mg/dL"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Run Diagnostic Prediction", type="primary", use_container_width=True)

# Build feature dictionary aligned with model's expected 15 inputs
def build_feature_vector():
    feature_dict = {
        'age': float(age),
        'hypertension': 1 if hypertension == "Yes" else 0,
        'heart_disease': 1 if heart_disease == "Yes" else 0,
        'bmi': float(bmi),
        'HbA1c_level': float(hba1c),
        'blood_glucose_level': float(glucose),
        'gender_Female': 1 if gender == "Female" else 0,
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'smoking_history_No Info': 1 if smoking_history == "No Info" else 0,
        'smoking_history_current': 1 if smoking_history == "current" else 0,
        'smoking_history_ever': 1 if smoking_history == "ever" else 0,
        'smoking_history_former': 1 if smoking_history == "former" else 0,
        'smoking_history_never': 1 if smoking_history == "never" else 0,
        'smoking_history_not current': 1 if smoking_history == "not current" else 0
    }
    return pd.DataFrame([feature_dict])

# Prediction and Output
if predict_btn and model is not None:
    input_df = build_feature_vector()
    
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    risk_score = probabilities[1] * 100

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Output")

    res_col1, res_col2 = st.columns([1.2, 0.8], gap="medium")

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h2 style="color: #f87171; margin: 0 0 10px 0;">⚠️ High Risk: Diabetes Detected</h2>
                <p style="font-size: 1.05rem; margin-bottom: 0;">The tree model classified this profile as <strong>Diabetic</strong>. Follow-up clinical screening and confirmation tests are recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h2 style="color: #34d399; margin: 0 0 10px 0;">✅ Low Risk: Non-Diabetic</h2>
                <p style="font-size: 1.05rem; margin-bottom: 0;">The model classified this profile as <strong>Non-Diabetic</strong> based on current physiological metrics and lifestyle indicators.</p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin-top:0; color:#94a3b8;">Risk Probability</h4>
            <h1 style="font-size: 2.4rem; color: {'#f87171' if risk_score >= 50 else '#34d399'}; margin: 0;">
                {risk_score:.1f}%
            </h1>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 5px;">Confidence: Non-Diabetic ({probabilities[0]*100:.1f}%) | Diabetic ({probabilities[1]*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
