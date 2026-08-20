import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page configuration to wide mode
st.set_page_config(
    page_title="Diabetes Risk Predictor Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for zero-scroll full viewport dashboard
st.markdown("""
<style>
    /* Viewport lock & Zero-scroll styling */
    html, body, [data-testid="stAppViewContainer"], .main {
        height: 100vh;
        overflow: hidden !important;
        background-color: #0b0f19 !important;
        color: #f3f4f6 !important;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Header layout */
    .header-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 8px;
        margin-bottom: 12px;
        border-bottom: 1px solid #1f2937;
    }
    
    /* Card Styles */
    .card-panel {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 14px;
        height: calc(100vh - 95px);
        box-sizing: border-box;
        overflow-y: hidden;
    }
    
    /* Typography & Compact Widgets */
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    
    .badge {
        background: #1e293b;
        color: #94a3b8;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #334155;
    }

    /* Result Card Styles */
    .result-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.35) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    
    .result-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.35) 100%);
        border: 1.5px solid #10b981;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }

    /* Streamlit widget size reduction */
    .stNumberInput, .stSelectbox, .stSlider, .stRadio {
        margin-bottom: -10px !important;
    }
    label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained Model
@st.cache_resource
def load_model():
    model_path = "Diabetes-Prediction-Project-Model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Header bar
st.markdown("""
<div class="header-box">
    <div>
        <span style="font-size: 1.25rem; font-weight: 800; color: #f9fafb;">🧬 Diabetes Intelligence Dashboard</span>
    </div>
    <div>
        <span class="badge">Model: <strong>DecisionTreeClassifier</strong></span>
        <span class="badge" style="margin-left: 6px;">Criterion: <strong>Gini Impurity</strong></span>
        <span class="badge" style="margin-left: 6px;">Max Depth: <strong>10</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Grid Layout: 3 Columns fitting standard displays
col1, col2, col3 = st.columns([1.1, 1.1, 1.3], gap="medium")

# --- COLUMN 1: Demographics & Lifestyle ---
with col1:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">1. Patient Profile</div>', unsafe_allow_html=True)
    
    age = st.slider("Age (Years)", min_value=1, max_value=100, value=45, step=1)
    gender = st.selectbox("Biological Gender", options=["Female", "Male", "Other"], index=0)
    smoking_history = st.selectbox("Smoking History", options=["never", "former", "current", "not current", "ever", "No Info"], index=0)
    
    st.markdown('<div class="section-title" style="margin-top: 14px;">Pre-Conditions</div>', unsafe_allow_html=True)
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        hypertension = st.radio("Hypertension", options=["No", "Yes"], horizontal=True)
    with sub_col2:
        heart_disease = st.radio("Heart Disease", options=["No", "Yes"], horizontal=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- COLUMN 2: Clinical Biomarkers ---
with col2:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">2. Clinical Biomarkers</div>', unsafe_allow_html=True)
    
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=27.32, step=0.1)
    hba1c = st.slider("HbA1c Level (%)", min_value=3.5, max_value=12.0, value=5.5, step=0.1)
    glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=350, value=130, step=1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Run Diagnostic Prediction", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature vector constructor
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

# --- COLUMN 3: Analytics & Diagnostic Output ---
with col3:
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">3. Model Insights & Output</div>', unsafe_allow_html=True)
    
    # Feature Importance Chart (Compact)
    if model is not None and hasattr(model, 'feature_importances_'):
        feat_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True).tail(5)
        
        st.markdown("<span style='font-size: 0.75rem; color:#94a3b8;'>Top 5 Decision Drivers (Gini Importance):</span>", unsafe_allow_html=True)
        st.bar_chart(feat_df.set_index('Feature'), height=130, color="#38bdf8")

    # Prediction Output Area
    if model is not None:
        input_df = build_feature_vector()
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        risk_score = probabilities[1] * 100

        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div style="color: #f87171; font-weight:700; font-size:1rem;">⚠️ HIGH RISK: DIABETIC</div>
                <div style="font-size: 1.5rem; font-weight:800; color: #ef4444; margin: 4px 0;">{risk_score:.1f}% Risk</div>
                <div style="font-size:0.75rem; color:#fca5a5;">Confidence: Diabetic ({probabilities[1]*100:.1f}%) | Non-Diabetic ({probabilities[0]*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div style="color: #34d399; font-weight:700; font-size:1rem;">✅ LOW RISK: NON-DIABETIC</div>
                <div style="font-size: 1.5rem; font-weight:800; color: #10b981; margin: 4px 0;">{risk_score:.1f}% Risk</div>
                <div style="font-size:0.75rem; color:#6ee7b7;">Confidence: Non-Diabetic ({probabilities[0]*100:.1f}%) | Diabetic ({probabilities[1]*100:.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Model file not detected.")

    st.markdown('</div>', unsafe_allow_html=True)
