import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Diabetes Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean High-Contrast Dark Theme
st.markdown("""
<style>
    /* Prevent extra margins and scrolling */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Background and Font */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Native container borders */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111827;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        padding: 12px 14px !important;
    }
    
    /* Compact widget labels and spacing */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: -4px !important;
    }
    
    label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load trained Model
@st.cache_resource
def load_model():
    model_path = "Diabetes-Prediction-Project-Model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Header with Model Specs
header_left, header_right = st.columns([1.5, 2.5])
with header_left:
    st.markdown("<h3 style='margin:0; color:#38bdf8;'>🩺 Diabetes Diagnostic System</h3>", unsafe_allow_html=True)
with header_right:
    st.markdown("""
    <div style='text-align: right; padding-top: 4px;'>
        <span style='background:#1e293b; color:#94a3b8; padding:3px 8px; border-radius:4px; font-size:0.78rem; border:1px solid #334155;'>Model: <b>DecisionTreeClassifier</b></span>
        <span style='background:#1e293b; color:#94a3b8; padding:3px 8px; border-radius:4px; font-size:0.78rem; border:1px solid #334155; margin-left:6px;'>Criterion: <b>Gini</b></span>
        <span style='background:#1e293b; color:#94a3b8; padding:3px 8px; border-radius:4px; font-size:0.78rem; border:1px solid #334155; margin-left:6px;'>Max Depth: <b>10</b></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

# 3-Column Single Frame Layout
col1, col2, col3 = st.columns([1, 1, 1.25], gap="small")

# --- COLUMN 1: Demographics & Lifestyle ---
with col1:
    with st.container(border=True):
        st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:8px;'>1. Patient Profile</div>", unsafe_allow_html=True)
        age = st.slider("Age (Years)", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Biological Gender", options=["Female", "Male", "Other"], index=0)
        smoking_history = st.selectbox("Smoking History", options=["never", "former", "current", "not current", "ever", "No Info"], index=0)
        
        st.markdown("<div style='font-size:0.78rem; font-weight:600; color:#94a3b8; margin-top:8px;'>Pre-Conditions</div>", unsafe_allow_html=True)
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        with sub_c2:
            heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

# --- COLUMN 2: Clinical Biomarkers ---
with col2:
    with st.container(border=True):
        st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:8px;'>2. Biomarkers</div>", unsafe_allow_html=True)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=27.32, step=0.1)
        hba1c = st.slider("HbA1c Level (%)", min_value=3.5, max_value=12.0, value=5.5, step=0.1)
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=350, value=130, step=1)
        
        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Run Diagnostic Prediction", type="primary", use_container_width=True)

# Feature dictionary aligned with model's 15 expected columns
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

# --- COLUMN 3: Model Insights & Diagnostic Output ---
with col3:
    with st.container(border=True):
        st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:4px;'>3. Feature Importance & Output</div>", unsafe_allow_html=True)
        
        # Horizontal Feature Importance Plot
        if model is not None and hasattr(model, 'feature_importances_'):
            feat_df = pd.DataFrame({
                'Feature': model.feature_names_in_,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True).tail(5)
            
            fig = go.Figure(go.Bar(
                x=feat_df['Importance'],
                y=feat_df['Feature'],
                orientation='h',
                marker=dict(color='#38bdf8')
            ))
            fig.update_layout(
                height=140,
                margin=dict(l=5, r=5, t=5, b=5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=11))
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        # Prediction Output
        if model is not None:
            input_df = build_feature_vector()
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            risk_score = probabilities[1] * 100

            if prediction == 1:
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.15); border: 1.5px solid #ef4444; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #f87171; font-weight:700; font-size:0.9rem;">⚠️ HIGH RISK: DIABETIC</div>
                    <div style="font-size: 1.6rem; font-weight:800; color: #ef4444; margin: 2px 0;">{risk_score:.1f}% Risk</div>
                    <div style="font-size:0.75rem; color:#fca5a5;">Confidence: Diabetic ({probabilities[1]*100:.1f}%) | Non-Diabetic ({probabilities[0]*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(16, 185, 129, 0.15); border: 1.5px solid #10b981; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="color: #34d399; font-weight:700; font-size:0.9rem;">✅ LOW RISK: NON-DIABETIC</div>
                    <div style="font-size: 1.6rem; font-weight:800; color: #10b981; margin: 2px 0;">{risk_score:.1f}% Risk</div>
                    <div style="font-size:0.75rem; color:#6ee7b7;">Confidence: Non-Diabetic ({probabilities[0]*100:.1f}%) | Diabetic ({probabilities[1]*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
