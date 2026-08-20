import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(
    page_title="Diabetes Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-contrast, clean dark theme with compact spacing
st.markdown("""
<style>
    /* Remove padding and prevent scroll */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }
    
    /* Global backgrounds */
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Native container styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161b22;
        border-color: #30363d !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    
    /* Widget spacing & typography */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: 4px !important;
    }
    label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #8b949e !important;
    }
    
    /* Headings */
    .column-header {
        font-size: 0.92rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
    }
    
    /* Top Bar Badge */
    .meta-badge {
        background-color: #21262d;
        color: #c9d1d9;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid #30363d;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model_path = "Diabetes-Prediction-Project-Model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Header
head_col1, head_col2 = st.columns([1.5, 2.5])
with head_col1:
    st.markdown("<h3 style='margin: 0; padding: 0; color: #f0f6fc;'>🩺 Diabetes Risk Assessment</h3>", unsafe_allow_html=True)
with head_col2:
    st.markdown("""
    <div style='text-align: right; padding-top: 4px;'>
        <span class='meta-badge'>Model: <b>DecisionTreeClassifier</b></span>
        <span class='meta-badge'>Criterion: <b>Gini</b></span>
        <span class='meta-badge'>Max Depth: <b>10</b></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# 3-Column Layout using Streamlit native containers
col1, col2, col3 = st.columns([1, 1, 1.3], gap="medium")

# --- COLUMN 1: Demographics & Lifestyle ---
with col1:
    with st.container(border=True):
        st.markdown("<div class='column-header'>1. Patient Profile</div>", unsafe_allow_html=True)
        age = st.slider("Age (Years)", min_value=1, max_value=100, value=45)
        gender = st.selectbox("Biological Gender", options=["Female", "Male", "Other"], index=0)
        smoking_history = st.selectbox("Smoking History", options=["never", "former", "current", "not current", "ever", "No Info"], index=0)
        
        st.markdown("<div style='margin-top: 10px; font-size: 0.82rem; font-weight: 600; color: #8b949e;'>Pre-Conditions</div>", unsafe_allow_html=True)
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        with sub_c2:
            heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

# --- COLUMN 2: Clinical Biomarkers ---
with col2:
    with st.container(border=True):
        st.markdown("<div class='column-header'>2. Clinical Biomarkers</div>", unsafe_allow_html=True)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=27.32, step=0.1)
        hba1c = st.slider("HbA1c Level (%)", min_value=3.5, max_value=12.0, value=5.5, step=0.1)
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=350, value=130, step=1)
        
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡ Run Diagnostic Prediction", type="primary", use_container_width=True)

# Construct Feature Vector
def get_input_dataframe():
    features = {
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
    return pd.DataFrame([features])

# --- COLUMN 3: Model Insights & Results ---
with col3:
    with st.container(border=True):
        st.markdown("<div class='column-header'>3. Model Insights & Output</div>", unsafe_allow_html=True)
        
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
                marker=dict(color='#58a6ff')
            ))
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, visible=False),
                yaxis=dict(showgrid=False, tickfont=dict(color='#8b949e', size=11))
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Prediction Evaluation
        if model is not None:
            input_df = get_input_dataframe()
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            risk_pct = prob[1] * 100
            
            if pred == 1:
                st.markdown(f"""
                <div style="background-color: rgba(248, 81, 73, 0.15); border: 1.5px solid #f85149; border-radius: 8px; padding: 12px; text-align: center; margin-top: 4px;">
                    <div style="color: #ff7b72; font-weight: 700; font-size: 0.95rem;">⚠️ HIGH RISK: DIABETIC</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #f85149; margin: 2px 0;">{risk_pct:.1f}% Risk</div>
                    <div style="font-size: 0.75rem; color: #ffa198;">Confidence: Diabetic ({prob[1]*100:.1f}%) | Non-Diabetic ({prob[0]*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(46, 160, 67, 0.15); border: 1.5px solid #2ea043; border-radius: 8px; padding: 12px; text-align: center; margin-top: 4px;">
                    <div style="color: #56d364; font-weight: 700; font-size: 0.95rem;">✅ LOW RISK: NON-DIABETIC</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #3fb950; margin: 2px 0;">{risk_pct:.1f}% Risk</div>
                    <div style="font-size: 0.75rem; color: #7ee787;">Confidence: Non-Diabetic ({prob[0]*100:.1f}%) | Diabetic ({prob[1]*100:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
