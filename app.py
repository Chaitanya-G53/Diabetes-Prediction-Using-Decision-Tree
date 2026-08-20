import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import textwrap

st.set_page_config(
    page_title="Diabetes Risk Lab | AI HUD",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Sci-Fi CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"], .main {
        background: radial-gradient(circle at 50% 10%, #081a26 0%, #030a10 100%) !important;
        color: #e2f1ff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #040e17 !important;
        border-right: 1px solid rgba(0, 240, 255, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }

    /* Terminal Header */
    .terminal-header {
        background: rgba(4, 18, 28, 0.75);
        border: 1px solid rgba(0, 240, 255, 0.2);
        border-radius: 12px;
        padding: 12px 20px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.08);
        margin-bottom: 16px;
    }

    .terminal-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 2px;
        color: #00f0ff;
        text-shadow: 0 0 12px rgba(0, 240, 255, 0.5);
    }

    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #648ba6;
        letter-spacing: 1px;
        margin-top: 3px;
    }

    /* HUD Glass Cards */
    .hud-card {
        background: rgba(6, 20, 32, 0.85);
        border: 1px solid rgba(0, 240, 255, 0.18);
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 12px;
    }

    .card-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        font-weight: 700;
        color: #00f0ff;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    /* Metric Vitals Meter Row */
    .vital-row {
        margin-bottom: 12px;
    }

    .vital-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 4px;
        color: #b0d4ec;
    }

    .vital-value {
        font-family: 'JetBrains Mono', monospace;
        color: #00f0ff;
    }

    .vital-track {
        height: 7px;
        background: rgba(14, 38, 56, 0.8);
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(0, 240, 255, 0.1);
    }

    .vital-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #00f0ff, #00ffa3);
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
    }

    /* Result Badges */
    .result-badge-low {
        background: rgba(0, 255, 163, 0.1);
        border: 1.5px solid #00ffa3;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        color: #00ffa3;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.2);
    }

    .result-badge-high {
        background: rgba(255, 59, 107, 0.1);
        border: 1.5px solid #ff3b6b;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        color: #ff3b6b;
        box-shadow: 0 0 20px rgba(255, 59, 107, 0.2);
    }

    /* Input fine-tuning */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: 2px !important;
    }

    label {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #7aa5c2 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to prevent Markdown from interpreting leading spaces as code blocks
def render_html(raw_html: str):
    st.markdown(textwrap.dedent(raw_html).strip(), unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = "Diabetes-Prediction-Project-Model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Sidebar: Controls
with st.sidebar:
    render_html("""
    <div style="text-align: center; margin-bottom: 12px;">
        <div style="font-size: 2rem;">🧪</div>
        <div style="font-family: JetBrains Mono; font-weight: 800; font-size: 1.1rem; color: #00f0ff;">PATIENT VITALS</div>
        <div style="font-size: 0.75rem; color: #648ba6;">Configure telemetry & biomarkers</div>
    </div>
    """)

    age = st.slider("👤 Age (Years)", min_value=1, max_value=100, value=45)
    gender = st.selectbox("⚧ Biological Gender", options=["Female", "Male", "Other"], index=0)
    smoking_history = st.selectbox("🚬 Smoking History", options=["never", "former", "current", "not current", "ever", "No Info"], index=0)
    
    st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
    bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=27.32, step=0.1)
    hba1c = st.slider("🩸 HbA1c Level (%)", min_value=3.5, max_value=12.0, value=5.5, step=0.1)
    glucose = st.number_input("🔬 Blood Glucose (mg/dL)", min_value=50, max_value=350, value=130, step=1)
    
    st.markdown("<div style='font-size: 0.8rem; font-weight:600; color: #7aa5c2; margin-top:6px;'>Cardiovascular Preconditions</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
    with sc2:
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

# Build feature dictionary
def build_feature_df():
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

# Predict
input_df = build_feature_df()
if model is not None:
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    risk_score = probabilities[1] * 100
else:
    prediction, risk_score, probabilities = 0, 0.0, [1.0, 0.0]

# Terminal Header
render_html("""
<div class="terminal-header">
    <div class="terminal-title">🧬 DIABETES RISK LAB // DECISION TREE ENGINE</div>
    <div class="terminal-subtitle">AI-Assisted Diagnostic Telemetry & Feature Importance Matrix</div>
</div>
""")

col_panel, col_diag = st.columns([1.1, 1], gap="medium")

# Biomarker Calculations
glucose_pct = min(100, int((glucose / 300) * 100))
hba1c_pct = min(100, int(((hba1c - 3.5) / 8.5) * 100))
bmi_pct = min(100, int(((bmi - 10) / 45) * 100))
age_pct = min(100, int(age))

with col_panel:
    render_html(f"""
    <div class="hud-card">
        <div class="card-title">🧪 Live Patient Biomarkers</div>
        
        <div class="vital-row">
            <div class="vital-labels">
                <span>Blood Glucose Level</span>
                <span class="vital-value">{glucose} mg/dL</span>
            </div>
            <div class="vital-track"><div class="vital-fill" style="width: {glucose_pct}%;"></div></div>
        </div>

        <div class="vital-row">
            <div class="vital-labels">
                <span>HbA1c Level</span>
                <span class="vital-value">{hba1c:.1f}%</span>
            </div>
            <div class="vital-track"><div class="vital-fill" style="width: {hba1c_pct}%;"></div></div>
        </div>

        <div class="vital-row">
            <div class="vital-labels">
                <span>Body Mass Index (BMI)</span>
                <span class="vital-value">{bmi:.1f} kg/m²</span>
            </div>
            <div class="vital-track"><div class="vital-fill" style="width: {bmi_pct}%;"></div></div>
        </div>

        <div class="vital-row">
            <div class="vital-labels">
                <span>Age Cohort</span>
                <span class="vital-value">{age} yrs</span>
            </div>
            <div class="vital-track"><div class="vital-fill" style="width: {age_pct}%;"></div></div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px;">
            <div style="background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.1); border-radius: 8px; padding: 8px; text-align: center;">
                <div style="font-size: 0.72rem; color: #7aa5c2;">Hypertension</div>
                <div style="font-family: JetBrains Mono; font-weight:700; color: {'#ff3b6b' if hypertension == 'Yes' else '#00ffa3'};">{hypertension}</div>
            </div>
            <div style="background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.1); border-radius: 8px; padding: 8px; text-align: center;">
                <div style="font-size: 0.72rem; color: #7aa5c2;">Heart Disease</div>
                <div style="font-family: JetBrains Mono; font-weight:700; color: {'#ff3b6b' if heart_disease == 'Yes' else '#00ffa3'};">{heart_disease}</div>
            </div>
        </div>
    </div>
    """)

with col_diag:
    # Diagnosis Result Header
    if prediction == 1:
        result_badge = """
        <div class="result-badge-high">
            <div style="font-family: JetBrains Mono; font-weight: 800; font-size: 1rem;">⚠️ HIGH DIABETES RISK</div>
            <div style="font-size: 0.75rem; margin-top: 2px;">Clinical profile aligns with positive Diabetic classification.</div>
        </div>
        """
    else:
        result_badge = """
        <div class="result-badge-low">
            <div style="font-family: JetBrains Mono; font-weight: 800; font-size: 1rem;">✅ LOW DIABETES RISK</div>
            <div style="font-size: 0.75rem; margin-top: 2px;">Indicators fall within nominal expected ranges.</div>
        </div>
        """

    render_html(f"""
    <div class="hud-card">
        <div class="card-title">🔬 Diagnostic Evaluation</div>
        {result_badge}
    </div>
    """)

    # Donut Risk Gauge
    gauge_color = '#ff3b6b' if risk_score >= 50 else '#00ffa3'
    fig_gauge = go.Figure(go.Pie(
        values=[risk_score, 100 - risk_score],
        hole=0.75,
        sort=False,
        direction='clockwise',
        marker=dict(colors=[gauge_color, 'rgba(14, 38, 56, 0.6)'], line=dict(color='#030a10', width=2)),
        textinfo='none',
        hoverinfo='none'
    ))
    
    fig_gauge.update_layout(
        showlegend=False,
        height=135,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=f"<span style='font-family: JetBrains Mono; font-size: 1.6rem; font-weight: 800; color: {gauge_color};'>{risk_score:.0f}%</span><br><span style='font-size: 0.72rem; color: #7aa5c2;'>Risk Score</span>",
                x=0.5, y=0.5,
                showarrow=False
            )
        ]
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

# Bottom: Feature Importance Matrix
if model is not None and hasattr(model, 'feature_importances_'):
    render_html("""
    <div class="hud-card" style="margin-top: 6px; padding: 12px 18px;">
        <div class="card-title" style="margin-bottom: 4px;">📊 Decision Tree Feature Importance Matrix</div>
    </div>
    """)
    
    feat_df = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True).tail(6)
    
    fig_bar = go.Figure(go.Bar(
        x=feat_df['Importance'],
        y=feat_df['Feature'],
        orientation='h',
        marker=dict(
            color=feat_df['Importance'],
            colorscale=[[0, '#00a3ff'], [1, '#00ffa3']],
            line=dict(color='rgba(0, 240, 255, 0.4)', width=1)
        )
    ))
    
    fig_bar.update_layout(
        height=125,
        margin=dict(l=10, r=10, t=5, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#648ba6', size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', color='#b0d4ec', size=11))
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
