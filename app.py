import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Diabetes Risk Lab | AI HUD",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk / Bio-Lab HUD Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    /* Lock viewport height and eliminate scrollbars */
    html, body, [data-testid="stAppViewContainer"], .main {
        background: radial-gradient(circle at 50% 10%, #081a26 0%, #030a10 100%) !important;
        color: #e2f1ff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        overflow: hidden !important;
        height: 100vh !important;
    }

    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* Sci-Fi Sidebar */
    section[data-testid="stSidebar"] {
        background: #040e17 !important;
        border-right: 1px solid rgba(0, 240, 255, 0.15) !important;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.8rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Container Glass Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(6, 20, 32, 0.85) !important;
        border: 1px solid rgba(0, 240, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.5) !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(0, 240, 255, 0.05) !important;
        border: 1px solid rgba(0, 240, 255, 0.15) !important;
        border-radius: 6px !important;
        padding: 4px 8px !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        color: #7aa5c2 !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem !important;
        color: #00f0ff !important;
    }

    /* Header text */
    .hud-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 700;
        color: #00f0ff;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    /* Compact inputs */
    .stSlider, .stNumberInput, .stSelectbox, .stRadio {
        margin-bottom: -6px !important;
    }
    label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #7aa5c2 !important;
    }
    div[data-testid="stAlert"] {
        padding: 6px 10px !important;
        margin-bottom: 2px !important;
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

# Initialize session state for persistent evaluation after button click
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.risk_score = 0.0
    st.session_state.prediction = 0
    st.session_state.raw_prob = 0.0

# --- SIDEBAR: Controls ---
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 4px;'><span style='font-size: 1.6rem;'>🧪</span><br><b style='font-family: JetBrains Mono; color: #00f0ff; font-size: 0.95rem;'>PATIENT VITALS</b></div>", unsafe_allow_html=True)

    age = st.slider("👤 Age (Years)", min_value=1, max_value=100, value=45)
    gender = st.selectbox("⚧ Biological Gender", options=["Female", "Male", "Other"], index=0)
    smoking_history = st.selectbox("🚬 Smoking History", options=["never", "former", "current", "not current", "ever", "No Info"], index=0)
    
    st.markdown("<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)
    bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=27.32, step=0.1)
    hba1c = st.slider("🩸 HbA1c Level (%)", min_value=3.5, max_value=12.0, value=5.5, step=0.1)
    glucose = st.number_input("🔬 Blood Glucose (mg/dL)", min_value=50, max_value=350, value=130, step=1)
    
    st.markdown("<div style='font-size: 0.75rem; font-weight:600; color: #7aa5c2; margin-top: 4px;'>Preconditions</div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
    with sc2:
        heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)

    st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ ANALYZE PATIENT RISK", type="primary", use_container_width=True)

# Build feature dictionary aligned with the model's expected 15 inputs
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

# Smooth continuous risk calculation
def calculate_continuous_risk(tree_prob, hba1c_val, glucose_val, bmi_val, age_val, hyp_val, hd_val):
    hba1c_risk = 1.0 / (1.0 + np.exp(-1.8 * (hba1c_val - 6.2)))
    glucose_risk = 1.0 / (1.0 + np.exp(-0.04 * (glucose_val - 150.0)))
    bmi_risk = 1.0 / (1.0 + np.exp(-0.15 * (bmi_val - 29.0)))
    age_risk = 1.0 / (1.0 + np.exp(-0.05 * (age_val - 50.0)))
    condition_bump = (0.08 if hyp_val == "Yes" else 0.0) + (0.08 if hd_val == "Yes" else 0.0)

    clinical_continuous = (
        0.35 * hba1c_risk +
        0.35 * glucose_risk +
        0.12 * bmi_risk +
        0.10 * age_risk +
        condition_bump
    )
    blended = 0.40 * tree_prob + 0.60 * clinical_continuous
    return float(np.clip(blended * 100, 1.0, 99.0))

# Execute prediction when button is clicked
if analyze_btn and model is not None:
    input_df = build_feature_df()
    raw_pred = model.predict(input_df)[0]
    raw_prob = model.predict_proba(input_df)[0][1]
    smooth_score = calculate_continuous_risk(raw_prob, hba1c, glucose, bmi, age, hypertension, heart_disease)
    
    st.session_state.analyzed = True
    st.session_state.risk_score = smooth_score
    st.session_state.prediction = 1 if smooth_score >= 50.0 else 0
    st.session_state.raw_prob = raw_prob

# --- TOP HEADER ---
st.markdown("<div style='background: rgba(4, 18, 28, 0.75); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 8px; padding: 6px 14px; text-align: center; margin-bottom: 8px;'><span style='font-family: JetBrains Mono; font-size: 1.15rem; font-weight: 800; color: #00f0ff; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,240,255,0.4);'>🧬 DIABETES RISK LAB // DECISION TREE ENGINE</span><br><span style='font-family: JetBrains Mono; font-size: 0.7rem; color: #648ba6;'>AI-Assisted Diagnostic Telemetry & Clinical Feature Importance</span></div>", unsafe_allow_html=True)

col_panel, col_diag = st.columns([1.15, 1], gap="small")

# --- COLUMN 1: Live Patient Biomarkers ---
with col_panel:
    with st.container(border=True):
        st.markdown("<div class='hud-title'>🧪 Patient Telemetry</div>", unsafe_allow_html=True)
        
        glucose_color = "#ff3b6b" if glucose >= 140 else ("#ffb800" if glucose >= 100 else "#00ffa3")
        hba1c_color = "#ff3b6b" if hba1c >= 6.5 else ("#ffb800" if hba1c >= 5.7 else "#00ffa3")
        bmi_color = "#ff3b6b" if bmi >= 30.0 else ("#ffb800" if bmi >= 25.0 else "#00ffa3")
        age_color = "#ffb800" if age >= 60 else "#00f0ff"

        bar_metrics = [
            {"name": "Age", "display": f"{age} yrs", "val": min(100, age), "color": age_color},
            {"name": "BMI", "display": f"{bmi:.1f} kg/m²", "val": min(100, int((bmi / 50) * 100)), "color": bmi_color},
            {"name": "HbA1c", "display": f"{hba1c:.1f}%", "val": min(100, int((hba1c / 12) * 100)), "color": hba1c_color},
            {"name": "Glucose", "display": f"{glucose} mg/dL", "val": min(100, int((glucose / 300) * 100)), "color": glucose_color},
        ]
        
        y_labels = [m["name"] for m in bar_metrics]
        x_vals = [m["val"] for m in bar_metrics]
        display_texts = [f"<b>{m['display']}</b>  " for m in bar_metrics]
        bar_colors = [m["color"] for m in bar_metrics]

        fig_telemetry = go.Figure()
        
        fig_telemetry.add_trace(go.Bar(
            y=y_labels,
            x=[100] * 4,
            orientation='h',
            marker=dict(color='rgba(14, 38, 56, 0.7)', line=dict(color='rgba(0, 240, 255, 0.15)', width=1)),
            hoverinfo='none',
            showlegend=False
        ))
        
        fig_telemetry.add_trace(go.Bar(
            y=y_labels,
            x=x_vals,
            orientation='h',
            text=display_texts,
            textposition='inside',
            insidetextanchor='start',
            textfont=dict(family='JetBrains Mono', size=10, color='#040e17'),
            marker=dict(color=bar_colors, line=dict(color='rgba(255, 255, 255, 0.6)', width=1)),
            hoverinfo='none',
            showlegend=False
        ))

        fig_telemetry.update_layout(
            barmode='overlay',
            height=125,
            margin=dict(l=5, r=5, t=2, b=2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, visible=False, range=[0, 100]),
            yaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', color='#00f0ff', size=11))
        )
        st.plotly_chart(fig_telemetry, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<div style='margin-top: -6px;'></div>", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Hypertension", hypertension)
        with m_col2:
            st.metric("Heart Disease", heart_disease)

# --- COLUMN 2: Diagnostic Evaluation & Continuous Donut Gauge ---
with col_diag:
    with st.container(border=True):
        st.markdown("<div class='hud-title'>🔬 Diagnostic Evaluation</div>", unsafe_allow_html=True)
        
        if not st.session_state.analyzed:
            st.info("👈 Set inputs and click **ANALYZE PATIENT RISK**.")
            gauge_val = 0.0
            gauge_color = '#00f0ff'
            display_text = "READY"
        else:
            gauge_val = st.session_state.risk_score
            gauge_color = '#ff3b6b' if gauge_val >= 50 else '#00ffa3'
            display_text = f"{gauge_val:.1f}%"
            
            if st.session_state.prediction == 1:
                st.error("⚠️ **HIGH DIABETES RISK** — Elevated risk profile.")
            else:
                st.success("✅ **LOW DIABETES RISK** — Nominal biomarkers.")

        # Plotly Donut Gauge
        fig_gauge = go.Figure(go.Pie(
            values=[gauge_val, max(0.0, 100.0 - gauge_val)],
            hole=0.76,
            sort=False,
            direction='clockwise',
            marker=dict(colors=[gauge_color, 'rgba(14, 38, 56, 0.7)'], line=dict(color='#030a10', width=2)),
            textinfo='none',
            hoverinfo='none'
        ))
        fig_gauge.update_layout(
            showlegend=False,
            height=110,
            margin=dict(l=0, r=0, t=2, b=2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            annotations=[
                dict(
                    text=f"<span style='font-family: JetBrains Mono; font-size: 1.3rem; font-weight: 800; color: {gauge_color};'>{display_text}</span><br><span style='font-size: 0.65rem; color: #7aa5c2;'>Risk Score</span>",
                    x=0.5, y=0.5,
                    showarrow=False
                )
            ]
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

# --- BOTTOM: Feature Importance Chart ---
if model is not None and hasattr(model, 'feature_importances_'):
    with st.container(border=True):
        st.markdown("<div class='hud-title' style='margin-bottom: 2px;'>📊 Decision Tree Feature Importance Matrix</div>", unsafe_allow_html=True)
        
        feat_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True).tail(5)
        
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
            height=105,
            margin=dict(l=10, r=10, t=2, b=2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', tickfont=dict(color='#648ba6', size=9)),
            yaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', color='#b0d4ec', size=10))
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
