from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HeartPulse – AI Heart Risk Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_pipeline.pkl"
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

FEATURES = [
    "ca",
    "cp",
    "exang",
    "thalach",
    "oldpeak",
    "thal",
    "slope",
    "sex",
    "age",
    "restecg",
    "chol",
]

# ============================================================================
# CUSTOM CSS & STYLING
# ============================================================================

CUSTOM_CSS = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    :root {
        --primary-red: #e63946;
        --primary-red-light: #f8d7da;
        --primary-red-dark: #c1121f;
        --success-green: #06a77d;
        --success-green-light: #d4f4e8;
        --bg-white: #ffffff;
        --bg-light: #f9fafb;
        --text-primary: #1a1a1a;
        --text-secondary: #6b7280;
        --border-light: #e5e7eb;
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
        --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.12);
        --shadow-lg: 0 16px 40px rgba(0, 0, 0, 0.14);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-white) !important;
    }

    .stApp {
        background: linear-gradient(135deg, var(--bg-white) 0%, #fafafa 100%);
    }

    [data-testid="stMainBlockContainer"] {
        padding: 2rem 1rem !important;
        max-width: 720px;
        margin: 0 auto !important;
    }

    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    p, span, label {
        color: var(--text-primary);
    }

    /* ========== HERO SECTION ========== */
    .hero-container {
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInDown 0.6s ease-out;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--primary-red-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* ========== CARDS ========== */
    .card {
        background: var(--bg-white);
        border: 1px solid var(--border-light);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1);
        animation: fadeInUp 0.6s ease-out;
    }

    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ========== FORM INPUTS ========== */
    [data-testid="stNumberInput"],
    [data-testid="stSelectbox"],
    [data-testid="stSlider"] {
        margin-bottom: 1.2rem;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] select,
    [data-baseweb="select"] > div {
        background-color: var(--bg-light) !important;
        border: 1.5px solid var(--border-light) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSelectbox"] select:focus,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--primary-red) !important;
        box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.1) !important;
    }

    [data-testid="stNumberInput"] label p,
    [data-testid="stSelectbox"] label p,
    [data-testid="stSlider"] label p {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* ========== GRID LAYOUT ========== */
    .input-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 1rem;
    }

    @media (max-width: 640px) {
        .input-grid {
            grid-template-columns: 1fr;
        }
    }

    /* ========== BUTTONS ========== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--primary-red-dark) 100%);
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        height: auto !important;
        transition: all 0.3s cubic-bezier(0.23, 1, 0.320, 1) !important;
        box-shadow: 0 4px 12px rgba(230, 57, 70, 0.2) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(230, 57, 70, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ========== RESULT CARD ========== */
    .result-card {
        background: linear-gradient(135deg, var(--bg-white) 0%, var(--bg-light) 100%);
        border: 2px solid var(--border-light);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
        animation: slideUp 0.5s ease-out;
        margin-top: 2rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .risk-badge.high {
        background-color: var(--primary-red-light);
        color: var(--primary-red-dark);
        border: 2px solid var(--primary-red);
    }

    .risk-badge.low {
        background-color: var(--success-green-light);
        color: var(--success-green);
        border: 2px solid var(--success-green);
    }

    .result-text {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 1rem 0;
        letter-spacing: -0.02em;
    }

    .confidence-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-red) 0%, var(--primary-red-dark) 100%) !important;
        border-radius: 8px !important;
        height: 8px !important;
    }

    /* ========== SIDEBAR MINIMAL ========== */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* ========== ANIMATIONS ========== */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* ========== FOOTER ========== */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-light);
        font-size: 0.85rem;
        color: var(--text-secondary);
        animation: fadeInUp 1s ease-out 0.8s both;
    }

    .footer a {
        color: var(--primary-red);
        text-decoration: none;
        font-weight: 600;
        transition: color 0.2s ease;
    }

    .footer a:hover {
        color: var(--primary-red-dark);
        text-decoration: underline;
    }

    /* ========== ALERTS ========== */
    .stAlert {
        border-radius: 10px !important;
        border-left: 4px solid !important;
    }

    .stAlert[data-testid="stAlert"] > div {
        padding: 1rem !important;
    }

    /* ========== COLUMNS ========== */
    [data-testid="column"] {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
"""


def inject_css() -> None:
    """Inject custom CSS for premium UI."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path) -> Tuple:
    """Load trained model and feature list from pickle."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_order = payload.get("features", FEATURES)
    else:
        model = payload
        feature_order = FEATURES
    return model, feature_order


@st.cache_data(show_spinner=False)
def load_data_profile(path: Path) -> Dict:
    """Load data profile for input defaults."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    profile = {}
    for col in df.columns:
        if col == "target":
            continue
        profile[col] = {
            "median": float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0,
            "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else 0,
            "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else 1,
        }
    return profile


def get_defaults(profile: Dict) -> Dict:
    """Generate default input values from data profile."""
    def v(name: str, fallback: float) -> float:
        return profile.get(name, {}).get("median", fallback)

    return {
        "age": int(round(v("age", 55))),
        "sex": int(round(v("sex", 1))),
        "cp": int(round(v("cp", 1))),
        "thalach": int(round(v("thalach", 150))),
        "oldpeak": float(round(v("oldpeak", 1.0), 1)),
        "ca": int(round(v("ca", 0))),
        "exang": int(round(v("exang", 0))),
        "slope": int(round(v("slope", 1))),
        "thal": int(round(v("thal", 2))),
        "restecg": int(round(v("restecg", 1))),
        "chol": int(round(v("chol", 240))),
    }


def validate_inputs(values: Dict) -> List[str]:
    """Validate input ranges."""
    errors = []
    if not 18 <= values["age"] <= 100:
        errors.append("Age must be between 18 and 100 years")
    if not 100 <= values["chol"] <= 600:
        errors.append("Cholesterol must be between 100 and 600 mg/dL")
    if not 60 <= values["thalach"] <= 230:
        errors.append("Max heart rate must be between 60 and 230 bpm")
    if not 0.0 <= values["oldpeak"] <= 8.0:
        errors.append("ST depression must be between 0.0 and 8.0")
    return errors


def build_prediction_frame(values: Dict, features: List[str]) -> pd.DataFrame:
    """Create DataFrame for model prediction."""
    row = {k: values[k] for k in features}
    return pd.DataFrame([row], columns=features)


def make_prediction(model, frame: pd.DataFrame) -> Tuple:
    """Run model inference."""
    pred = int(model.predict(frame)[0])
    confidence = None
    prob_pos = None
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        prob_pos = float(proba[1])
        confidence = float(proba[pred])
    
    return pred, confidence, prob_pos


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application logic."""
    inject_css()

    # Header
    st.markdown(
        '<div class="hero-container">'
        '<h1 class="hero-title">❤️ HeartPulse</h1>'
        '<p class="hero-subtitle">AI-Powered Heart Disease Risk Assessment</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load model and data
    if not MODEL_PATH.exists():
        st.error("❌ Model file not found. Please ensure heart_disease_pipeline.pkl exists.")
        st.stop()

    model, feature_order = load_pipeline(MODEL_PATH)
    profile = load_data_profile(DATA_PATH)
    defaults = get_defaults(profile)

    # Initialize session state
    if "form_values" not in st.session_state:
        st.session_state.form_values = defaults.copy()
    if "result" not in st.session_state:
        st.session_state.result = None

    # Input Form Card
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">📋 Patient Information</div>',
            unsafe_allow_html=True,
        )

        vals = st.session_state.form_values.copy()

        # 2-column grid for inputs
        st.markdown('<div class="input-grid">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            vals["age"] = st.number_input(
                "Age (years)",
                min_value=18,
                max_value=100,
                value=int(vals["age"]),
                step=1,
            )
            vals["sex"] = st.selectbox(
                "Sex",
                [0, 1],
                index=int(vals["sex"]),
                format_func=lambda x: "Female" if x == 0 else "Male",
            )
            vals["cp"] = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                index=int(vals["cp"]),
                format_func=lambda x: ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][x],
            )
            vals["chol"] = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=100,
                max_value=600,
                value=int(vals["chol"]),
                step=5,
            )
            vals["oldpeak"] = st.number_input(
                "ST Depression (oldpeak)",
                min_value=0.0,
                max_value=8.0,
                value=float(vals["oldpeak"]),
                step=0.1,
            )
            vals["slope"] = st.selectbox(
                "ST Slope",
                [0, 1, 2],
                index=int(vals["slope"]),
                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
            )

        with col2:
            vals["thalach"] = st.number_input(
                "Max Heart Rate (bpm)",
                min_value=60,
                max_value=230,
                value=int(vals["thalach"]),
                step=1,
            )
            vals["exang"] = st.selectbox(
                "Exercise Induced Angina",
                [0, 1],
                index=int(vals["exang"]),
                format_func=lambda x: "No" if x == 0 else "Yes",
            )
            vals["ca"] = st.selectbox(
                "Major Vessels (ca)",
                [0, 1, 2, 3, 4],
                index=int(vals["ca"]),
            )
            vals["thal"] = st.selectbox(
                "Thalassemia",
                [0, 1, 2, 3],
                index=int(vals["thal"]),
                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible", "Severe"][x],
            )
            vals["restecg"] = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                index=int(vals["restecg"]),
                format_func=lambda x: ["Normal", "ST-T Abnormal", "LV Hypertrophy"][x],
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.form_values = vals
        errors = validate_inputs(vals)

        if errors:
            for error in errors:
                st.warning(f"⚠️ {error}")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_clicked = st.button(
                "🔍 Predict Risk",
                use_container_width=True,
                disabled=bool(errors),
            )

        with col_btn2:
            if st.button("↻ Reset Form", use_container_width=True):
                st.session_state.form_values = defaults.copy()
                st.session_state.result = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # Make prediction
    if predict_clicked and not errors:
        with st.spinner("🔄 Analyzing..."):
            frame = build_prediction_frame(vals, feature_order)
            pred, confidence, prob_pos = make_prediction(model, frame)
            st.session_state.result = {
                "pred": pred,
                "confidence": confidence,
                "prob_pos": prob_pos,
            }

    # Result Display
    if st.session_state.result is not None:
        result = st.session_state.result
        is_high_risk = result["pred"] == 1

        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        risk_class = "high" if is_high_risk else "low"
        risk_text = "⚠️ HIGH RISK" if is_high_risk else "✓ LOW RISK"

        st.markdown(
            f'<div class="risk-badge {risk_class}">{risk_text}</div>',
            unsafe_allow_html=True,
        )

        result_message = (
            "Heart disease is likely based on the provided factors."
            if is_high_risk
            else "Heart disease is unlikely based on the provided factors."
        )

        st.markdown(
            f'<div class="result-text">{result_message}</div>',
            unsafe_allow_html=True,
        )

        if result["confidence"] is not None:
            st.markdown(
                f'<div class="confidence-label">Model Confidence</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(result["confidence"], 0.0), 1.0))
            st.markdown(
                f'**{result["confidence"] * 100:.1f}% confident** in this prediction',
                unsafe_allow_html=True,
            )

        if result["prob_pos"] is not None:
            st.markdown(
                f'<div class="confidence-label">Probability of Heart Disease</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(result["prob_pos"], 0.0), 1.0))
            st.caption(f"Risk Score: {result['prob_pos'] * 100:.1f}%")

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📊 Raw Input Values"):
            input_df = pd.DataFrame([vals])
            st.dataframe(input_df, use_container_width=True)

    # Footer
    st.markdown(
        '<div class="footer">'
        '<p>HeartPulse © 2024 | Powered by Machine Learning</p>'
        '<p style="font-size: 0.75rem; margin-top: 0.5rem;">⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Always consult a healthcare professional.</p>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
