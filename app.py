from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HeartPulse - AI Heart Risk Predictor",
    page_icon="❤️",
    layout="wide",
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

CUSTOM_CSS = """
<style>
    :root {
        --bg: #ffffff;
        --surface: #ffffff;
        --surface-soft: #fff7f7;
        --line: rgba(17, 24, 39, 0.08);
        --line-strong: rgba(230, 57, 70, 0.16);
        --text: #111827;
        --muted: #6b7280;
        --red: #e63946;
        --red-dark: #bf1d2d;
        --green: #1f9d68;
        --green-soft: #e7f7ef;
        --shadow: 0 18px 50px rgba(17, 24, 39, 0.08);
        --shadow-soft: 0 8px 24px rgba(17, 24, 39, 0.06);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 6%, rgba(230, 57, 70, 0.08), transparent 18%),
            radial-gradient(circle at 14% 12%, rgba(230, 57, 70, 0.05), transparent 14%),
            linear-gradient(180deg, #ffffff 0%, #fffdfd 100%);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1220px;
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    .shell {
        display: grid;
        grid-template-columns: 230px minmax(0, 1fr);
        gap: 1.4rem;
        align-items: start;
    }

    .rail, .card, .result-card, .hero-card, .metric-card, .info-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: var(--shadow-soft);
    }

    .rail {
        position: sticky;
        top: 1rem;
        padding: 1.4rem 1.2rem;
        min-height: calc(100vh - 2rem);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .brand {
        color: var(--red);
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
    }

    .brand small {
        display: block;
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 500;
    }

    .nav-list {
        margin-top: 1.2rem;
        display: grid;
        gap: 0.5rem;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.95rem 1rem;
        border-radius: 14px;
        color: var(--text);
        background: transparent;
        border: 1px solid transparent;
        font-weight: 600;
        transition: all 180ms ease;
    }

    .nav-item.active {
        background: linear-gradient(90deg, rgba(230, 57, 70, 0.12), rgba(230, 57, 70, 0.04));
        border-color: var(--line-strong);
        color: var(--red);
    }

    .nav-item:hover {
        transform: translateX(2px);
        border-color: var(--line-strong);
        box-shadow: var(--shadow-soft);
    }

    .rail-card {
        padding: 1.15rem;
        border: 1px solid rgba(230, 57, 70, 0.14);
        border-radius: 18px;
        background: linear-gradient(180deg, #fff, #fff7f8);
    }

    .rail-card h4 {
        color: var(--red);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .rail-card p, .footer-mini {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .footer-mini {
        text-align: center;
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid var(--line);
        font-size: 0.8rem;
    }

    .content {
        min-width: 0;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        padding: 1.6rem 1.7rem;
        margin-bottom: 1rem;
        animation: rise 380ms ease-out;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        right: -30px;
        top: -35px;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(230,57,70,0.14), rgba(230,57,70,0.02) 60%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        color: var(--red);
        font-size: clamp(2.1rem, 4vw, 3.35rem);
        font-weight: 800;
        line-height: 1.02;
        letter-spacing: -0.05em;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 0.55rem;
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 600;
    }

    .hero-copy {
        margin-top: 0.35rem;
        color: var(--muted);
        max-width: 720px;
        line-height: 1.6;
    }

    .metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 1rem 1.05rem;
        min-height: 88px;
        transition: transform 180ms ease, box-shadow 180ms ease;
        animation: rise 420ms ease-out;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 700;
    }

    .metric-value.red { color: var(--red); }
    .metric-value.green { color: var(--green); }

    .main-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.85fr);
        gap: 1rem;
        align-items: start;
    }

    .card, .result-card {
        padding: 1.25rem;
        animation: rise 420ms ease-out;
    }

    .card-title, .result-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .section-note {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }

    .input-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.9rem 1rem;
    }

    .field {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.85rem 0.85rem 0.2rem;
        background: #fff;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }

    .field:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-soft);
        border-color: var(--line-strong);
    }

    .field .stSelectbox, .field .stNumberInput, .field .stSlider {
        margin-bottom: 0;
    }

    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] > div {
        background: #fff !important;
        border-radius: 12px !important;
        border: 1px solid var(--line) !important;
        color: var(--text) !important;
    }

    [data-testid="stNumberInput"] label p,
    [data-testid="stSelectbox"] label p,
    [data-testid="stSlider"] label p {
        color: var(--text) !important;
        font-weight: 650 !important;
        font-size: 0.9rem !important;
    }

    .stButton > button {
        border: 0 !important;
        border-radius: 14px !important;
        min-height: 3rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) scale(1.01) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--red) 0%, var(--red-dark) 100%) !important;
        color: white !important;
        box-shadow: 0 12px 24px rgba(230, 57, 70, 0.24) !important;
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 14px 28px rgba(230, 57, 70, 0.32) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #fff !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
    }

    .actions {
        display: grid;
        grid-template-columns: 1.4fr 1fr;
        gap: 0.8rem;
        margin-top: 1rem;
    }

    .result-card {
        position: sticky;
        top: 1rem;
    }

    .result-ring {
        width: 182px;
        height: 182px;
        border-radius: 50%;
        margin: 0.35rem auto 0.9rem;
        display: grid;
        place-items: center;
        border: 10px solid rgba(230, 57, 70, 0.95);
        border-top-color: rgba(230, 57, 70, 0.2);
        border-right-color: rgba(230, 57, 70, 0.55);
        animation: spinIn 700ms ease-out;
    }

    .result-ring.low {
        border-color: rgba(31, 157, 104, 0.95);
        border-top-color: rgba(31, 157, 104, 0.2);
        border-right-color: rgba(31, 157, 104, 0.55);
    }

    .result-inner {
        text-align: center;
        line-height: 1.15;
    }

    .result-status {
        color: var(--red);
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .result-status.low {
        color: var(--green);
    }

    .result-percent {
        color: var(--red);
        font-size: 1.9rem;
        font-weight: 900;
        letter-spacing: -0.05em;
    }

    .result-percent.low { color: var(--green); }

    .result-caption {
        color: var(--muted);
        font-size: 0.85rem;
        font-weight: 600;
    }

    .alert-box {
        margin-top: 0.85rem;
        padding: 0.95rem 1rem;
        border-radius: 14px;
        background: #fff6f6;
        border: 1px solid rgba(230, 57, 70, 0.18);
        color: var(--text);
    }

    .alert-box.low {
        background: var(--green-soft);
        border-color: rgba(31, 157, 104, 0.16);
    }

    .result-list {
        margin-top: 1rem;
        display: grid;
        gap: 0.55rem;
    }

    .result-item {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: var(--text);
        font-size: 0.92rem;
        padding: 0.55rem 0.2rem;
        border-bottom: 1px solid rgba(17, 24, 39, 0.06);
    }

    .footer {
        margin-top: 1.2rem;
        padding-top: 1rem;
        color: var(--muted);
        font-size: 0.8rem;
        text-align: center;
    }

    .helper-row {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.6rem;
    }

    .helper-chip {
        padding: 0.36rem 0.62rem;
        border-radius: 999px;
        background: #fff7f7;
        color: var(--red);
        border: 1px solid rgba(230, 57, 70, 0.12);
        font-size: 0.78rem;
        font-weight: 700;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--red) 0%, var(--red-dark) 100%) !important;
        border-radius: 999px !important;
        height: 0.7rem !important;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes spinIn {
        from { opacity: 0; transform: scale(0.94) rotate(-8deg); }
        to { opacity: 1; transform: scale(1) rotate(0deg); }
    }

    @media (max-width: 1080px) {
        .shell, .main-grid {
            grid-template-columns: 1fr;
        }

        .rail {
            position: relative;
            min-height: auto;
        }

        .result-card {
            position: relative;
            top: 0;
        }
    }

    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }

        .metrics, .input-grid, .actions {
            grid-template-columns: 1fr;
        }

        .hero-card, .card, .result-card, .rail {
            padding: 1rem;
        }

        .result-ring {
            width: 160px;
            height: 160px;
        }
    }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path) -> Tuple:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_order = payload.get("features") or FEATURES
    else:
        model = payload
        feature_order = list(getattr(model, "feature_names_in_", FEATURES))
    return model, feature_order


@st.cache_data(show_spinner=False)
def load_profile(path: Path) -> Dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    profile = {}
    for column in df.columns:
        if column == "target":
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            profile[column] = {
                "median": float(df[column].median()),
                "min": float(df[column].min()),
                "max": float(df[column].max()),
            }
    return profile


def default_values(profile: Dict) -> Dict:
    def med(name: str, fallback: float) -> float:
        return profile.get(name, {}).get("median", fallback)

    return {
        "age": int(round(med("age", 45))),
        "sex": int(round(med("sex", 1))),
        "cp": int(round(med("cp", 1))),
        "restbp": int(round(med("trestbps", 120))),
        "chol": int(round(med("chol", 200))),
        "fbs": int(round(med("fbs", 0))),
        "restecg": int(round(med("restecg", 0))),
        "thalach": int(round(med("thalach", 150))),
        "exang": int(round(med("exang", 0))),
        "oldpeak": float(round(med("oldpeak", 1.0), 1)),
        "slope": int(round(med("slope", 1))),
        "ca": int(round(med("ca", 0))),
        "thal": int(round(med("thal", 2))),
    }


def validate_inputs(values: Dict) -> List[str]:
    issues = []
    if not 18 <= values["age"] <= 100:
        issues.append("Age should be between 18 and 100")
    if not 80 <= values["restbp"] <= 220:
        issues.append("Resting blood pressure should be between 80 and 220")
    if not 100 <= values["chol"] <= 600:
        issues.append("Cholesterol should be between 100 and 600")
    if not 60 <= values["thalach"] <= 220:
        issues.append("Max heart rate should be between 60 and 220")
    if not 0.0 <= values["oldpeak"] <= 6.0:
        issues.append("Oldpeak should be between 0.0 and 6.0")
    return issues


def build_frame(values: Dict, feature_order: List[str]) -> pd.DataFrame:
    mapping = {
        "age": values["age"],
        "sex": values["sex"],
        "cp": values["cp"],
        "trestbps": values["restbp"],
        "chol": values["chol"],
        "fbs": values["fbs"],
        "restecg": values["restecg"],
        "thalach": values["thalach"],
        "exang": values["exang"],
        "oldpeak": values["oldpeak"],
        "slope": values["slope"],
        "ca": values["ca"],
        "thal": values["thal"],
    }
    row = {name: mapping.get(name, 0) for name in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict(model, frame: pd.DataFrame) -> Tuple[int, Optional[float], Optional[float]]:
    pred = int(model.predict(frame)[0])
    confidence = None
    prob_pos = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        prob_pos = float(proba[1])
        confidence = float(proba[pred])
    return pred, confidence, prob_pos


def render_rail() -> None:
    st.markdown(
        """
        <div class="rail">
            <div>
                <div class="brand">HeartPulse ❤️<small>AI-Powered Heart Disease Risk Prediction</small></div>
                <div class="nav-list">
                    <div class="nav-item active">⌂ Home</div>
                    <div class="nav-item">▥ About Model</div>
                    <div class="nav-item">◎ How It Works</div>
                </div>
            </div>
            <div>
                <div class="rail-card">
                    <h4>Your heart, our priority.</h4>
                    <p>Take control of your health with AI-powered insights.</p>
                </div>
                <div class="footer-mini">© 2026 HeartPulse ❤️<br/>All rights reserved.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">HeartPulse ❤️</h1>
            <div class="hero-subtitle">AI-Powered Heart Disease Risk Prediction</div>
            <div class="hero-copy">Get insights about your heart health based on clinical factors using advanced machine learning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(is_high_risk: Optional[bool]) -> None:
    if is_high_risk is None:
        metric_label = "READY"
        metric_class = "red"
        metric_text = "Model loaded and ready"
    elif is_high_risk:
        metric_label = "HIGH RISK"
        metric_class = "red"
        metric_text = "Based on your inputs"
    else:
        metric_label = "LOW RISK"
        metric_class = "green"
        metric_text = "Based on your inputs"

    st.markdown(
        f"""
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Current status</div>
                <div class="metric-value {metric_class}">{metric_label}</div>
                <div class="metric-label">{metric_text}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Stay proactive</div>
                <div class="metric-value">Regular checkups save lives</div>
                <div class="metric-label">Small steps matter</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Private & secure</div>
                <div class="metric-value green">Your data is safe with us</div>
                <div class="metric-label">No data leaves your session</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inputs(values: Dict) -> Dict:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Patient Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-note">Enter the clinical factors below. The layout is optimized for fast scanning and clean form flow.</div>', unsafe_allow_html=True)

    out = values.copy()
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["age"] = st.slider("Age (years)", 18, 100, int(out["age"]), help="Patient age")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["cp"] = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=int(out["cp"]), format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["chol"] = st.slider("Serum Cholesterol (mg/dl)", 100, 600, int(out["chol"]), step=1, help="Higher values can indicate risk")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["restecg"] = st.selectbox("Resting ECG Results", [0, 1, 2], index=int(out["restecg"]), format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["exang"] = st.selectbox("Exercise Induced Angina", [0, 1], index=int(out["exang"]), format_func=lambda x: ["No", "Yes"][x])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["slope"] = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], index=int(out["slope"]), format_func=lambda x: ["Up Sloping", "Flat", "Down Sloping"][x])
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["sex"] = st.selectbox("Sex", [0, 1], index=int(out["sex"]), format_func=lambda x: ["Female", "Male"][x])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["restbp"] = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, int(out["restbp"]), step=1, help="Blood pressure at rest")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["thalach"] = st.slider("Max Heart Rate Achieved", 60, 220, int(out["thalach"]), step=1, help="Peak heart rate during exercise")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["oldpeak"] = st.slider("ST Depression (oldpeak)", 0.0, 6.0, float(out["oldpeak"]), step=0.1, help="Exercise-induced ST depression")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=int(out["fbs"]), format_func=lambda x: ["No", "Yes"][x])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["ca"] = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], index=min(int(out["ca"]), 3))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="field">', unsafe_allow_html=True)
        out["thal"] = st.selectbox("Thalassemia", [0, 1, 2, 3], index=int(out["thal"]), format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="helper-row"><div class="helper-chip">White UI</div><div class="helper-chip">Medical red</div><div class="helper-chip">Fast prediction</div><div class="helper-chip">Mobile-friendly</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return out


def render_result(pred: int, confidence: Optional[float], prob_pos: Optional[float]) -> None:
    high = pred == 1
    ring_class = "" if high else "low"
    status = "High Risk" if high else "Low Risk"
    percent = (prob_pos if prob_pos is not None else confidence if confidence is not None else 0.0) * 100
    message = (
        "You are at high risk of heart disease. Please consult a cardiologist and maintain a healthy lifestyle."
        if high
        else "Low risk detected. Keep following healthy routines and regular checkups."
    )

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="result-title">📌 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-ring {ring_class}"><div class="result-inner"><div class="result-status {ring_class}">{status}</div><div class="result-percent {ring_class}">{percent:.0f}%</div><div class="result-caption">Probability</div></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-box {"low" if not high else ""}">{message}</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-list">', unsafe_allow_html=True)
    for text in ["Eat a heart-healthy diet", "Exercise regularly", "Manage stress", "Get regular checkups", "Avoid smoking & alcohol"]:
        st.markdown(f'<div class="result-item">• {text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if confidence is not None:
        st.markdown('<div class="section-note" style="margin-top:1rem;">Model confidence</div>', unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, confidence)))

    if prob_pos is not None:
        st.markdown('<div class="section-note" style="margin-top:0.9rem;">Heart disease probability</div>', unsafe_allow_html=True)
        st.progress(max(0.0, min(1.0, prob_pos)))

    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    inject_css()

    if not MODEL_PATH.exists():
        st.error("Model file heart_disease_pipeline.pkl not found.")
        st.stop()

    model, feature_order = load_pipeline(MODEL_PATH)
    profile = load_profile(DATA_PATH)
    defaults = default_values(profile)

    if "form_values" not in st.session_state:
        st.session_state.form_values = defaults.copy()
    if "result" not in st.session_state:
        st.session_state.result = None

    shell_left, shell_right = st.columns([0.22, 0.78], gap="large")

    with shell_left:
        render_rail()

    with shell_right:
        st.markdown('<div class="content">', unsafe_allow_html=True)
        render_header()
        render_metrics(st.session_state.result["pred"] == 1 if st.session_state.result else None)

        current = render_inputs(st.session_state.form_values.copy())
        errors = validate_inputs(current)
        st.session_state.form_values = current

        if errors:
            for error in errors:
                st.warning(error)

        st.markdown('<div class="actions">', unsafe_allow_html=True)
        predict_col, reset_col = st.columns(2)
        with predict_col:
            predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True, disabled=bool(errors))
        with reset_col:
            if st.button("Reset All", use_container_width=True):
                st.session_state.form_values = defaults.copy()
                st.session_state.result = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_clicked and not errors:
            with st.spinner("Analyzing your inputs..."):
                frame = build_frame(current, feature_order)
                pred, confidence, prob_pos = predict(model, frame)
                st.session_state.result = {
                    "pred": pred,
                    "confidence": confidence,
                    "prob_pos": prob_pos,
                }

        if st.session_state.result is not None:
            render_result(
                st.session_state.result["pred"],
                st.session_state.result["confidence"],
                st.session_state.result["prob_pos"],
            )

        st.markdown('<div class="footer">© 2026 HeartPulse • Built for fast, clear heart-risk screening</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
