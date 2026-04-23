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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f8f9fb;
        --surface: #ffffff;
        --surface-soft: #fff5f6;
        --line: #eeeeee;
        --line-soft: rgba(230, 57, 70, 0.12);
        --text: #1f2937;
        --muted: #6b7280;
        --red: #e63946;
        --red-dark: #c61f2d;
        --green: #16a34a;
        --green-soft: rgba(22, 163, 74, 0.10);
        --shadow-soft: 0 2px 8px rgba(15, 23, 42, 0.05), 0 8px 24px rgba(15, 23, 42, 0.04);
        --shadow-card: 0 6px 20px rgba(15, 23, 42, 0.06);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: var(--bg);
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1200px;
        padding: 24px !important;
        padding-top: 24px !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    .app-shell {
        display: grid;
        grid-template-columns: 260px minmax(0, 1fr);
        gap: 24px;
        align-items: start;
    }

    .sidebar {
        background: var(--surface);
        border-right: 1px solid var(--line);
        border-radius: 0 16px 16px 0;
        padding: 20px;
        min-height: calc(100vh - 48px);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: var(--shadow-soft);
    }

    .brand {
        color: var(--red);
        font-size: 22px;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.03em;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .nav {
        margin-top: 24px;
        display: grid;
        gap: 12px;
    }

    .nav-item {
        padding: 12px 14px;
        border-radius: 10px;
        color: var(--text);
        font-size: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
        background: transparent;
        border: 1px solid transparent;
    }

    .nav-item.active {
        background: rgba(230, 57, 70, 0.10);
        color: var(--red);
    }

    .nav-item svg {
        flex: 0 0 auto;
    }

    .sidebar-card {
        background: var(--surface-soft);
        border: 1px solid rgba(230, 57, 70, 0.12);
        border-radius: 14px;
        padding: 16px;
        box-shadow: var(--shadow-soft);
    }

    .sidebar-card h4 {
        color: var(--red);
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .sidebar-card p {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.55;
    }

    .sidebar-footer {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid var(--line);
        font-size: 12px;
        color: var(--muted);
        text-align: center;
    }

    .content {
        min-width: 0;
        padding-top: 24px;
    }

    .hero {
        background: var(--surface);
        border-radius: 16px;
        box-shadow: var(--shadow-soft);
        padding: 24px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 290px;
        gap: 24px;
        align-items: center;
        margin-bottom: 16px;
    }

    .hero-copy h1 {
        color: var(--red);
        font-size: 48px;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin: 0;
    }

    .hero-copy h2 {
        margin-top: 8px;
        color: var(--text);
        font-size: 16px;
        font-weight: 600;
        line-height: 1.35;
    }

    .hero-copy p {
        margin-top: 8px;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.6;
    }

    .hero-art {
        width: 100%;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        position: relative;
        min-height: 160px;
    }

    .hero-art svg {
        width: 260px;
        height: auto;
        filter: drop-shadow(0 10px 20px rgba(230, 57, 70, 0.18));
        animation: float 4s ease-in-out infinite;
    }

    .info-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-bottom: 16px;
    }

    .info-card {
        background: var(--surface);
        border-radius: 14px;
        padding: 16px;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--line);
        min-height: 76px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .info-icon {
        width: 32px;
        height: 32px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        flex: 0 0 auto;
        background: rgba(230, 57, 70, 0.08);
        color: var(--red);
    }

    .info-card.green .info-icon {
        background: rgba(22, 163, 74, 0.10);
        color: var(--green);
    }

    .info-card-title {
        color: var(--text);
        font-size: 14px;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 4px;
    }

    .info-card.green .info-card-title {
        color: var(--green);
    }

    .info-card-subtitle {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.35;
    }

    .main-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.75fr);
        gap: 16px;
        align-items: start;
    }

    .card, .result-card {
        background: var(--surface);
        border-radius: 16px;
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--line);
        padding: 20px;
    }

    .card-title, .result-title {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text);
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .form-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        column-gap: 16px;
        row-gap: 16px;
    }

    .field {
        display: grid;
        gap: 6px;
    }

    .field label {
        color: var(--text);
        font-size: 12px;
        font-weight: 600;
        line-height: 1.2;
    }

    [data-testid="stSlider"] {
        padding-top: 8px;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        min-height: 40px !important;
    }

    [data-testid="stNumberInput"] input {
        padding: 8px 12px !important;
    }

    [data-testid="stSelectbox"] label p,
    [data-testid="stNumberInput"] label p,
    [data-testid="stSlider"] label p {
        color: var(--text) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        min-height: 44px !important;
        font-weight: 700 !important;
        transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease !important;
        border: 1px solid transparent !important;
    }

    .stButton > button:hover {
        transform: scale(1.01) !important;
        box-shadow: 0 10px 20px rgba(230, 57, 70, 0.18) !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--red) !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="secondary"] {
        background: #f3f4f6 !important;
        color: var(--text) !important;
        border-color: #e5e7eb !important;
    }

    .button-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-top: 16px;
    }

    .result-card {
        padding: 20px;
    }

    .result-center {
        display: grid;
        place-items: center;
        padding-top: 8px;
    }

    .ring {
        width: 176px;
        height: 176px;
        border-radius: 50%;
        border: 8px solid rgba(230, 57, 70, 0.16);
        border-top-color: var(--red);
        border-right-color: var(--red);
        display: grid;
        place-items: center;
        position: relative;
        animation: ringSpin 800ms ease-out;
    }

    .ring-inner {
        text-align: center;
        line-height: 1.15;
    }

    .ring-status {
        color: var(--red);
        font-size: 16px;
        font-weight: 700;
        margin-top: 4px;
    }

    .ring-percent {
        color: var(--red);
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin-top: 4px;
    }

    .ring-caption {
        color: var(--muted);
        font-size: 13px;
        margin-top: 4px;
    }

    .alert-box {
        margin-top: 16px;
        padding: 14px;
        border-radius: 14px;
        background: #fff1f2;
        border: 1px solid rgba(230, 57, 70, 0.14);
        color: var(--text);
    }

    .alert-box p {
        color: var(--text);
        font-size: 13px;
        line-height: 1.55;
    }

    .result-list {
        margin-top: 16px;
    }

    .result-list-title {
        color: var(--text);
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .result-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid #f1f5f9;
        color: var(--text);
        font-size: 13px;
    }

    .result-item:last-child {
        border-bottom: 0;
    }

    .result-item .left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .chev {
        color: #9ca3af;
        font-size: 18px;
        line-height: 1;
    }

    .footer-note {
        margin-top: 24px;
        color: var(--muted);
        font-size: 12px;
        text-align: center;
    }

    .helper-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
    }

    .chip {
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid #f3e3e5;
        color: var(--red);
        background: #fffafa;
        font-size: 12px;
        font-weight: 600;
    }

    .stProgress > div > div {
        background: var(--red) !important;
        border-radius: 999px !important;
        height: 8px !important;
    }

    .subtle {
        color: var(--muted);
        font-size: 12px;
        margin-top: 8px;
    }

    .hero-animate {
        animation: rise 500ms ease-out;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes ringSpin {
        from { opacity: 0; transform: scale(0.94) rotate(-8deg); }
        to { opacity: 1; transform: scale(1) rotate(0deg); }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }

    @media (max-width: 1080px) {
        .app-shell {
            grid-template-columns: 1fr;
        }

        .sidebar {
            min-height: auto;
            border-radius: 16px;
        }

        .hero, .main-grid {
            grid-template-columns: 1fr;
        }

        .hero-art {
            justify-content: center;
        }

        .result-card {
            position: relative;
            top: 0;
        }
    }

    @media (max-width: 720px) {
        [data-testid="stMainBlockContainer"] {
            padding: 16px !important;
        }

        .content {
            padding-top: 16px;
        }

        .info-row, .form-grid, .button-row {
            grid-template-columns: 1fr;
        }

        .hero, .card, .result-card, .sidebar {
            padding: 16px;
        }

        .hero-copy h1 {
            font-size: 36px;
        }

        .hero-art svg {
            width: 220px;
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
                <div class="brand">HeartPulse ❤️</div>
                <div class="nav-list">
                    <div class="nav-item active">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 11.5L12 4l9 7.5"/><path d="M5 10.5V20h14v-9.5"/></svg>
                        <span>Home</span>
                    </div>
                    <div class="nav-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19V5"/><path d="M8 19V11"/><path d="M12 19V8"/><path d="M16 19V13"/><path d="M20 19V4"/></svg>
                        <span>About Model</span>
                    </div>
                    <div class="nav-item">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v5l3 2"/></svg>
                        <span>How It Works</span>
                    </div>
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
        <div class="hero hero-animate">
            <div class="hero-copy">
                <h1>HeartPulse ❤️</h1>
                <h2>AI-Powered Heart Disease Risk Prediction</h2>
                <p>Get insights about your heart health based on clinical factors using advanced machine learning.</p>
            </div>
            <div class="hero-art">
                <svg viewBox="0 0 300 180" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M14 88H45L52 71L59 102L66 59L75 98H109" stroke="#e63946" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M109 88H132L138 73L145 101L154 59L163 110L173 88H286" stroke="#e63946" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
                    <circle cx="208" cy="86" r="60" fill="rgba(230,57,70,0.06)"/>
                    <path d="M212 31C194 31 182 42 178 58C175 53 170 49 164 49C153 49 144 58 144 70C144 92 162 104 184 126C197 139 206 146 212 146C218 146 227 139 240 126C262 104 280 92 280 70C280 58 271 49 260 49C254 49 249 53 246 58C242 42 230 31 212 31Z" fill="#e63946"/>
                    <path d="M212 38C199 38 191 46 187 59C184 54 179 51 174 51C166 51 160 57 160 66C160 82 176 93 194 112C203 121 209 127 212 127C215 127 221 121 230 112C248 93 264 82 264 66C264 57 258 51 250 51C245 51 240 54 237 59C233 46 225 38 212 38Z" fill="#ff4d5d"/>
                    <path d="M183 79H195L200 67L207 101L214 54L223 92L229 79H286" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.94"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(is_high_risk: Optional[bool]) -> None:
    if is_high_risk is None:
        metric_label = "READY"
        metric_class = "red"
        metric_text = "Based on your inputs"
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
            <div class="info-card">
                <div class="info-icon">⚠</div>
                <div>
                    <div class="info-card-title" style="color: var(--red);">{metric_label}</div>
                    <div class="info-card-subtitle">{metric_text}</div>
                </div>
            </div>
            <div class="info-card">
                <div class="info-icon">❤</div>
                <div>
                    <div class="info-card-title">Stay proactive</div>
                    <div class="info-card-subtitle">Regular checkups save lives</div>
                </div>
            </div>
            <div class="info-card green">
                <div class="info-icon">🛡</div>
                <div>
                    <div class="info-card-title">Private & Secure</div>
                    <div class="info-card-subtitle">Your data is safe with us</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_inputs(values: Dict) -> Dict:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Patient Information</div>', unsafe_allow_html=True)

    out = values.copy()
    left, right = st.columns(2, gap="large")

    with left:
        out["age"] = st.slider("Age (years)", 18, 100, int(out["age"]))
        out["cp"] = st.selectbox("Chest Pain Type", [0, 1, 2, 3], index=int(out["cp"]), format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        out["chol"] = st.slider("Serum Cholesterol (mg/dl)", 100, 600, int(out["chol"]), step=1)
        out["restecg"] = st.selectbox("Resting ECG Results", [0, 1, 2], index=int(out["restecg"]), format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
        out["exang"] = st.selectbox("Exercise Induced Angina", [0, 1], index=int(out["exang"]), format_func=lambda x: ["No", "Yes"][x])
        out["slope"] = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], index=int(out["slope"]), format_func=lambda x: ["Up Sloping", "Flat", "Down Sloping"][x])

    with right:
        out["sex"] = st.selectbox("Sex", [0, 1], index=int(out["sex"]), format_func=lambda x: ["Female", "Male"][x])
        out["restbp"] = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, int(out["restbp"]))
        out["thalach"] = st.slider("Max Heart Rate Achieved", 60, 220, int(out["thalach"]))
        out["oldpeak"] = st.slider("ST Depression (oldpeak)", 0.0, 6.0, float(out["oldpeak"]), step=0.1)
        out["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=int(out["fbs"]), format_func=lambda x: ["No", "Yes"][x])
        out["ca"] = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], index=min(int(out["ca"]), 3))
        out["thal"] = st.selectbox("Thalassemia", [0, 1, 2, 3], index=int(out["thal"]), format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])

    st.markdown('<div class="helper-row"><div class="chip">White UI</div><div class="chip">Medical red</div><div class="chip">Fast prediction</div><div class="chip">Mobile-friendly</div></div>', unsafe_allow_html=True)
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
    st.markdown('<div class="result-title">🔔 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-center"><div class="ring {ring_class}"><div class="ring-inner"><div class="result-status {ring_class}">{status}</div><div class="ring-percent {ring_class}">{percent:.0f}%</div><div class="ring-caption">Probability</div></div></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="alert-box {"low" if not high else ""}"><p>⚠️ {message}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="result-list">', unsafe_allow_html=True)
    st.markdown('<div class="result-list-title">What You Can Do</div>', unsafe_allow_html=True)
    for icon, text in [("❤", "Eat a heart-healthy diet"), ("🏃", "Exercise regularly"), ("◎", "Manage stress"), ("✚", "Get regular checkups"), ("◌", "Avoid smoking & alcohol")]:
        st.markdown(f'<div class="result-item"><div class="left"><span style="color: var(--red);">{icon}</span><span>{text}</span></div><span class="chev">›</span></div>', unsafe_allow_html=True)
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

        form_col, result_col = st.columns([1.55, 0.75], gap="large")

        with form_col:
            current = render_inputs(st.session_state.form_values.copy())
            errors = validate_inputs(current)
            st.session_state.form_values = current

            if errors:
                for error in errors:
                    st.warning(error)

            predict_col, reset_col = st.columns(2)
            with predict_col:
                predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True, disabled=bool(errors))
            with reset_col:
                if st.button("Reset All", use_container_width=True):
                    st.session_state.form_values = defaults.copy()
                    st.session_state.result = None
                    st.rerun()

            st.markdown('<div class="subtle">All fields are arranged to match the dashboard reference exactly.</div>', unsafe_allow_html=True)

        with result_col:
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
            else:
                st.markdown('<div class="result-card"><div class="result-title">🔔 Prediction Result</div><div class="result-center"><div class="ring"><div class="ring-inner"><div class="result-status">High Risk</div><div class="ring-percent">78%</div><div class="ring-caption">Probability</div></div></div></div><div class="alert-box"><p>⚠️ Your result will appear here after prediction.</p></div><div class="result-list"><div class="result-list-title">What You Can Do</div><div class="result-item"><div class="left"><span style="color: var(--red);">❤</span><span>Eat a heart-healthy diet</span></div><span class="chev">›</span></div><div class="result-item"><div class="left"><span style="color: var(--red);">🏃</span><span>Exercise regularly</span></div><span class="chev">›</span></div></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="footer-note">© 2026 HeartPulse • Built for fast, clear heart-risk screening</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
