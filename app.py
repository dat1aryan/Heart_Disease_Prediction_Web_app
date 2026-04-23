from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HeartPulse - AI Heart Risk Predictor",
    page_icon="HP",
    layout="wide",
    initial_sidebar_state="expanded",
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

ICON_PATHS = {
    "heart": '<path d="M12 21s-6.7-4.35-9.3-8.08C.38 9.53 2.26 5.5 6 5.5c2.12 0 3.46 1.12 4 2.12.54-1 1.88-2.12 4-2.12 3.74 0 5.62 4.03 3.3 7.42C18.7 16.65 12 21 12 21z"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "shield": '<path d="M12 3 4 7v6c0 5 3.5 8.74 8 10 4.5-1.26 8-5 8-10V7l-8-4z"/>',
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "user": '<path d="M20 21a8 8 0 0 0-16 0"/><circle cx="12" cy="7" r="4"/>',
    "chevron-right": '<polyline points="9 18 15 12 9 6"/>',
    "droplet": '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>',
    "trending-down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 12 12 17 22 12"/><polyline points="2 17 12 22 22 17"/>',
}



def brand_heart_svg(size: int = 24) -> str:
    return f'''
    <svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="url(#heartGrad)" aria-hidden="true" style="filter: drop-shadow(0px 2px 8px rgba(230,57,70,0.35));">
        <defs>
            <linearGradient id="heartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#ff7b88" />
                <stop offset="100%" stop-color="#e63946" />
            </linearGradient>
        </defs>
        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
    </svg>
    '''

def icon_svg(name: str, size: int = 18, color: str = "currentColor", css_class: str = "") -> str:
    path = ICON_PATHS.get(name, "")
    return (
        f'<svg class="{css_class}" xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #f8f9fb;
    --surface: #ffffff;
    --line: #eeeeee;
    --text: #1f2937;
    --muted: #6b7280;
    --red: #e63946;
    --red-soft: rgba(230, 57, 70, 0.10);
    --green: #16a34a;
    --green-soft: rgba(22, 163, 74, 0.10);
    --shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: "SF Pro Display", "Inter", "Helvetica Neue", sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1100px;
    padding: 24px !important;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #eeeeee !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding: 20px 16px !important;
}

.sidebar-brand {
    color: var(--red);
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 20px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.02em;
}

.sidebar-card {
    margin-top: 24px;
    border-radius: 14px;
    border: 1px solid rgba(230, 57, 70, 0.14);
    background: #fff5f6;
    padding: 16px;
}

.sidebar-card-title {
    color: var(--red);
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}

.sidebar-card-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.5;
}

.sidebar-footer {
    margin-top: 16px;
    color: var(--muted);
    font-size: 12px;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 8px;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 12px 14px;
    margin: 0 !important;
}

[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(230, 57, 70, 0.10);
    border-color: rgba(230, 57, 70, 0.18);
}


[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(1) p::before {
    content: '';
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 12px;
    vertical-align: middle;
    background: currentColor;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") no-repeat center / contain;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(2) p::before {
    content: '';
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 12px;
    vertical-align: middle;
    background: currentColor;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") no-repeat center / contain;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(3) p::before {
    content: '';
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 12px;
    vertical-align: middle;
    background: currentColor;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='12' y1='16' x2='12' y2='12'/%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='10'/%3E%3Cline x1='12' y1='16' x2='12' y2='12'/%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'/%3E%3C/svg%3E") no-repeat center / contain;
}

[data-testid="stSidebar"] .stRadio label p {
    color: var(--text) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

.page-section {
    margin-bottom: 24px;
}

.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 20px;
    box-shadow: var(--shadow);
    transition: transform 180ms ease, box-shadow 180ms ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 220px;
    gap: 24px;
    align-items: center;
}

.hero-title {
    color: var(--red);
    font-size: 48px;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    margin: 0;
}

.hero-brand-row {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.hero-brand-icon {
    color: var(--red);
    filter: drop-shadow(0 4px 12px rgba(230, 57, 70, 0.24));
}

.hero-subtitle {
    color: var(--text);
    font-size: 16px;
    font-weight: 600;
    margin-top: 8px;
}

.hero-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
    margin-top: 8px;
}

.hero-heart {
    display: flex;
    justify-content: center;
}

.hero-heart svg {
    width: 200px;
    height: auto;
    filter: drop-shadow(0 10px 18px rgba(230, 57, 70, 0.20));
}

.info-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
}

.info-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px;
    box-shadow: var(--shadow);
}

.info-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.info-title.red {
    color: var(--red);
}

.info-title.green {
    color: var(--green);
}

.info-copy {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.4;
}

.main-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.75fr);
    gap: 16px;
}


.input-label {
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.form-title, .result-title {
    color: var(--text);
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 16px;
}

[data-testid="stNumberInput"],
[data-testid="stSelectbox"],
[data-testid="stSlider"] {
    margin-bottom: 12px;
}

[data-testid="stNumberInput"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stSlider"] label p {
    color: var(--text) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

[data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid var(--line) !important;
    color: var(--text) !important;
    background: #ffffff !important;
}

.stButton > button {
    border-radius: 10px !important;
    min-height: 44px !important;
    font-weight: 700 !important;
    border: 1px solid transparent !important;
    transition: transform 140ms ease, box-shadow 140ms ease !important;
}

.stButton > button:hover {
    transform: scale(1.01) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e63946 0%, #ff5663 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(230, 57, 70, 0.24) !important;
}

.stButton > button[kind="secondary"] {
    background: #f3f4f6 !important;
    color: var(--text) !important;
    border-color: #e5e7eb !important;
}

.result-ring-wrap {
    display: grid;
    place-items: center;
    margin-top: 8px;
    margin-bottom: 16px;
}

.result-ring {
    width: 172px;
    height: 172px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    padding: 8px;
}

.result-ring-inner {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: #fff;
    display: grid;
    place-items: center;
}

.result-status {
    color: var(--red);
    font-size: 20px;
    font-weight: 700;
    text-align: center;
}

.result-percent {
    color: var(--red);
    font-size: 40px;
    font-weight: 800;
    line-height: 1;
    text-align: center;
}

.result-prob {
    color: var(--muted);
    font-size: 13px;
    text-align: center;
}

.alert-box {
    border-radius: 14px;
    border: 1px solid rgba(230, 57, 70, 0.14);
    background: #fff1f2;
    padding: 14px;
    color: var(--text);
    font-size: 13px;
    line-height: 1.55;
    margin-bottom: 16px;
}

.tip-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 12px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.tip-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid #f1f5f9;
    color: var(--text);
    font-size: 13px;
}

.tip-left {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.icon-danger {
    color: var(--red);
}

.icon-ok {
    color: var(--green);
}

.icon-muted {
    color: #9ca3af;
}

.tip-item:last-child {
    border-bottom: none;
}

.step-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}

.step-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px;
    box-shadow: var(--shadow);
}

.step-number {
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: rgba(230, 57, 70, 0.12);
    color: var(--red);
    font-size: 13px;
    font-weight: 700;
    display: inline-grid;
    place-items: center;
    margin-bottom: 8px;
}

.step-title {
    color: var(--text);
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
}

.step-copy {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
}

.about-list {
    margin: 0;
    padding-left: 18px;
    color: var(--text);
}

.about-list li {
    margin-bottom: 8px;
    color: var(--text);
    font-size: 14px;
}

.muted {
    color: var(--muted);
    font-size: 14px;
}

@media (max-width: 1024px) {
    .hero-grid,
    .main-grid,
    .info-row {
        grid-template-columns: 1fr;
    }

    .hero-heart {
        justify-content: flex-start;
    }
}

@media (max-width: 720px) {
    [data-testid="stMainBlockContainer"] {
        padding: 16px !important;
    }

    .step-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_order = payload.get("features") or FEATURES
        metadata = payload.get("metrics") or {}
    else:
        model = payload
        feature_order = list(getattr(model, "feature_names_in_", FEATURES))
        metadata = {}
    return model, feature_order, metadata


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
            }
    return profile


@st.cache_data(show_spinner=False)
def get_dataset_meta(path: Path) -> Dict:
    if not path.exists():
        return {"rows": "N/A", "cols": "N/A"}
    df = pd.read_csv(path)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }


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


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">HeartPulse ❤️</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Home", "About Model", "How It Works"],
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <div style="text-align: center; margin-bottom: 8px;">
                    <span style="font-size: 32px; filter: drop-shadow(0 2px 4px rgba(230,57,70,0.3));">❤️</span>
                </div>
                <div class="sidebar-card-title" style="text-align: center;">Your heart, our priority.</div>
                <div class="sidebar-card-copy" style="text-align: center;">Take control of your health with AI-powered insights.</div>
            </div>
            <div class="sidebar-footer">© 2026 HeartPulse ❤️<br/>All rights reserved.</div>
            """,
            unsafe_allow_html=True,
        )
    return page


def render_hero() -> None:
    st.markdown(
        """
        <div class="card page-section">
            <div class="hero-grid">
                <div>
                    <div class="hero-brand-row">
                        <h1 class="hero-title">HeartPulse ❤️</h1>
                    </div>
                    <div class="hero-subtitle">AI-Powered Heart Disease Risk Prediction</div>
                    <div class="hero-copy">Get insights about your heart health based on clinical factors using advanced machine learning.</div>
                </div>
                <div class="hero-heart">
<img src="data:image/png;base64,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ZotqkW39qviOnCOgVyksHtUgh9qthKXZQBWEVPEdWAlKFoAhCU4JU22l20CIgtOC1IBS4FMCPbShak2+1LtpDI9tKFp+KXFAhmKXFOopjG4pQKdRigQlLS4ooC4mKKWigAoopRQAYopaKBBRRRQAUtFFAwooopCCiiimAh6UlKaSkAUUUlMAooooAKKKKQBSUtFACUUtJTuMKKKKYCUUtFIBKKWg0WAbgUYp2KSgBuKMU6igBlBFPx7UmKAGbaMU8ikxQAzFG2n4ox7UARkUbakxSYpAR7aNtSYoxTGRbaNtSYoxSER7aTbUuKNpoAi20bal20baBkO2jbU22jbQIh20bam20baLARbaNtS4o20wIglLtqXFGKAI9tAWpMUYoAaFoxT8UuKAGYpcU7FLigBuKAKdiloAbilxRS0AJS0YpcUAJRilooATFLiilxSASjtS0UCCkpaKYBRS0UwCiiigAooooAKKSlpAFFFFFgCiiigApaSgUgEzSUUUwCiiigAooopDCiiimAUUUUAFFFFIQUUUUAFGKKKAExRS0lMYlFLRRcBKWiigAooooASjFLSUDDFGKUUUCG4op1JQMTHtRilooEJikp1GKAG0U7FFADcUU6jFAxuKMU6igQmKTFOooGJijFLRQITFLiiigAxRiiigYYopaKACijFLQISgClooASloopAFFFFMAoFFFAC0UZooEFLSUUwFopKKQxaKKKBBRRSUxhRRRSEAooooAWkoopgLSUUlAC0CikpAFFFFABRRmjNAwoopDQAtFJRSEGaKKKBhS5pKKYC5ozSUUhC5ozSUUABooopjCiiigAooo/GkAlLSUUDFpKKKYBS0UUAFFFFABRScUtAgopM0tAxKWkpaACiiigBKKWigAooooAKKKKBBRSUtABRRRQAUUUUAFFFFIQClpKKAFopKKYxaKSloAKKKKACiiigAoFFFABRRRQAUUZooAM0UUUAFFFFAgooooCwUUlFIYtFFFMQUUGkpAf/Z" style="width:100%; max-width:280px; height:auto; filter: drop-shadow(0 15px 30px rgba(230,57,70,0.2));" alt="HeartPulse Hero Image"/>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_row(result: Optional[Dict]) -> None:
    status_label = "HIGH RISK" if result and result.get("pred") == 1 else "READY"
    st.markdown(
        f"""
        <div class="info-row page-section">
            <div class="info-card">
                <div class="info-title red">{icon_svg("alert", 16, "currentColor")} {status_label}</div>
                <div class="info-copy">Based on your inputs</div>
            </div>
            <div class="info-card">
                <div class="info-title">{icon_svg("activity", 16, "currentColor")} Stay proactive</div>
                <div class="info-copy">Regular checkups save lives</div>
            </div>
            <div class="info-card">
                <div class="info-title green">{icon_svg("shield", 16, "currentColor")} Private & Secure</div>
                <div class="info-copy">Your data is safe with us</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home(model, feature_order: List[str], defaults: Dict) -> None:
    render_hero()
    render_info_row(st.session_state.get("result"))

    st.markdown('<div class="main-grid page-section">', unsafe_allow_html=True)
    form_col, result_col = st.columns([1.55, 0.75], gap="large")

    with form_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="form-title">{icon_svg("user", 18, "currentColor")} Patient Information</div>',
            unsafe_allow_html=True,
        )

        vals = st.session_state.form_values.copy()
        left, right = st.columns(2, gap="medium")

        with left:
            st.markdown(f'<div class="input-label">{icon_svg("user", 16, "var(--red)")} Age (years)</div>', unsafe_allow_html=True)
            vals["age"] = st.slider("Age", 18, 100, int(vals["age"]), label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("heart", 16, "var(--red)")} Chest Pain Type</div>', unsafe_allow_html=True)
            vals["cp"] = st.selectbox("CP", [0, 1, 2, 3], index=int(vals["cp"]), format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x], label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("shield", 16, "var(--red)")} Serum Cholesterol (mg/dl)</div>', unsafe_allow_html=True)
            vals["chol"] = st.slider("Chol", 100, 600, int(vals["chol"]), step=1, label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("activity", 16, "var(--red)")} Resting ECG Results</div>', unsafe_allow_html=True)
            vals["restecg"] = st.selectbox("Rest ECG", [0, 1, 2], index=int(vals["restecg"]), format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x], label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("zap", 16, "var(--red)")} Exercise Induced Angina</div>', unsafe_allow_html=True)
            vals["exang"] = st.selectbox("Exang", [0, 1], index=int(vals["exang"]), format_func=lambda x: ["No", "Yes"][x], label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("trending-up", 16, "var(--red)")} Slope of Peak Exercise ST Segment</div>', unsafe_allow_html=True)
            vals["slope"] = st.selectbox("Slope", [0, 1, 2], index=int(vals["slope"]), format_func=lambda x: ["Up Sloping", "Flat", "Down Sloping"][x], label_visibility="collapsed")

        with right:
            st.markdown(f'<div class="input-label">{icon_svg("user", 16, "var(--red)")} Sex</div>', unsafe_allow_html=True)
            vals["sex"] = st.selectbox("Sex", [0, 1], index=int(vals["sex"]), format_func=lambda x: ["Female", "Male"][x], label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("droplet", 16, "var(--red)")} Resting Blood Pressure (mm Hg)</div>', unsafe_allow_html=True)
            vals["restbp"] = st.slider("Rest BP", 80, 220, int(vals["restbp"]), label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("heart", 16, "var(--red)")} Max Heart Rate Achieved</div>', unsafe_allow_html=True)
            vals["thalach"] = st.slider("Thalach", 60, 220, int(vals["thalach"]), label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("trending-down", 16, "var(--red)")} ST Depression (oldpeak)</div>', unsafe_allow_html=True)
            vals["oldpeak"] = st.slider("Oldpeak", 0.0, 6.0, float(vals["oldpeak"]), step=0.1, label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("droplet", 16, "var(--red)")} Fasting Blood Sugar > 120 mg/dl</div>', unsafe_allow_html=True)
            vals["fbs"] = st.selectbox("FBS", [0, 1], index=int(vals["fbs"]), format_func=lambda x: ["No", "Yes"][x], label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("layers", 16, "var(--red)")} Number of Major Vessels (0-3)</div>', unsafe_allow_html=True)
            vals["ca"] = st.selectbox("CA", [0, 1, 2, 3], index=min(int(vals["ca"]), 3), label_visibility="collapsed")
            st.markdown(f'<div class="input-label">{icon_svg("shield", 16, "var(--red)")} Thalassemia</div>', unsafe_allow_html=True)
            vals["thal"] = st.selectbox("Thal", [0, 1, 2, 3], index=int(vals["thal"]), format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x], label_visibility="collapsed")

        st.session_state.form_values = vals
        errors = validate_inputs(vals)

        if errors:
            for err in errors:
                st.warning(err)

        btn_left, btn_right = st.columns(2, gap="medium")
        with btn_left:
            predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True, disabled=bool(errors))
        with btn_right:
            if st.button("Reset All", use_container_width=True):
                st.session_state.form_values = defaults.copy()
                st.session_state.result = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        if predict_clicked and not errors:
            with st.spinner("Analyzing your inputs..."):
                frame = build_frame(st.session_state.form_values, feature_order)
                pred, confidence, prob_pos = predict(model, frame)
                st.session_state.result = {
                    "pred": pred,
                    "confidence": confidence,
                    "prob_pos": prob_pos,
                }

        result = st.session_state.get("result")

        if result is None:
            pred = 1
            prob_pos = 0.78
            confidence = 0.78
        else:
            pred = result["pred"]
            prob_pos = result["prob_pos"] if result["prob_pos"] is not None else 0.78
            confidence = result["confidence"] if result["confidence"] is not None else 0.78

        is_high = pred == 1
        status_text = "High Risk" if is_high else "Low Risk"
        risk_value = max(0.0, min(1.0, prob_pos if prob_pos is not None else confidence))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-title">{icon_svg("home", 18, "var(--red)")} Prediction Result</div>',
            unsafe_allow_html=True,
        )

        ring_p = risk_value * 100
        ring_bg = f"conic-gradient(from -90deg, #ff6b76 0%, #e63946 {ring_p}%, rgba(230,57,70,0.16) {ring_p}%, rgba(230,57,70,0.16) 100%)"
        
        st.markdown(
            f"""
            <div class="result-ring-wrap">
                <div class="result-ring" style="background: {ring_bg};">
                    <div class="result-ring-inner">
                        <div>
                            <div class="result-status">{status_text}</div>
                            <div class="result-percent">{risk_value * 100:.0f}%</div>
                            <div class="result-prob">Probability</div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        alert_copy = (
            "You are at high risk of heart disease. Please consult a cardiologist and maintain a healthy lifestyle."
            if is_high
            else "Low risk detected. Continue healthy routines and regular health checkups."
        )
        st.markdown(f'<div class="alert-box">{alert_copy}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="tip-title">{icon_svg("activity", 16, "currentColor")} What You Can Do</div>',
            unsafe_allow_html=True,
        )
        for item in [
            "Eat a heart-healthy diet",
            "Exercise regularly",
            "Manage stress",
            "Get regular checkups",
            "Avoid smoking & alcohol",
        ]:
            st.markdown(
                f'<div class="tip-item"><span class="tip-left">{icon_svg("heart", 16, "#e63946")}<span>{item}</span></span><span class="icon-muted">{icon_svg("chevron-right", 16, "currentColor")}</span></div>',
                unsafe_allow_html=True,
            )

        st.progress(risk_value)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_about(model, feature_order: List[str], dataset_meta: Dict, metrics: Dict) -> None:
    model_name = model.__class__.__name__
    if hasattr(model, "named_steps"):
        try:
            model_name = list(model.named_steps.values())[-1].__class__.__name__
        except Exception:
            model_name = model.__class__.__name__

    test_acc = metrics.get("test_accuracy")
    cv_acc = metrics.get("cv_accuracy")

    st.markdown('<div class="card page-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">About Model</div>', unsafe_allow_html=True)

    st.markdown(f"<p class='muted'><strong>Model type:</strong> {model_name}</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p class='muted'><strong>Dataset used:</strong> Heart disease clinical dataset ({dataset_meta['rows']} rows, {dataset_meta['cols']} columns).</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<p class='muted'><strong>Features used:</strong></p>", unsafe_allow_html=True)
    st.markdown("<ul class='about-list'>" + "".join([f"<li>{f}</li>" for f in feature_order]) + "</ul>", unsafe_allow_html=True)

    if test_acc is not None or cv_acc is not None:
        perf_parts = []
        if test_acc is not None:
            perf_parts.append(f"Test accuracy: {float(test_acc) * 100:.1f}%")
        if cv_acc is not None:
            perf_parts.append(f"Cross-validation accuracy: {float(cv_acc) * 100:.1f}%")
        perf_text = " | ".join(perf_parts)
    else:
        perf_text = "Training notebook reported approximately 89% holdout accuracy and around 82% cross-validation accuracy."

    st.markdown(f"<p class='muted'><strong>Performance:</strong> {perf_text}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_how_it_works() -> None:
    steps = [
        ("1", "User inputs data", "Clinical factors are entered using structured form controls."),
        ("2", "Data preprocessing", "Input values are mapped into the model feature schema."),
        ("3", "Model prediction", "The trained ML model computes class and probability."),
        ("4", "Output generation", "Risk level, probability, and action guidance are displayed."),
    ]

    st.markdown('<div class="card page-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-grid">', unsafe_allow_html=True)

    cols = st.columns(2, gap="medium")
    for idx, (num, title, copy) in enumerate(steps):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">{num}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-copy">{copy}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def main() -> None:
    inject_css()

    if not MODEL_PATH.exists():
        st.error("Model file heart_disease_pipeline.pkl not found.")
        st.stop()

    model, feature_order, metrics = load_pipeline(MODEL_PATH)
    profile = load_profile(DATA_PATH)
    dataset_meta = get_dataset_meta(DATA_PATH)
    defaults = default_values(profile)

    if "form_values" not in st.session_state:
        st.session_state.form_values = defaults.copy()
    if "result" not in st.session_state:
        st.session_state.result = None

    page = render_sidebar()

    if page == "Home":
        render_home(model, feature_order, defaults)
    elif page == "About Model":
        render_about(model, feature_order, dataset_meta, metrics)
    else:
        render_how_it_works()


if __name__ == "__main__":
    main()
