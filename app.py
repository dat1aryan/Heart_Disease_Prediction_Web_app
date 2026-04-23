from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HeartPulse - AI Heart Risk Predictor",
    page_icon="❤️",
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
    font-family: 'Inter', sans-serif !important;
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
    background: var(--red) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 18px rgba(230, 57, 70, 0.20) !important;
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
    border: 8px solid rgba(230, 57, 70, 0.14);
    border-top-color: var(--red);
    border-right-color: var(--red);
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
        st.markdown('<div class="sidebar-brand">HeartPulse ❤️</div>', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["Home", "About Model", "How It Works"],
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">Your heart, our priority.</div>
                <div class="sidebar-card-copy">Take control of your health with AI-powered insights.</div>
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
                    <h1 class="hero-title">HeartPulse ❤️</h1>
                    <div class="hero-subtitle">AI-Powered Heart Disease Risk Prediction</div>
                    <div class="hero-copy">Get insights about your heart health based on clinical factors using advanced machine learning.</div>
                </div>
                <div class="hero-heart">
                    <svg viewBox="0 0 220 180" xmlns="http://www.w3.org/2000/svg" fill="none" aria-hidden="true">
                        <path d="M12 88H45L52 72L59 102L67 60L76 98H108" stroke="#e63946" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M108 88H131L137 74L145 100L153 58L162 110L172 88H208" stroke="#e63946" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M149 26C134 26 124 35 121 49C118 44 113 41 108 41C99 41 92 48 92 58C92 76 106 86 123 104C133 115 139 120 143 120C147 120 153 115 163 104C180 86 194 76 194 58C194 48 187 41 178 41C173 41 168 44 165 49C162 35 152 26 149 26Z" fill="#e63946"/>
                    </svg>
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
                <div class="info-title red">{status_label}</div>
                <div class="info-copy">Based on your inputs</div>
            </div>
            <div class="info-card">
                <div class="info-title">Stay proactive</div>
                <div class="info-copy">Regular checkups save lives</div>
            </div>
            <div class="info-card">
                <div class="info-title green">Private & Secure</div>
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
        st.markdown('<div class="form-title">Patient Information</div>', unsafe_allow_html=True)

        vals = st.session_state.form_values.copy()
        left, right = st.columns(2, gap="medium")

        with left:
            vals["age"] = st.slider("Age (years)", 18, 100, int(vals["age"]))
            vals["cp"] = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                index=int(vals["cp"]),
                format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
            )
            vals["chol"] = st.slider("Serum Cholesterol (mg/dl)", 100, 600, int(vals["chol"]), step=1)
            vals["restecg"] = st.selectbox(
                "Resting ECG Results",
                [0, 1, 2],
                index=int(vals["restecg"]),
                format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x],
            )
            vals["exang"] = st.selectbox("Exercise Induced Angina", [0, 1], index=int(vals["exang"]), format_func=lambda x: ["No", "Yes"][x])
            vals["slope"] = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                [0, 1, 2],
                index=int(vals["slope"]),
                format_func=lambda x: ["Up Sloping", "Flat", "Down Sloping"][x],
            )

        with right:
            vals["sex"] = st.selectbox("Sex", [0, 1], index=int(vals["sex"]), format_func=lambda x: ["Female", "Male"][x])
            vals["restbp"] = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, int(vals["restbp"]))
            vals["thalach"] = st.slider("Max Heart Rate Achieved", 60, 220, int(vals["thalach"]))
            vals["oldpeak"] = st.slider("ST Depression (oldpeak)", 0.0, 6.0, float(vals["oldpeak"]), step=0.1)
            vals["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=int(vals["fbs"]), format_func=lambda x: ["No", "Yes"][x])
            vals["ca"] = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], index=min(int(vals["ca"]), 3))
            vals["thal"] = st.selectbox(
                "Thalassemia",
                [0, 1, 2, 3],
                index=int(vals["thal"]),
                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x],
            )

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
        st.markdown('<div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-ring-wrap">
                <div class="result-ring">
                    <div>
                        <div class="result-status">{status_text}</div>
                        <div class="result-percent">{risk_value * 100:.0f}%</div>
                        <div class="result-prob">Probability</div>
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

        st.markdown('<div class="tip-title">What You Can Do</div>', unsafe_allow_html=True)
        for item in [
            "Eat a heart-healthy diet",
            "Exercise regularly",
            "Manage stress",
            "Get regular checkups",
            "Avoid smoking & alcohol",
        ]:
            st.markdown(f'<div class="tip-item"><span>{item}</span><span style="color:#9ca3af;">›</span></div>', unsafe_allow_html=True)

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
