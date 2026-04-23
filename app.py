from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="PulseGuard - Heart Risk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_pipeline.pkl"
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

FALLBACK_FEATURES = [
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --paper: #eff2f7;
                --ink: #0b1220;
                --muted: #5b6475;
                --panel: rgba(255, 255, 255, 0.82);
                --panel-strong: rgba(255, 255, 255, 0.94);
                --line: rgba(13, 21, 37, 0.12);
                --blue: #0284c7;
                --teal: #14b8a6;
                --good: #059669;
                --bad: #dc2626;
            }

            .stApp {
                background:
                    radial-gradient(900px 600px at -8% -10%, rgba(2,132,199,0.22), transparent 45%),
                    radial-gradient(860px 600px at 108% -8%, rgba(20,184,166,0.2), transparent 44%),
                    var(--paper);
            }

            .block-container {
                max-width: 1220px;
                padding-top: 1.0rem;
                padding-bottom: 2rem;
            }

            .orb {
                position: fixed;
                width: 260px;
                height: 260px;
                border-radius: 50%;
                filter: blur(48px);
                z-index: -1;
                animation: drift 13s ease-in-out infinite;
                opacity: 0.24;
            }

            .orb.one {
                background: #0284c7;
                left: 4%;
                top: 12%;
            }

            .orb.two {
                background: #14b8a6;
                right: 6%;
                top: 26%;
                animation-delay: 1.9s;
            }

            @keyframes drift {
                0%, 100% { transform: translateY(0px) translateX(0px); }
                50% { transform: translateY(-16px) translateX(8px); }
            }

            .hero-card {
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 1.25rem 1.3rem;
                background: linear-gradient(145deg, var(--panel-strong), var(--panel));
                backdrop-filter: blur(10px);
                box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
                margin-bottom: 0.9rem;
                animation: rise 340ms ease-out 1;
            }

            @keyframes rise {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0px); }
            }

            .hero-title {
                margin: 0;
                color: var(--ink);
                font-size: 2.0rem;
                letter-spacing: -0.02em;
                font-weight: 800;
            }

            .hero-sub {
                margin: 0.45rem 0 0 0;
                color: var(--muted);
                font-size: 1.0rem;
            }

            .mini-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.7rem;
                margin-top: 0.95rem;
            }

            .mini-card {
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 0.55rem 0.7rem;
                background: rgba(255,255,255,0.74);
            }

            .mini-card h4 {
                margin: 0;
                color: var(--ink);
                font-size: 0.85rem;
                font-weight: 600;
            }

            .mini-card p {
                margin: 0.18rem 0 0 0;
                color: var(--muted);
                font-size: 0.82rem;
            }

            .panel-card {
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem 0.6rem 1rem;
                background: linear-gradient(160deg, var(--panel-strong), var(--panel));
                box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
            }

            .panel-head {
                margin: 0 0 0.5rem 0;
                color: var(--ink);
                font-size: 1.2rem;
                font-weight: 700;
            }

            .result-wrap {
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 0.95rem 1rem 0.95rem 1rem;
                margin-top: 0.9rem;
                background: rgba(255,255,255,0.9);
                box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
                animation: rise 260ms ease-out 1;
            }

            .risk-pill {
                display: inline-block;
                padding: 0.36rem 0.7rem;
                border-radius: 999px;
                font-weight: 700;
                font-size: 0.88rem;
                margin-bottom: 0.5rem;
                border: 1px solid transparent;
            }

            .risk-pill.high {
                color: #7f1d1d;
                background: rgba(239,68,68,0.16);
                border-color: rgba(220,38,38,0.28);
            }

            .risk-pill.low {
                color: #064e3b;
                background: rgba(16,185,129,0.17);
                border-color: rgba(5,150,105,0.3);
            }

            .result-main {
                color: var(--ink);
                font-size: 1.45rem;
                font-weight: 800;
                margin: 0;
            }

            .result-sub {
                color: var(--muted);
                margin: 0.28rem 0 0 0;
                font-size: 0.95rem;
            }

            .stButton > button {
                border-radius: 12px;
                min-height: 2.8rem;
                font-weight: 650;
                transition: transform 130ms ease, box-shadow 130ms ease;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 8px 18px rgba(2, 132, 199, 0.24);
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(130deg, var(--blue), var(--teal));
                border: 0;
                color: white;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid var(--line);
                background: linear-gradient(180deg, #0f172a, #111827);
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] li,
            [data-testid="stSidebar"] div {
                color: #e5e7eb;
            }

            [data-testid="stSidebar"] .stAlert {
                background: rgba(22, 163, 74, 0.22);
                color: #d1fae5;
                border: 1px solid rgba(34,197,94,0.36);
            }

            [data-testid="stNumberInput"] input,
            [data-testid="stTextInput"] input,
            [data-baseweb="select"] > div {
                background: rgba(255, 255, 255, 0.94) !important;
                color: #0b1220 !important;
                border: 1px solid rgba(13, 21, 37, 0.16) !important;
            }

            [data-testid="stNumberInput"] label p,
            [data-testid="stSelectbox"] label p,
            [data-testid="stSlider"] label p,
            .stMarkdown p,
            .stMarkdown li {
                color: #0b1220;
            }

            @media (max-width: 980px) {
                .hero-title { font-size: 1.45rem; }
                .mini-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); }
            }
        </style>
        <div class="orb one"></div>
        <div class="orb two"></div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_order = payload.get("features") or FALLBACK_FEATURES
    else:
        model = payload
        feature_order = list(getattr(model, "feature_names_in_", FALLBACK_FEATURES))
    return model, feature_order


@st.cache_data(show_spinner=False)
def load_data_profile(path: Path):
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


def defaults_from_profile(profile: dict) -> dict:
    def v(name: str, fallback: float):
        return profile.get(name, {}).get("median", fallback)

    return {
        "ca": int(round(v("ca", 0))),
        "cp": int(round(v("cp", 1))),
        "exang": int(round(v("exang", 0))),
        "thalach": int(round(v("thalach", 150))),
        "oldpeak": float(round(v("oldpeak", 1.0), 1)),
        "thal": int(round(v("thal", 2))),
        "slope": int(round(v("slope", 1))),
        "sex": int(round(v("sex", 1))),
        "age": int(round(v("age", 55))),
        "restecg": int(round(v("restecg", 1))),
        "chol": int(round(v("chol", 240))),
    }


def validate(values: dict) -> list[str]:
    issues = []
    if not 18 <= values["age"] <= 100:
        issues.append("Age should be between 18 and 100")
    if not 100 <= values["chol"] <= 600:
        issues.append("Cholesterol should be between 100 and 600")
    if not 60 <= values["thalach"] <= 230:
        issues.append("Max heart rate should be between 60 and 230")
    if not 0.0 <= values["oldpeak"] <= 8.0:
        issues.append("Oldpeak should be between 0.0 and 8.0")
    return issues


def build_input_frame(values: dict, feature_order: list[str]) -> pd.DataFrame:
    row = {k: values[k] for k in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def infer(model, frame: pd.DataFrame):
    pred = int(model.predict(frame)[0])
    confidence = None
    prob_pos = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        prob_pos = float(proba[1])
        confidence = float(proba[pred])
    return pred, confidence, prob_pos


def show_result(pred: int, confidence: float | None, prob_pos: float | None):
    is_high = pred == 1
    pill_class = "high" if is_high else "low"
    status = "High Risk" if is_high else "Low Risk"

    st.markdown("<div class='result-wrap'>", unsafe_allow_html=True)
    st.markdown(f"<span class='risk-pill {pill_class}'>{status}</span>", unsafe_allow_html=True)
    st.markdown(f"<p class='result-main'>{'Heart disease likely' if is_high else 'Heart disease unlikely'}</p>", unsafe_allow_html=True)

    if confidence is not None:
        st.markdown(f"<p class='result-sub'>Model confidence: {confidence * 100:.1f}%</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result-sub'>Confidence unavailable for this model.</p>", unsafe_allow_html=True)

    if prob_pos is not None:
        st.progress(max(0.0, min(1.0, prob_pos)))
        st.caption(f"Predicted probability of heart disease: {prob_pos * 100:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)


inject_styles()

st.markdown(
    """
    <div class="hero-card">
        <h1 class="hero-title">PulseGuard Heart Risk</h1>
        <p class="hero-sub">Responsive clinical risk check with a trained machine learning pipeline.</p>
        <div class="mini-grid">
            <div class="mini-card">
                <h4>Latency First</h4>
                <p>Cached model loading and single-pass prediction.</p>
            </div>
            <div class="mini-card">
                <h4>Clean Input Flow</h4>
                <p>Hardcoded medical controls with strict validation.</p>
            </div>
            <div class="mini-card">
                <h4>Deployment Ready</h4>
                <p>Uses saved pipeline artifact directly.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.caption("Premium UI for heart risk prediction")
    st.caption("Sex: 0 Female, 1 Male")
    st.caption("Exang: 0 No, 1 Yes")
    st.caption("cp: 0-3, thal: 0-3, slope: 0-2, restecg: 0-2")

if not MODEL_PATH.exists():
    st.error("Model file not found. Place heart_disease_pipeline.pkl in this folder.")
    st.stop()

model, feature_order = load_pipeline(MODEL_PATH)
profile = load_data_profile(DATA_PATH)
defaults = defaults_from_profile(profile)

if "form_values" not in st.session_state:
    st.session_state.form_values = defaults.copy()
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.success("Model ready")
    if st.button("Reset form", use_container_width=True):
        st.session_state.form_values = defaults.copy()
        st.session_state.result = None
        st.rerun()

st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
st.markdown("<h2 class='panel-head'>Patient inputs</h2>", unsafe_allow_html=True)

left, right = st.columns(2, gap="large")
vals = st.session_state.form_values.copy()

with left:
    vals["ca"] = st.selectbox("Major vessels (ca)", [0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4].index(int(vals["ca"])) if int(vals["ca"]) in [0, 1, 2, 3, 4] else 0)
    vals["exang"] = st.selectbox("Exercise induced angina", [0, 1], index=int(vals["exang"]) if int(vals["exang"]) in [0, 1] else 0)
    vals["oldpeak"] = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=8.0, value=float(vals["oldpeak"]), step=0.1)
    vals["slope"] = st.selectbox("Slope", [0, 1, 2], index=int(vals["slope"]) if int(vals["slope"]) in [0, 1, 2] else 0)
    vals["age"] = st.number_input("Age", min_value=18, max_value=100, value=int(vals["age"]), step=1)
    vals["chol"] = st.number_input("Cholesterol", min_value=100, max_value=600, value=int(vals["chol"]), step=1)

with right:
    vals["cp"] = st.selectbox("Chest pain type", [0, 1, 2, 3], index=int(vals["cp"]) if int(vals["cp"]) in [0, 1, 2, 3] else 0)
    vals["thalach"] = st.number_input("Max heart rate", min_value=60, max_value=230, value=int(vals["thalach"]), step=1)
    vals["thal"] = st.selectbox("Thal", [0, 1, 2, 3], index=int(vals["thal"]) if int(vals["thal"]) in [0, 1, 2, 3] else 0)
    vals["sex"] = st.selectbox("Sex", [0, 1], index=int(vals["sex"]) if int(vals["sex"]) in [0, 1] else 0)
    vals["restecg"] = st.selectbox("Resting ECG", [0, 1, 2], index=int(vals["restecg"]) if int(vals["restecg"]) in [0, 1, 2] else 0)

st.session_state.form_values = vals
errors = validate(vals)
if errors:
    for err in errors:
        st.warning(err)

cta_col, _ = st.columns([1, 2.4])
with cta_col:
    do_predict = st.button("Predict now", type="primary", use_container_width=True, disabled=bool(errors))

st.markdown("</div>", unsafe_allow_html=True)

if do_predict and not errors:
    frame = build_input_frame(vals, feature_order)
    with st.spinner("Scoring patient..."):
        pred, confidence, prob_pos = infer(model, frame)
    st.session_state.result = {
        "pred": pred,
        "confidence": confidence,
        "prob_pos": prob_pos,
    }

if st.session_state.result is not None:
    r = st.session_state.result
    show_result(r["pred"], r["confidence"], r["prob_pos"])

    with st.expander("Submitted values", expanded=False):
        st.dataframe(pd.DataFrame([vals]), use_container_width=True)
