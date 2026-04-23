from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Risk",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_pipeline.pkl"
DATA_PATH = Path(__file__).resolve().parent / "heart.csv"

FEATURES_FALLBACK = [
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

FEATURE_META = {
    "age": {"label": "Age", "kind": "int", "min": 18, "max": 100, "step": 1, "help": "Age in years."},
    "sex": {"label": "Sex", "kind": "cat", "options": [0, 1], "help": "0 = Female, 1 = Male."},
    "cp": {"label": "Chest Pain Type", "kind": "cat", "options": [0, 1, 2, 3], "help": "Chest pain category."},
    "chol": {"label": "Cholesterol", "kind": "int", "min": 100, "max": 600, "step": 1, "help": "Serum cholesterol in mg/dl."},
    "thalach": {"label": "Max Heart Rate", "kind": "int", "min": 60, "max": 230, "step": 1, "help": "Maximum heart rate achieved."},
    "exang": {"label": "Exercise Induced Angina", "kind": "cat", "options": [0, 1], "help": "0 = No, 1 = Yes."},
    "oldpeak": {"label": "ST Depression (Oldpeak)", "kind": "float", "min": 0.0, "max": 8.0, "step": 0.1, "help": "ST depression induced by exercise."},
    "slope": {"label": "Slope", "kind": "cat", "options": [0, 1, 2], "help": "Slope of peak exercise ST segment."},
    "ca": {"label": "Major Vessels", "kind": "cat", "options": [0, 1, 2, 3, 4], "help": "Number of major vessels colored by fluoroscopy."},
    "thal": {"label": "Thal", "kind": "cat", "options": [0, 1, 2, 3], "help": "Thalassemia category."},
    "restecg": {"label": "Resting ECG", "kind": "cat", "options": [0, 1, 2], "help": "Resting electrocardiographic results."},
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f4f7fb;
                --card: rgba(255, 255, 255, 0.88);
                --text: #0f172a;
                --muted: #475569;
                --ok: #16a34a;
                --warn: #dc2626;
                --line: rgba(15, 23, 42, 0.08);
                --accent: #0ea5e9;
                --accent2: #22c55e;
            }
            .stApp {
                background:
                    radial-gradient(1100px 500px at 0% -10%, rgba(14, 165, 233, 0.18), transparent 45%),
                    radial-gradient(900px 480px at 100% -20%, rgba(34, 197, 94, 0.16), transparent 50%),
                    var(--bg);
            }
            .block-container {
                max-width: 1160px;
                padding-top: 1.1rem;
                padding-bottom: 2rem;
            }
            .hero {
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1.1rem 1.25rem;
                background: var(--card);
                backdrop-filter: blur(8px);
                box-shadow: 0 10px 28px rgba(2, 8, 23, 0.06);
                margin-bottom: 0.8rem;
            }
            .hero h1 {
                margin: 0;
                color: var(--text);
                font-size: 1.7rem;
                letter-spacing: -0.01em;
            }
            .hero p {
                margin: 0.4rem 0 0 0;
                color: var(--muted);
                font-size: 0.95rem;
            }
            .section-card {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.8rem 0.9rem 0.4rem 0.9rem;
                background: var(--card);
                box-shadow: 0 8px 24px rgba(2, 8, 23, 0.05);
                transition: transform 160ms ease, box-shadow 160ms ease;
            }
            .section-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 28px rgba(2, 8, 23, 0.08);
            }
            .result-card {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem 1rem 0.9rem 1rem;
                margin-top: 0.8rem;
                background: var(--card);
                box-shadow: 0 10px 28px rgba(2, 8, 23, 0.06);
            }
            .risk-high {
                color: var(--warn);
                font-size: 1.25rem;
                font-weight: 700;
            }
            .risk-low {
                color: var(--ok);
                font-size: 1.25rem;
                font-weight: 700;
            }
            .caption {
                color: var(--muted);
                font-size: 0.9rem;
                margin-top: 0.2rem;
            }
            div[data-testid="stSidebar"] {
                border-right: 1px solid var(--line);
            }
            .stButton > button[kind="primary"] {
                border-radius: 12px;
                border: none;
                background: linear-gradient(135deg, var(--accent), var(--accent2));
                color: white;
                font-weight: 600;
                min-height: 2.8rem;
            }
            .stButton > button {
                border-radius: 10px;
            }
            @media (max-width: 900px) {
                .hero h1 { font-size: 1.35rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline(model_path: Path):
    payload = joblib.load(model_path)

    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        features = payload.get("features") or FEATURES_FALLBACK
    else:
        model = payload
        features = list(getattr(model, "feature_names_in_", FEATURES_FALLBACK))

    return model, features


@st.cache_data(show_spinner=False)
def load_reference_stats(data_path: Path):
    if not data_path.exists():
        return {}
    df = pd.read_csv(data_path)
    stats = {}
    for feature in df.columns:
        if feature == "target":
            continue
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            stats[feature] = {
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return stats


def build_default_values(feature_order, stats):
    defaults = {}
    for feature in feature_order:
        meta = FEATURE_META.get(feature, {"kind": "float", "min": 0.0, "max": 1.0, "step": 0.1, "label": feature})
        if meta["kind"] == "cat":
            defaults[feature] = meta["options"][0]
        else:
            median = stats.get(feature, {}).get("median")
            if median is None:
                median = (meta.get("min", 0) + meta.get("max", 1)) / 2
            if meta["kind"] == "int":
                defaults[feature] = int(round(median))
            else:
                defaults[feature] = float(round(median, 1))
    return defaults


def validate_inputs(values):
    issues = []

    age = values.get("age")
    chol = values.get("chol")
    oldpeak = values.get("oldpeak")
    thalach = values.get("thalach")

    if age is not None and not (18 <= age <= 100):
        issues.append("Age should be between 18 and 100.")
    if chol is not None and not (100 <= chol <= 600):
        issues.append("Cholesterol should be between 100 and 600 mg/dl.")
    if oldpeak is not None and not (0 <= oldpeak <= 8):
        issues.append("Oldpeak should be between 0.0 and 8.0.")
    if thalach is not None and not (60 <= thalach <= 230):
        issues.append("Max heart rate should be between 60 and 230.")

    return issues


def prepare_input_frame(values, feature_order):
    row = {feature: values[feature] for feature in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def predict(model, input_df):
    pred = int(model.predict(input_df)[0])

    confidence = None
    prob_pos = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        prob_pos = float(proba[1])
        confidence = float(proba[pred])

    return pred, confidence, prob_pos


def render_result(pred, confidence, prob_pos):
    label = "High Risk" if pred == 1 else "Low Risk"
    risk_class = "risk-high" if pred == 1 else "risk-low"

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='{risk_class}'>{label}</div>", unsafe_allow_html=True)

    if confidence is not None:
        st.markdown(f"<div class='caption'>Confidence: {confidence * 100:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='caption'>Confidence score unavailable for this model type.</div>", unsafe_allow_html=True)

    if prob_pos is not None:
        st.write("Risk probability")
        st.progress(min(max(prob_pos, 0.0), 1.0))
        st.caption(f"Probability of heart disease: {prob_pos * 100:.1f}%")

    st.markdown("</div>", unsafe_allow_html=True)


inject_styles()

st.markdown(
    """
    <div class="hero">
        <h1>Heart Disease Risk Prediction</h1>
        <p>Fast clinical-style screening interface powered by a trained machine learning pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("About")
    st.caption("Model-backed prediction app for heart disease risk assessment.")

    if MODEL_PATH.exists():
        st.success("Model file detected")
    else:
        st.error("Model file missing")

    st.markdown("Feature coding")
    st.caption("sex: 0 Female, 1 Male")
    st.caption("exang: 0 No, 1 Yes")
    st.caption("cp: 0-3, thal: 0-3, slope: 0-2, restecg: 0-2")


if not MODEL_PATH.exists():
    st.error("Model file heart_disease_pipeline.pkl was not found in the app directory.")
    st.stop()

model, feature_order = load_pipeline(MODEL_PATH)
reference_stats = load_reference_stats(DATA_PATH)
defaults = build_default_values(feature_order, reference_stats)

if "input_values" not in st.session_state:
    st.session_state.input_values = defaults.copy()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    if st.button("Reset Inputs", use_container_width=True):
        st.session_state.input_values = defaults.copy()
        st.session_state.last_result = None
        st.rerun()

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Patient Inputs")

left_col, right_col = st.columns(2, gap="large")
current_values = st.session_state.input_values.copy()

for idx, feature in enumerate(feature_order):
    meta = FEATURE_META.get(feature, {"label": feature, "kind": "float", "min": 0.0, "max": 1.0, "step": 0.1, "help": ""})
    label = meta["label"]
    widget_key = f"input_{feature}"

    target_col = left_col if idx % 2 == 0 else right_col

    with target_col:
        if meta["kind"] == "cat":
            options = meta["options"]
            default_index = options.index(current_values.get(feature, options[0])) if current_values.get(feature, options[0]) in options else 0
            value = st.selectbox(label, options=options, index=default_index, help=meta.get("help", ""), key=widget_key)
        elif meta["kind"] == "int":
            value = st.number_input(
                label,
                min_value=int(meta["min"]),
                max_value=int(meta["max"]),
                value=int(current_values.get(feature, defaults.get(feature, int(meta["min"])))),
                step=int(meta["step"]),
                help=meta.get("help", ""),
                key=widget_key,
            )
        else:
            value = st.number_input(
                label,
                min_value=float(meta["min"]),
                max_value=float(meta["max"]),
                value=float(current_values.get(feature, defaults.get(feature, float(meta["min"])))),
                step=float(meta["step"]),
                help=meta.get("help", ""),
                key=widget_key,
            )

        current_values[feature] = value

st.session_state.input_values = current_values
issues = validate_inputs(current_values)
if issues:
    for issue in issues:
        st.warning(issue)

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_pressed = st.button(
        "Predict Heart Disease Risk",
        type="primary",
        use_container_width=True,
        disabled=bool(issues),
    )

st.markdown("</div>", unsafe_allow_html=True)

if predict_pressed and not issues:
    input_df = prepare_input_frame(st.session_state.input_values, feature_order)
    with st.spinner("Running prediction..."):
        pred, confidence, prob_pos = predict(model, input_df)
    st.session_state.last_result = {
        "pred": pred,
        "confidence": confidence,
        "prob_pos": prob_pos,
    }

if st.session_state.last_result is not None:
    result = st.session_state.last_result
    render_result(result["pred"], result["confidence"], result["prob_pos"])

    with st.expander("Input snapshot", expanded=False):
        st.dataframe(pd.DataFrame([st.session_state.input_values]), use_container_width=True)
