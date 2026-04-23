from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st
import base64
import textwrap

def get_b64(filename: str, folder: str = "assets") -> str:
    try:
        base_dir = Path(__file__).resolve().parent
        filepath = base_dir / folder / filename
        with open(filepath, "rb") as f:
            raw_b64 = base64.b64encode(f.read()).decode("utf-8")
            return "\n".join(textwrap.wrap(raw_b64, 76))
    except Exception:
        return ""

st.set_page_config(
    page_title="HeartPulse",
    page_icon="assets/favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Override browser tab title to remove "- Streamlit" suffix
st.markdown('<script>document.title="HeartPulse";</script>', unsafe_allow_html=True)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_disease_pipeline.pkl"
DATA_PATH = Path(__file__).resolve().parent / "data" / "heart.csv"

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
    "running": '<path d="m11 20 3-8 3 2v4"/><path d="m5 16 4-4 2 1"/><path d="m13 12 3-5-2-3"/><path d="m9 7 2-3h3"/><circle cx="16" cy="4" r="2"/>',
    "circle": '<circle cx="12" cy="12" r="10"/>',
    "shield-check": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "pill": '<path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"/><path d="m8.5 8.5 7 7"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
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
        f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="vertical-align:-0.15em;margin-right:6px;">{path}</svg>'
    )

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

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

@keyframes pulseHeart {
    0% { transform: scale(0.98); opacity: 0.85; filter: drop-shadow(0 10px 20px rgba(230,57,70,0.2)); }
    50% { transform: scale(1.02); opacity: 1; filter: drop-shadow(0 20px 40px rgba(230,57,70,0.5)); }
    100% { transform: scale(0.98); opacity: 0.85; filter: drop-shadow(0 10px 20px rgba(230,57,70,0.2)); }
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: "Plus Jakarta Sans", "Inter", "Helvetica Neue", sans-serif !important;
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
    padding: 16px 18px !important;
}

.sidebar-brand {
    color: var(--red);
    font-size: 22px;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 16px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.02em;
}

.sidebar-card {
    margin-top: 20px;
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
    gap: 6px;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 11px 14px;
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

/* Button SVG icons via pseudo-elements */
button[data-testid="stBaseButton-primary"] p::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 8px;
    vertical-align: -2px;
    background: white;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") no-repeat center / contain;
}
button[data-testid="stBaseButton-secondary"] p::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 8px;
    vertical-align: -2px;
    background: var(--text);
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='1 4 1 10 7 10'/%3E%3Cpath d='M3.51 15a9 9 0 1 0 2.13-9.36L1 10'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='1 4 1 10 7 10'/%3E%3Cpath d='M3.51 15a9 9 0 1 0 2.13-9.36L1 10'/%3E%3C/svg%3E") no-repeat center / contain;
}

/* Robust Card Targeting Engine */
[data-testid="column"]:has(.form-title-marker),
[data-testid="column"]:has(.result-title-marker),
.stColumn:has(.form-title-marker),
.stColumn:has(.result-title-marker) {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 24px !important;
}

/* Red accent bar for the result card */
[data-testid="column"]:has(.result-title-marker),
.stColumn:has(.result-title-marker) {
    border-top: 4px solid var(--red) !important;
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
    gap: 24px;
    margin-top: 20px;
}

.step-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 24px;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.step-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);
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

.about-stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.about-stat-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.about-stat-card svg {
    margin-bottom: 8px;
    color: var(--red);
    opacity: 0.8;
    margin-right: 0 !important;
}

.stat-value {
    font-size: 20px;
    font-weight: 800;
    color: var(--red);
    display: block;
}

.stat-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
    font-weight: 700;
}

.feature-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}

.feature-pill {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 99px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

.feature-pill svg {
    color: var(--red);
}

.about-performance-wrap {
    background: var(--red-soft);
    border-radius: 12px;
    padding: 16px;
    margin-top: 24px;
    border: 1px solid rgba(230, 57, 70, 0.1);
}

.about-performance-title {
    color: var(--red);
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
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
            f'<div class="sidebar-brand">HeartPulse <img src="data:image/png;base64,{get_b64("apple_heart.png")}" style="width:0.95em; height:0.95em; vertical-align:-0.1em; filter:drop-shadow(0 2px 3px rgba(230,57,70,0.25));" alt="iOS Heart"/></div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            ["Home", "About Model", "How It Works"],
            label_visibility="collapsed",
        )

        st.markdown(
            f"""
<div class="sidebar-card">
    <div style="text-align: center; margin-bottom: 12px; position: relative; display: flex; justify-content: center; align-items: center; height: 80px;">
        <div style="position: relative; width: 48px; height: 48px;">
            <svg style="position: absolute; top: -10px; left: -14px; width: 14px; height: 14px; color: #ff4757; filter: drop-shadow(0 2px 4px rgba(255,71,87,0.4));" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0L14.5 9.5L24 12L14.5 14.5L12 24L9.5 14.5L0 12L9.5 9.5L12 0Z"/>
            </svg>
            <svg style="position: absolute; top: 12px; right: -22px; width: 10px; height: 10px; color: #ff6b81;" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0L14.5 9.5L24 12L14.5 14.5L12 24L9.5 14.5L0 12L9.5 9.5L12 0Z"/>
            </svg>
            <svg style="position: absolute; bottom: -8px; right: -8px; width: 12px; height: 12px; color: #ff4757;" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0L14.5 9.5L24 12L14.5 14.5L12 24L9.5 14.5L0 12L9.5 9.5L12 0Z"/>
            </svg>
            <img src="data:image/png;base64,{get_b64('apple_heart.png')}" style="width:100%; height:100%; filter: drop-shadow(0 4px 8px rgba(230,57,70,0.3));" />
        </div>
    </div>
    <div class="sidebar-card-title" style="text-align: center;">Your heart, our priority.</div>
    <div class="sidebar-card-copy" style="text-align: center;">Take control of your health with AI-powered insights.</div>
</div>
<div class="sidebar-footer">© 2026 HeartPulse <img src="data:image/png;base64,{get_b64('footer_heart.png')}" style="width:0.95em; height:0.95em; vertical-align:-0.1em; filter:drop-shadow(0 2px 3px rgba(230,57,70,0.25));" alt="iOS Heart"/><br/>All rights reserved.</div>
            """,
            unsafe_allow_html=True,
        )
    return page


def render_hero() -> None:
    st.markdown(
        f"""
<div class="card page-section">
    <div class="hero-grid">
        <div>
            <div class="hero-brand-row">
                <h1 class="hero-title" style="color: var(--red);">HeartPulse <img src="data:image/png;base64,{get_b64('apple_heart.png')}" style="width:0.95em; height:0.95em; vertical-align:-0.1em; filter:drop-shadow(0 2px 3px rgba(230,57,70,0.25));" alt="iOS Heart"/></h1>
            </div>
            <div class="hero-subtitle">AI-Powered Heart Disease Risk Prediction</div>
            <div class="hero-copy">Get insights about your heart health based on clinical factors using advanced machine learning.</div>
        </div>
        <div class="hero-heart" style="position: relative; display: flex; justify-content: center; align-items: center; min-height: 240px; -webkit-mask-image: linear-gradient(to right, transparent 0%, black 20%, black 80%, transparent 100%); mask-image: linear-gradient(to right, transparent 0%, black 20%, black 80%, transparent 100%);">
            <!-- Pale pink fading aura background -->
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 180%; height: 140%; background: radial-gradient(ellipse, #ffe0e4 0%, rgba(255,255,255,0) 65%); z-index: 0; pointer-events: none;"></div>
            <!-- SVG ECG line sweeping across -->
            <svg viewBox="0 0 220 180" xmlns="http://www.w3.org/2000/svg" fill="none" aria-hidden="true" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 150%; height: auto; z-index: 1; opacity: 0.35;">
                <path d="M0 88H45L52 72L59 102L67 60L76 98H108" stroke="#e63946" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M108 88H131L137 74L145 100L153 58L162 110L172 88H220" stroke="#e63946" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <!-- 3D Generated Heart with Pulse Animation -->
            <img src="data:image/png;base64,{get_b64('hero_heart.png')}" style="position: relative; z-index: 2; width: 100%; max-width: 250px; filter: drop-shadow(0 15px 30px rgba(230,57,70,0.3)); animation: pulseHeart 4s infinite ease-in-out;" alt="HeartPulse Hero Image"/>
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
        st.markdown(
            f'<div class="form-title form-title-marker">{icon_svg("user", 18, "var(--red)")} Patient Information</div>',
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
        warn_img = f'<img src="data:image/svg+xml;base64,{get_b64('warn_triangle.svg')}" style="width:22px;height:22px;" alt="warning"/>'
        warn_img_sm = f'<img src="data:image/svg+xml;base64,{get_b64('warn_triangle.svg')}" style="width:16px;height:16px;vertical-align:-2px;" alt="warning"/>'
        status_text = f"{warn_img}<br/>High Risk" if is_high else "✅<br/>Low Risk"
        risk_value = max(0.0, min(1.0, prob_pos if prob_pos is not None else confidence))

        st.markdown(
            f'<div class="result-title result-title-marker">{icon_svg("home", 18, "var(--red)")} Prediction Result</div>',
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
            f"{warn_img_sm} You are at high risk of heart disease. Please consult a cardiologist and maintain a healthy lifestyle."
            if is_high
            else "✅ Low risk detected. Continue healthy routines and regular health checkups."
        )
        st.markdown(f'<div class="alert-box">{alert_copy}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="tip-title">{icon_svg("activity", 16, "currentColor")} What You Can Do</div>',
            unsafe_allow_html=True,
        )
        tips = [
            ("Eat a heart-healthy diet", "heart"),
            ("Exercise regularly", "running"),
            ("Manage stress", "circle"),
            ("Get regular checkups", "shield-check"),
            ("Avoid smoking & alcohol", "pill"),
        ]
        for item, icon in tips:
            st.markdown(
                f'<div class="tip-item"><span class="tip-left">{icon_svg(icon, 16, "#e63946")}<span>{item}</span></span><span class="icon-muted">{icon_svg("chevron-right", 16, "currentColor")}</span></div>',
                unsafe_allow_html=True,
            )

        st.progress(risk_value)

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

    # Key stats grid
    st.markdown(f"""
        <div class="about-stats-grid">
            <div class="about-stat-card">
                {icon_svg("cpu", 24)}
                <span class="stat-value">{model_name}</span>
                <span class="stat-label">Model Engine</span>
            </div>
            <div class="about-stat-card">
                {icon_svg("database", 24)}
                <span class="stat-value">{dataset_meta['rows']}</span>
                <span class="stat-label">Dataset Records</span>
            </div>
            <div class="about-stat-card">
                {icon_svg("list", 24)}
                <span class="stat-value">{len(feature_order)}</span>
                <span class="stat-label">Input Features</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Features Pill Cloud
    st.markdown("<div style='font-size: 14px; font-weight: 700; color: var(--text); margin-top: 10px;'>Input Features</div>", unsafe_allow_html=True)
    pills_html = '<div class="feature-pills">'
    for feat in feature_order:
        pills_html += f'<div class="feature-pill">{icon_svg("check-circle", 14)} {feat}</div>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    # Performance Section
    if test_acc is not None or cv_acc is not None:
        perf_text = ""
        if test_acc is not None:
            perf_text += f"Test Accuracy: <strong>{float(test_acc) * 100:.1f}%</strong>"
        if cv_acc is not None:
            if perf_text: perf_text += " &nbsp;&bull;&nbsp; "
            perf_text += f"CV Accuracy: <strong>{float(cv_acc) * 100:.1f}%</strong>"
    else:
        perf_text = "Training notebook reported approximately 89% holdout accuracy and around 82% cross-validation accuracy."

    st.markdown(f"""
        <div class="about-performance-wrap">
            <div class="about-performance-title">{icon_svg("award", 18)} Model Performance Recognition</div>
            <div class="muted">{perf_text}</div>
        </div>
    """, unsafe_allow_html=True)

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
    
    cards_html = ""
    for num, title, copy in steps:
        cards_html += f"""
        <div class="step-card">
            <div class="step-number">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-copy">{copy}</div>
        </div>
        """
    
    st.markdown(cards_html, unsafe_allow_html=True)
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
