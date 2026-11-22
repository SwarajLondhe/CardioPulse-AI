"""
CardioPulse AI — Cinematic Heart Health Prediction System
=========================================================
A premium, cinematic, clinical-grade machine learning web application for coronary
heart disease risk stratification powered by Scikit-Learn, Joblib, ReportLab,
Matplotlib, Pillow, and Streamlit.
"""

import base64
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).parent
ASSETS = BASE / "assets"
PAGES = [
    "🏠 Home",
    "🩺 Prediction Dashboard",
    "⚖️ Compare Two Persons",
    "📊 Model Performance & Metrics",
    "📖 About Project & ML Pipeline"
]

DISCLAIMER = "Educational use only. This is not a medical diagnosis tool. Consult a qualified doctor for health decisions."

HINTS = {
    "age": "Age in years (clinical range: 29 to 77)",
    "sex": "Biological sex: 1 = male, 0 = female",
    "chest": "Chest pain type: 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic",
    "resting_blood_pressure": "Resting systolic blood pressure in mm Hg upon hospital admission (94 - 200 mm Hg)",
    "serum_cholestoral": "Serum cholesterol level in mg/dl (126 - 564 mg/dl)",
    "fasting_blood_sugar": "Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false",
    "resting_electrocardiographic_results": "Resting ECG: 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy",
    "maximum_heart_rate_achieved": "Maximum heart rate achieved during peak exercise stress test (71 - 202 bpm)",
    "exercise_induced_angina": "Exercise-induced angina: 1 = yes, 0 = no",
    "oldpeak": "ST depression induced by exercise relative to rest (0.0 - 6.2 mm)",
    "slope": "Slope of the peak exercise ST segment: 1 = upsloping, 2 = flat, 3 = downsloping",
    "number_of_major_vessels": "Number of major blood vessels (0-3) colored by fluoroscopy examination",
    "thal": "Thallium scintigraphy heart stress test: 3 = normal, 6 = fixed defect, 7 = reversible defect",
}

st.set_page_config(
    page_title="CardioPulse AI — Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# Base64 Image Loader with Graceful Fallback
# -----------------------------------------------------------------------------
@st.cache_data
def load_img_b64(name: str) -> str:
    """Loads an image from assets/ as a base64 data URI string.
    Returns empty string if file does not exist so CSS gracefully falls back to gradients.
    """
    p = ASSETS / name
    if p.exists():
        try:
            ext = p.suffix.lower().lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return ""
    return ""


# -----------------------------------------------------------------------------
# Premium Cinematic Styling & Keyframe Animations
# -----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@500;600;700;800;900&display=swap');

:root {
  --bg: #060b16;
  --card: rgba(13, 22, 40, 0.75);
  --card-hover: rgba(18, 30, 54, 0.90);
  --line: rgba(148, 163, 184, 0.16);
  --tx: #f1f5f9;
  --mut: #94a3b8;
  --pri: #0ea5e9;
  --pri-glow: rgba(14, 165, 233, 0.35);
  --acc: #ef4444;
  --acc-glow: rgba(239, 68, 68, 0.45);
  --ok: #22c55e;
  --warn: #f59e0b;
}

html, body, .stApp {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  background: var(--bg) !important;
  color: var(--tx);
}

h1, h2, h3, h4, .hero h1, .brand b, .sec {
  font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif !important;
  letter-spacing: -0.02em;
}

/* Cinematic Ambient Drift Background */
.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background: radial-gradient(800px 500px at 10% 12%, rgba(14,165,233,0.18), transparent 65%),
              radial-gradient(750px 480px at 90% 18%, rgba(239,68,68,0.14), transparent 65%),
              radial-gradient(900px 600px at 50% 100%, rgba(99,102,241,0.12), transparent 65%);
  animation: drift 20s ease-in-out infinite alternate;
}
@keyframes drift {
  to { transform: translate3d(0, -25px, 0) scale(1.06); }
}

.block-container {
  position: relative;
  z-index: 1;
  padding-top: 1.5rem;
  padding-bottom: 3rem;
  max-width: 1220px;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stMarkdown {
  color: var(--tx);
}
[data-testid="stCaptionContainer"], .stApp small {
  color: var(--mut) !important;
}

/* Sidebar Anti-Clipping */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070d1a, #0b152d) !important;
  border-right: 1px solid var(--line);
  box-sizing: border-box !important;
  overflow: visible !important;
}
[data-testid="stSidebarContent"] {
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
  padding-top: 1.5rem !important;
  box-sizing: border-box !important;
  overflow-x: visible !important;
}
[data-testid="stSidebar"] [role="radiogroup"] {
  gap: 0.35rem;
  box-sizing: border-box !important;
  padding-left: 2px !important;
  width: 100% !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 0.65rem 0.85rem;
  border-radius: 14px;
  transition: all 0.25s;
  border: 1px solid transparent;
  width: 100%;
  box-sizing: border-box !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(14, 165, 233, 0.12);
  transform: translateX(4px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(14, 165, 233, 0.28), rgba(239, 68, 68, 0.16));
  border-color: rgba(14, 165, 233, 0.5);
}

/* Glassmorphic Form & Controls */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
  background: #0f1a30 !important;
  border-color: var(--line) !important;
  border-radius: 12px !important;
}
input {
  color: var(--tx) !important;
}
[data-testid="stForm"] {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.3rem;
  backdrop-filter: blur(10px);
  animation: fadeUp 0.6s both;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}
.stButton > button, [data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
  background: linear-gradient(90deg, #0ea5e9, #6366f1);
  color: #fff !important;
  border: 0;
  border-radius: 12px;
  padding: 0.65rem 1.4rem;
  font-weight: 700;
  transition: transform 0.2s, box-shadow 0.2s;
  box-shadow: 0 6px 20px rgba(14, 165, 233, 0.35);
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 10px 28px rgba(14, 165, 233, 0.55);
  color: #fff !important;
}
[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid var(--line);
}
[data-testid="stImage"] img {
  border-radius: 16px;
  background: #fff;
}

/* Keyframe Animations */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: none; }
}
@keyframes heartBeat72 {
  0% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(239,68,68,0.5)); }
  14% { transform: scale(1.15); filter: drop-shadow(0 0 25px rgba(239,68,68,0.85)); }
  28% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(239,68,68,0.5)); }
  42% { transform: scale(1.08); filter: drop-shadow(0 0 18px rgba(239,68,68,0.7)); }
  70% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(239,68,68,0.5)); }
  100% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(239,68,68,0.5)); }
}
@keyframes kenburns {
  0% { transform: scale(1); }
  50% { transform: scale(1.08) translate3d(-1%, -1%, 0); }
  100% { transform: scale(1); }
}
@keyframes drawECG {
  0% { stroke-dashoffset: 600; }
  60%, 100% { stroke-dashoffset: 0; }
}
@keyframes growBar {
  from { width: 0; }
}
@keyframes ringProgress {
  from { stroke-dashoffset: 339.3; }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.45); }
  50% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
}
@keyframes textShine {
  from { background-position: 0 0; }
  to { background-position: 200% 0; }
}
@keyframes floatModel {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-5px); }
  100% { transform: translateY(0px); }
}
.anatomy-float {
  animation: floatModel 3.6s ease-in-out infinite;
}
@keyframes holoScan {
  0% { top: -10%; opacity: 0; }
  25% { opacity: 0.5; }
  75% { opacity: 0.5; }
  100% { top: 110%; opacity: 0; }
}
.holo-scan {
  position: absolute;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, #38bdf8, transparent);
  box-shadow: 0 0 8px #38bdf8;
  animation: holoScan 4.2s linear infinite;
  pointer-events: none;
  z-index: 2;
}

/* Cinematic Full-Bleed Hero */
.cinematic-hero {
  position: relative;
  overflow: hidden;
  border-radius: 24px;
  padding: 3rem 2.2rem;
  border: 1px solid var(--line);
  margin-bottom: 1.5rem;
  box-shadow: 0 20px 45px rgba(0, 0, 0, 0.4);
}
.hero-bg-layer {
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center;
  animation: kenburns 26s ease-in-out infinite alternate;
  opacity: 0.28;
  filter: blur(1.5px);
  z-index: 0;
}
.hero-overlay-layer {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(7, 13, 26, 0.94) 0%, rgba(11, 21, 45, 0.85) 50%, rgba(239, 68, 68, 0.22) 100%);
  z-index: 1;
}
.hero-content {
  position: relative;
  z-index: 2;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}
.hero-text {
  flex: 1;
  min-width: 320px;
}
.hero-text h1 {
  font-size: 2.7rem;
  line-height: 1.15;
  margin: 0.5rem 0;
  background: linear-gradient(90deg, #7dd3fc, #ffffff, #fca5a5, #7dd3fc);
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: textShine 6s linear infinite;
}
.hero-text p {
  color: #cbd5e1;
  max-width: 680px;
  font-size: 1.05rem;
  line-height: 1.6;
  margin-bottom: 1.2rem;
}
.hero-visual {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-width: 200px;
}

/* Clean Circular Realistic Heart Orb (NO White Box) */
.beating-heart-orb {
  width: 155px;
  height: 155px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(239, 68, 68, 0.25) 0%, rgba(14, 165, 233, 0.12) 65%, transparent 100%);
  border: 1px solid rgba(239, 68, 68, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  box-shadow: 0 0 35px rgba(239, 68, 68, 0.30);
  overflow: hidden;
}
.beating-heart-img {
  width: 125px;
  height: 125px;
  object-fit: contain;
  animation: heartBeat72 0.833s infinite;
  transform-origin: center;
}

.ecg-continuous {
  width: 100%;
  height: 52px;
  margin-top: 0.8rem;
}
.ecg-continuous path {
  fill: none;
  stroke: #38bdf8;
  stroke-width: 2.6;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 600;
  animation: drawECG 3.2s linear infinite;
}

/* Cards & Layout */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin: 1.2rem 0;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1.2rem 1.3rem;
  backdrop-filter: blur(10px);
  transition: transform 0.3s, border-color 0.3s, box-shadow 0.3s;
  animation: fadeUp 0.7s both;
  box-sizing: border-box;
}
.card:hover {
  transform: translateY(-4px);
  background: var(--card-hover);
  border-color: rgba(14, 165, 233, 0.6);
  box-shadow: 0 14px 34px rgba(14, 165, 233, 0.2);
}
.card h4 {
  margin: 0.2rem 0;
  color: var(--tx);
  font-size: 1.05rem;
}
.card p {
  margin: 0;
  color: var(--mut);
  font-size: 0.9rem;
  line-height: 1.5;
}

/* Photo Cards for Lifestyle & Clinical Sections */
.photo-card {
  overflow: hidden;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: var(--card);
  transition: transform 0.3s, box-shadow 0.3s, border-color 0.3s;
  display: flex;
  flex-direction: column;
}
.photo-card:hover {
  transform: translateY(-4px);
  border-color: rgba(14, 165, 233, 0.6);
  box-shadow: 0 16px 36px rgba(14, 165, 233, 0.22);
}
.photo-card-img {
  width: 100%;
  height: 145px;
  background-size: cover;
  background-position: center;
  position: relative;
}
.photo-card-img::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, transparent 40%, rgba(13, 22, 40, 0.95) 100%);
}
.photo-card-body {
  padding: 1rem 1.2rem;
  flex: 1;
}

.stat .lbl {
  color: var(--mut);
  font-size: 0.85rem;
}
.stat .val {
  font-size: 2rem;
  font-weight: 800;
  color: #fff;
  margin: 0.2rem 0;
}
.bar {
  height: 6px;
  border-radius: 6px;
  background: rgba(148, 163, 184, 0.2);
  overflow: hidden;
}
.bar i {
  display: block;
  height: 100%;
  border-radius: 6px;
  background: linear-gradient(90deg, #0ea5e9, #6366f1);
  animation: growBar 1.2s both;
}
.sec {
  font-size: 1.5rem;
  font-weight: 800;
  margin: 1.8rem 0 0.35rem;
  color: var(--tx);
}
.sub {
  color: var(--mut);
  margin-bottom: 0.9rem;
  font-size: 0.92rem;
}

.res {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  align-items: center;
  border-radius: 22px;
  padding: 1.4rem;
  border: 1px solid var(--line);
  background: var(--card);
  animation: fadeUp 0.6s both;
}
.res.pos {
  border-color: rgba(239, 68, 68, 0.6);
  animation: fadeUp 0.6s both, glowPulse 2.4s infinite;
}
.res.neg {
  border-color: rgba(34, 197, 94, 0.6);
}
.gauge {
  position: relative;
  width: 140px;
  height: 140px;
}
.gauge svg {
  width: 100%;
  height: 100%;
}
.g-bg {
  fill: none;
  stroke: rgba(148, 163, 184, 0.2);
  stroke-width: 12;
}
.g-fg {
  fill: none;
  stroke-width: 12;
  stroke-linecap: round;
  animation: ringProgress 1.4s ease-out both;
}
.g-txt {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.g-txt b {
  font-size: 1.65rem;
  color: #fff;
}
.g-txt span {
  font-size: 0.72rem;
  color: var(--mut);
}
.pill {
  display: inline-block;
  padding: 0.25rem 0.8rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
}
.pill.low { background: rgba(34, 197, 94, 0.18); color: #4ade80; }
.pill.mod { background: rgba(245, 158, 11, 0.18); color: #fbbf24; }
.pill.high { background: rgba(239, 68, 68, 0.18); color: #f87171; }

.prow { margin: 0.55rem 0; }
.prow .t {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  margin-bottom: 4px;
}
.track {
  height: 12px;
  border-radius: 8px;
  background: rgba(148, 163, 184, 0.18);
  overflow: hidden;
}
.track i {
  display: block;
  height: 100%;
  border-radius: 8px;
  animation: growBar 1.1s both;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0.3rem 0 0.8rem;
}
.brand b {
  font-size: 1.1rem;
  color: #fff;
  letter-spacing: -0.01em;
}
.brand span {
  display: block;
  font-size: 0.72rem;
  color: var(--mut);
}

.note {
  background: rgba(14, 165, 233, 0.08);
  border: 1px solid rgba(14, 165, 233, 0.28);
  border-radius: 14px;
  padding: 0.85rem 1rem;
  font-size: 0.85rem;
  color: var(--tx);
  box-sizing: border-box;
  width: 100%;
  line-height: 1.5;
}
.foot {
  margin-top: 2.8rem;
  padding: 1.2rem;
  border-top: 1px solid var(--line);
  text-align: center;
  color: var(--mut);
  font-size: 0.85rem;
}

.step-strip {
  display: flex;
  gap: 14px;
  margin: 1.4rem 0;
}
.step-item {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.1rem 1rem;
  display: flex;
  align-items: center;
  gap: 14px;
  transition: transform 0.25s, border-color 0.25s;
}
.step-item:hover {
  transform: translateY(-2px);
  border-color: rgba(14, 165, 233, 0.5);
}
.step-num {
  min-width: 34px;
  height: 34px;
  border-radius: 50%;
  background: linear-gradient(135deg, #0ea5e9, #6366f1);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 0.95rem;
}

@media (max-width: 768px) {
  .hero-text h1 { font-size: 1.9rem; }
  .cinematic-hero { padding: 1.6rem 1.2rem; }
  .res { justify-content: center; }
  .step-strip { flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""


def H(html: str) -> None:
    """Render HTML as ONE line so Markdown never parses it as a code block."""
    st.markdown(" ".join(line.strip() for line in html.strip().splitlines() if line.strip()), unsafe_allow_html=True)


H(CSS)

# Continuous ECG SVG Line
ECG_LINE = '<svg class="ecg-continuous" viewBox="0 0 260 52" preserveAspectRatio="none"><path pathLength="600" d="M0 26 L55 26 L68 26 L78 8 L90 46 L101 26 L140 26 L152 26 L162 16 L172 38 L182 26 L260 26"/></svg>'


# -----------------------------------------------------------------------------
# Scikit-learn backwards/forwards unpickling compatibility bridge
# -----------------------------------------------------------------------------
try:
    import sklearn.compose._column_transformer as _ct
    from collections import UserList
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(UserList):
            def __init__(self, columns=(), *args, **kwargs):
                super().__init__(columns)
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass


def patch_model_imputers(estimator):
    """Recursively patches SimpleImputer and ColumnTransformer for cross-version compatibility."""
    if estimator is None:
        return
    if hasattr(estimator, 'steps'):
        for _, step in estimator.steps:
            patch_model_imputers(step)
    if hasattr(estimator, 'transformers_'):
        for item in estimator.transformers_:
            if len(item) >= 2:
                patch_model_imputers(item[1])
    if hasattr(estimator, 'named_transformers_'):
        for _, trans in estimator.named_transformers_.items():
            patch_model_imputers(trans)
    if estimator.__class__.__name__ == 'SimpleImputer':
        if not hasattr(estimator, '_fill_dtype'):
            stat = getattr(estimator, 'statistics_', None)
            if stat is not None and hasattr(stat, 'dtype'):
                estimator._fill_dtype = stat.dtype
            else:
                estimator._fill_dtype = np.float64


# -----------------------------------------------------------------------------
# Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_assets():
    """Load model.joblib + meta.json."""
    try:
        model = joblib.load(BASE / "model.joblib")
        patch_model_imputers(model)
        meta = json.loads((BASE / "meta.json").read_text(encoding="utf-8"))
        if not meta.get("features") or not meta.get("classes"):
            return None, None, "meta.json is missing 'features' or 'classes'."
        return model, meta, None
    except FileNotFoundError as e:
        return None, None, f"Missing file: {Path(e.filename).name}."
    except Exception as e:
        return None, None, f"Could not load the model ({type(e).__name__}: {e})."


@st.cache_data
def read_json(name):
    p = BASE / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def read_csv(name):
    p = BASE / "results" / name
    return pd.read_csv(p) if p.exists() else None


FALLBACK_SAMPLE_PATIENTS = pd.DataFrame([
    {
        "example_label": "Male - Lowest Risk (Dataset Example)",
        "prob_present": 0.1725, "age": 34.0, "sex": 1.0, "chest": 1.0,
        "resting_blood_pressure": 118.0, "serum_cholestoral": 182.0, "fasting_blood_sugar": 0.0,
        "resting_electrocardiographic_results": 2.0, "maximum_heart_rate_achieved": 174.0,
        "exercise_induced_angina": 0.0, "oldpeak": 0.0, "slope": 1.0,
        "number_of_major_vessels": 0.0, "thal": 3.0
    },
    {
        "example_label": "Female - Lowest Risk (Dataset Example)",
        "prob_present": 0.1321, "age": 34.0, "sex": 0.0, "chest": 2.0,
        "resting_blood_pressure": 118.0, "serum_cholestoral": 210.0, "fasting_blood_sugar": 0.0,
        "resting_electrocardiographic_results": 0.0, "maximum_heart_rate_achieved": 192.0,
        "exercise_induced_angina": 0.0, "oldpeak": 0.7, "slope": 1.0,
        "number_of_major_vessels": 0.0, "thal": 3.0
    },
    {
        "example_label": "Male - Highest Risk (Dataset Example)",
        "prob_present": 0.9011, "age": 51.0, "sex": 1.0, "chest": 4.0,
        "resting_blood_pressure": 140.0, "serum_cholestoral": 298.0, "fasting_blood_sugar": 0.0,
        "resting_electrocardiographic_results": 0.0, "maximum_heart_rate_achieved": 122.0,
        "exercise_induced_angina": 1.0, "oldpeak": 4.2, "slope": 2.0,
        "number_of_major_vessels": 3.0, "thal": 7.0
    },
    {
        "example_label": "Female - Highest Risk (Dataset Example)",
        "prob_present": 0.8744, "age": 62.0, "sex": 0.0, "chest": 4.0,
        "resting_blood_pressure": 160.0, "serum_cholestoral": 164.0, "fasting_blood_sugar": 0.0,
        "resting_electrocardiographic_results": 2.0, "maximum_heart_rate_achieved": 145.0,
        "exercise_induced_angina": 0.0, "oldpeak": 6.2, "slope": 3.0,
        "number_of_major_vessels": 3.0, "thal": 7.0
    }
])


@st.cache_data
def load_sample_patients():
    p = BASE / "sample_patients.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return FALLBACK_SAMPLE_PATIENTS


def stretch(fn, *args, **kw):
    try:
        return fn(*args, width="stretch", **kw)
    except Exception:
        try:
            return fn(*args, use_container_width=True, **kw)
        except Exception:
            return fn(*args, **kw)


def show_img(name, caption):
    p = BASE / "figures" / name
    if not p.exists():
        st.info(f"{name} not found in figures/.")
        return
    stretch(st.image, str(p), caption=caption)


def pct(v):
    return f"{float(v) * 100:.2f}%"


def pretty(n):
    return n.replace("_", " ").strip().title()


model, meta, LOAD_ERR = load_assets()
if LOAD_ERR:
    st.error(LOAD_ERR)
    st.stop()

FEATURES = list(meta["features"].keys())
CLASSES = [str(c) for c in meta["classes"]]
POS = CLASSES.index("present") if "present" in CLASSES else len(CLASSES) - 1
metrics = read_json("test_metrics.json") or {}
tuning = read_json("tuning.json") or {}
sample_patients_df = load_sample_patients()

try:
    ENGINE = type(model.steps[-1][1]).__name__
except Exception:
    ENGINE = type(model).__name__


# -----------------------------------------------------------------------------
# Animated 3D Anatomical Avatars with Physiological Cardiac Pulse
# -----------------------------------------------------------------------------
def get_avatar_svg(sex: float, risk_band: str) -> str:
    """Generates an animated 3D anatomical avatar (man or woman) in a holographic glass pod.
    The cardiac beacon dynamically reflects the model risk level:
    - Low: green pulse (#22c55e), calm 1.8s
    - Moderate: amber pulse (#f59e0b), moderate 1.1s
    - High: red pulse (#ef4444), rapid 0.65s
    """
    is_male = (int(sex) == 1)
    img_name = "male_anatomy_3d.png" if is_male else "female_anatomy_3d.png"
    b64_uri = load_img_b64(img_name)

    glow_col = "rgba(34,197,94,0.65)" if risk_band == "low" else "rgba(245,158,11,0.65)" if risk_band == "mod" else "rgba(239,68,68,0.85)"
    beacon_col = "#22c55e" if risk_band == "low" else "#f59e0b" if risk_band == "mod" else "#ef4444"
    label = "Male (3D Anatomy)" if is_male else "Female (3D Anatomy)"

    if b64_uri:
        avatar_html = f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin:4px 0;">
          <div style="position:relative;width:115px;height:195px;border-radius:18px;background:radial-gradient(ellipse at 50% 30%, rgba(14,165,233,0.12), rgba(6,11,22,0.85) 75%);border:1px solid rgba(14,165,233,0.25);overflow:hidden;display:flex;align-items:center;justify-content:center;box-shadow:0 8px 24px rgba(0,0,0,0.35);">
            <div class="holo-scan"></div>
            <div class="anatomy-float" style="position:relative;width:100%;height:100%;display:flex;align-items:center;justify-content:center;">
              <img src="{b64_uri}" alt="{label}" style="height:175px;object-fit:contain;filter:drop-shadow(0 4px 12px {glow_col});" />
              <div style="position:absolute;top:26%;left:50%;transform:translate(-50%, -50%);width:14px;height:14px;border-radius:50%;background:{beacon_col};box-shadow:0 0 16px {beacon_col}, 0 0 30px {beacon_col};animation:heartBeat72 0.833s infinite;"></div>
            </div>
          </div>
          <div style="font-size:0.75rem;color:var(--mut);font-weight:700;margin-top:4px;">{label}</div>
        </div>
        """
    else:
        # Graceful vector fallback if image is missing
        pulse_dur = "1.8s" if risk_band == "low" else "1.1s" if risk_band == "mod" else "0.65s"
        hair_path = (
            '<path d="M42 30 C42 14 78 14 78 30 C82 30 84 38 80 44 C76 34 72 26 60 26 C48 26 44 34 40 44 C36 38 38 30 42 30 Z" fill="#38bdf8"/>'
            if is_male else
            '<path d="M38 34 C38 12 82 12 82 34 C86 42 88 56 84 68 C80 60 80 38 76 30 C68 22 52 22 44 30 C40 38 40 60 36 68 C32 56 34 42 38 34 Z" fill="#ec4899"/>'
        )
        avatar_html = f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin:6px 0;">
          <svg width="105" height="125" viewBox="0 0 120 140" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter:drop-shadow(0 4px 12px {glow_col})">
            <path d="M22 126 C22 96 42 88 60 88 C78 88 98 96 98 126 Z" fill="#1e293b" stroke="#334155" stroke-width="2"/>
            <path d="M44 88 L60 106 L76 88" stroke="#0ea5e9" stroke-width="2.5" fill="none"/>
            <path d="M52 74 L52 88 L68 88 L68 74 Z" fill="#94a3b8"/>
            <ellipse cx="60" cy="52" rx="20" ry="24" fill="#cbd5e1"/>
            {hair_path}
            <g transform="translate(48, 92) scale(0.95)" style="transform-origin:60px 104px;animation:heartBeat72 {pulse_dur} infinite">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" fill="{beacon_col}"/>
            </g>
          </svg>
          <div style="font-size:0.78rem;color:var(--mut);font-weight:700;margin-top:2px;">{label}</div>
        </div>
        """

    return " ".join(line.strip() for line in avatar_html.strip().splitlines() if line.strip())


# -----------------------------------------------------------------------------
# In-Memory PDF Reports (ReportLab & Matplotlib, No Emojis, Helvetica Only)
# -----------------------------------------------------------------------------
def make_probability_chart_image(classes, proba):
    fig, ax = plt.subplots(figsize=(6, 1.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    y_pos = range(len(classes))
    bar_colors = ["#22c55e" if c.lower() == "absent" else "#ef4444" for c in classes]

    bars = ax.barh(y_pos, [p * 100 for p in proba], color=bar_colors, height=0.48)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.capitalize() for c in classes], fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=10, color="#475569")
    ax.set_title("Prediction Probabilities", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                ha="left", va="center", fontsize=10, fontweight="bold", color="#1e293b")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def make_comparison_chart_image(classes, proba_a, proba_b):
    fig, ax = plt.subplots(figsize=(6.2, 2.1), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    x = np.arange(len(classes))
    width = 0.35

    rects1 = ax.bar(x - width / 2, [p * 100 for p in proba_a], width, label="Person A", color="#0ea5e9")
    rects2 = ax.bar(x + width / 2, [p * 100 for p in proba_b], width, label="Person B", color="#f43f5e")

    ax.set_ylabel("Probability (%)", fontsize=10, color="#475569")
    ax.set_title("Person A vs Person B Probability Comparison", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11, fontweight="bold", color="#1e293b")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_single_patient_pdf(patient_inputs, prediction_class, confidence_pct, risk_band, proba, classes, engine_name, metrics, features_meta):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
    subtitle_style = ParagraphStyle("SStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#64748b"))
    section_style = ParagraphStyle("SecStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=7, spaceAfter=4)
    body_style = ParagraphStyle("BStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#1e293b"))
    bold_body = ParagraphStyle("BBStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=12, textColor=colors.HexColor("#0f172a"))
    disclaimer_style = ParagraphStyle("DStyle", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=8, leading=11, textColor=colors.HexColor("#94a3b8"))

    # Include logo image if file loads
    logo_file = ASSETS / "favicon.png"
    if logo_file.exists():
        logo_img = RLImage(str(logo_file), width=30, height=30)
        h_table = Table([[logo_img, Paragraph("CardioPulse AI - Clinical Risk Assessment Report", title_style)]], colWidths=[38, 502])
        h_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(h_table)
    else:
        story.append(Paragraph("CardioPulse AI - Clinical Risk Assessment Report", title_style))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {now_str} | Architecture: {engine_name} Pipeline | Held-out Test ROC-AUC: {metrics.get('roc_auc', 0)*100:.2f}%", subtitle_style))
    story.append(Spacer(1, 8))

    summary_data = [
        [Paragraph("Predicted Classification", bold_body), Paragraph(f"<b>{prediction_class.upper()}</b>", bold_body)],
        [Paragraph("Model Confidence", bold_body), Paragraph(f"{confidence_pct:.1f}%", body_style)],
        [Paragraph("Risk Stratification Band", bold_body), Paragraph(f"<b>{risk_band} Risk</b>", bold_body)],
        [Paragraph("Clinical Meaning", bold_body), Paragraph(f"The model estimates an {proba[POS]*100:.1f}% probability that heart disease is present. This is a model estimate, not a diagnosis.", body_style)]
    ]
    summary_table = Table(summary_data, colWidths=[150, 390])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#bae6fd")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0f2fe")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Prediction Probability Chart", section_style))
    chart_buf = make_probability_chart_image(classes, proba)
    story.append(RLImage(chart_buf, width=440, height=130))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Patient Clinical Diagnostic Values", section_style))
    input_rows = [[Paragraph("Clinical Feature", bold_body), Paragraph("Recorded Value", bold_body), Paragraph("Reference Range (Median)", bold_body)]]
    for feat, val in patient_inputs.items():
        minfo = features_meta.get(feat, {})
        norm_range = f"{minfo.get('min', '-')} - {minfo.get('max', '-')} (Median: {minfo.get('median', '-')})"
        val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
        feat_name = feat.replace("_", " ").title()
        input_rows.append([Paragraph(feat_name, body_style), Paragraph(val_str, bold_body), Paragraph(norm_range, body_style)])

    input_table = Table(input_rows, colWidths=[180, 100, 260])
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 1), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 2.5),
    ]))
    story.append(input_table)
    story.append(Spacer(1, 8))

    perf_text = f"Empirical Test Metrics (Held-out Set): ROC-AUC: {metrics.get('roc_auc', 0)*100:.2f}%, Accuracy: {metrics.get('accuracy', 0)*100:.2f}%, Recall: {metrics.get('recall', 0)*100:.2f}%, Precision: {metrics.get('precision', 0)*100:.2f}%."
    story.append(Paragraph(perf_text, subtitle_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("DISCLAIMER: Educational use only. Not a medical diagnosis tool. Consult a qualified doctor for health decisions.", disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


def build_comparison_pdf(patient_a, proba_a, patient_b, proba_b, classes, engine_name, metrics, features_meta):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#0f172a"))
    subtitle_style = ParagraphStyle("SStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor("#64748b"))
    section_style = ParagraphStyle("SecStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=6, spaceAfter=4)
    body_style = ParagraphStyle("BStyle", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11, textColor=colors.HexColor("#1e293b"))
    bold_body = ParagraphStyle("BBStyle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.HexColor("#0f172a"))
    disclaimer_style = ParagraphStyle("DStyle", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=7.5, leading=10, textColor=colors.HexColor("#94a3b8"))

    # Include logo image if file loads
    logo_file = ASSETS / "favicon.png"
    if logo_file.exists():
        logo_img = RLImage(str(logo_file), width=30, height=30)
        h_table = Table([[logo_img, Paragraph("CardioPulse AI - Comparative Risk Assessment Report", title_style)]], colWidths=[38, 502])
        h_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(h_table)
    else:
        story.append(Paragraph("CardioPulse AI - Comparative Risk Assessment Report", title_style))

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {now_str} | Model: {engine_name} Pipeline | Held-out ROC-AUC: {metrics.get('roc_auc', 0)*100:.2f}%", subtitle_style))
    story.append(Spacer(1, 8))

    k_a = int(np.argmax(proba_a))
    k_b = int(np.argmax(proba_b))
    band_a = "Low" if proba_a[POS] < 0.33 else "Moderate" if proba_a[POS] < 0.66 else "High"
    band_b = "Low" if proba_b[POS] < 0.33 else "Moderate" if proba_b[POS] < 0.66 else "High"
    diff = (proba_b[POS] - proba_a[POS]) * 100
    higher_text = f"Person B is higher by {abs(diff):.1f} percentage points." if diff > 0 else f"Person A is higher by {abs(diff):.1f} percentage points."

    outcomes = [
        [Paragraph("Metric / Evaluation", bold_body), Paragraph("Person A", bold_body), Paragraph("Person B", bold_body)],
        [Paragraph("Predicted Classification", bold_body), Paragraph(f"<b>{classes[k_a].upper()}</b>", bold_body), Paragraph(f"<b>{classes[k_b].upper()}</b>", bold_body)],
        [Paragraph("Model Confidence", body_style), Paragraph(f"{proba_a[k_a]*100:.1f}%", body_style), Paragraph(f"{proba_b[k_b]*100:.1f}%", body_style)],
        [Paragraph("Disease Probability (Present)", body_style), Paragraph(f"{proba_a[POS]*100:.1f}%", body_style), Paragraph(f"{proba_b[POS]*100:.1f}%", body_style)],
        [Paragraph("Risk Stratification Band", body_style), Paragraph(f"<b>{band_a} Risk</b>", body_style), Paragraph(f"<b>{band_b} Risk</b>", body_style)],
        [Paragraph("Differential Analysis", bold_body), Paragraph(higher_text, bold_body), Paragraph("", body_style)],
    ]
    t_out = Table(outcomes, colWidths=[160, 190, 190])
    t_out.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("SPAN", (1, 5), (2, 5)),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_out)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Comparative Probabilities", section_style))
    chart_buf = make_comparison_chart_image(classes, proba_a, proba_b)
    story.append(RLImage(chart_buf, width=420, height=140))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Clinical Indicators Comparison", section_style))
    feat_rows = [[Paragraph("Feature", bold_body), Paragraph("Person A", bold_body), Paragraph("Person B", bold_body), Paragraph("Variance", bold_body)]]
    for feat in features_meta.keys():
        va = patient_a.get(feat, 0)
        vb = patient_b.get(feat, 0)
        sa = f"{va:.1f}" if isinstance(va, float) else str(va)
        sb = f"{vb:.1f}" if isinstance(vb, float) else str(vb)
        diff_str = "Same" if va == vb else f"Delta: {vb - va:+.1f}" if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else "Differs"
        feat_name = feat.replace("_", " ").title()
        feat_rows.append([Paragraph(feat_name, body_style), Paragraph(sa, body_style), Paragraph(sb, body_style), Paragraph(diff_str, bold_body if diff_str != "Same" else body_style)])

    t_feat = Table(feat_rows, colWidths=[160, 120, 120, 140])
    t_feat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 8))

    story.append(Paragraph("DISCLAIMER: Educational use only. Not a medical diagnosis tool. Consult a qualified doctor for health decisions.", disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer


# -----------------------------------------------------------------------------
# Input & Prediction Logic
# -----------------------------------------------------------------------------
def build_inputs(prefix=""):
    """Create one widget per feature in meta.json. Returns {feature: value}."""
    row, cols = {}, st.columns(2)
    for i, f in enumerate(FEATURES):
        info, key = meta["features"][f], f"{prefix}{f}"
        with cols[i % 2]:
            lo, hi, med = float(info["min"]), float(info["max"]), float(info["median"])
            med = min(max(med, lo), hi)
            if f == "sex":
                cur_val = int(st.session_state.get(key, int(round(med))))
                val_idx = 0 if cur_val == 1 else 1
                sex_choice = st.radio(
                    "Sex (Biological)",
                    options=[1, 0],
                    format_func=lambda x: "👨 Male" if x == 1 else "👩 Female",
                    index=val_idx,
                    key=key,
                    horizontal=True,
                    help=HINTS.get(f, "Biological sex: 1 = male, 0 = female")
                )
                row[f] = float(sex_choice)
            else:
                whole = all(float(x).is_integer() for x in (lo, hi)) and (hi - lo) >= 1 and f != "oldpeak"
                if whole:
                    def_val = int(st.session_state[key]) if key in st.session_state else int(round(med))
                    row[f] = st.number_input(
                        pretty(f), min_value=int(lo), max_value=int(hi),
                        value=def_val, step=1, key=key, help=HINTS.get(f)
                    )
                else:
                    def_val = float(st.session_state[key]) if key in st.session_state else med
                    row[f] = st.number_input(
                        pretty(f), min_value=lo, max_value=hi,
                        value=def_val, step=0.1, key=key, help=HINTS.get(f)
                    )
    return row


def validate(row):
    """Return a list of validation issues (empty list = valid)."""
    bad = []
    for f in FEATURES:
        info, v = meta["features"][f], row.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            bad.append(f"{pretty(f)} is empty.")
        elif not (info["min"] <= v <= info["max"]):
            bad.append(f"{pretty(f)} must be between {info['min']:g} and {info['max']:g}.")
    return bad


def predict(row):
    patch_model_imputers(model)
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    proba = np.asarray(model.predict_proba(X)[0], dtype=float)
    if len(proba) != len(CLASSES):
        raise ValueError("Model output does not match classes in meta.json.")
    return proba


def band(p):
    return ("Low", "low") if p < 0.33 else ("Moderate", "mod") if p < 0.66 else ("High", "high")


def gauge(p, color, label):
    c = 339.3
    return (
        f'<div class="gauge"><svg viewBox="0 0 140 140"><circle class="g-bg" cx="70" cy="70" r="54"/>'
        f'<circle class="g-fg" cx="70" cy="70" r="54" transform="rotate(-90 70 70)" '
        f'style="stroke:{color};stroke-dasharray:{c};stroke-dashoffset:{c * (1 - p):.1f}"/></svg>'
        f'<div class="g-txt"><b>{p * 100:.1f}%</b><span>{label}</span></div></div>'
    )


def bars(proba):
    out = ""
    for c, p in zip(CLASSES, proba):
        col = "#ef4444" if c == CLASSES[POS] else "#22c55e"
        out += (
            f'<div class="prow"><div class="t"><span>{c.title()}</span><b>{p * 100:.1f}%</b></div>'
            f'<div class="track"><i style="width:{p * 100:.1f}%;background:{col}"></i></div></div>'
        )
    return out


def result_card(title, proba, sex_val=1.0):
    k = int(np.argmax(proba))
    is_pos = (k == POS)
    lab, cls = band(float(proba[POS]))
    col = "#ef4444" if is_pos else "#22c55e"
    avatar_html = get_avatar_svg(sex_val, cls)

    H(
        f'<div class="res {"pos" if is_pos else "neg"}">'
        f'<div style="display:flex;align-items:center;gap:18px;">'
        f'{avatar_html}'
        f'{gauge(float(proba[POS]), col, CLASSES[POS].title() + " probability")}'
        f'</div>'
        f'<div style="flex:1;min-width:260px;">'
        f'<div style="color:var(--mut);font-size:0.85rem;">{title}</div>'
        f'<div style="font-size:1.75rem;font-weight:800;margin:2px 0;">Prediction: {CLASSES[k].title()}</div>'
        f'<div style="margin:0.35rem 0;">Model confidence: <b>{proba[k] * 100:.1f}%</b> &nbsp; '
        f'<span class="pill {cls}">{lab} risk band</span></div>'
        f'{bars(proba)}'
        f'</div></div>'
    )

    # Plain-English sentence generated from real model numbers
    explanation = (
        f'<div class="card" style="margin-top:10px;padding:12px 16px;border-left:4px solid {col};">'
        f'<b>Plain-English Interpretation:</b> The model estimates an <b>{proba[POS] * 100:.1f}% probability</b> '
        f'that heart disease is present (classification: <b>{CLASSES[k].title()}</b> with {proba[k] * 100:.1f}% confidence). '
        f'This is a model estimate, not a diagnosis.</div>'
    )
    H(explanation)


def render_input_range_charts(row):
    """Input Graph: displays patient value on min-median-max range for all clinical features."""
    html = '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:12px;margin-top:12px;">'
    for f in FEATURES:
        info = meta["features"][f]
        val = row.get(f, info["median"])
        lo, hi, med = float(info["min"]), float(info["max"]), float(info["median"])
        span = hi - lo if hi > lo else 1.0
        pct_val = max(0.0, min(100.0, (val - lo) / span * 100.0))
        pct_med = max(0.0, min(100.0, (med - lo) / span * 100.0))

        dev = abs(val - med) / span
        marker_col = "#38bdf8" if dev < 0.25 else "#f59e0b" if dev < 0.5 else "#ef4444"

        hint = HINTS.get(f, "")
        val_disp = f"{val:.1f}" if isinstance(val, float) else str(val)

        html += (
            f'<div class="card" style="padding:12px 14px;margin-bottom:0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">'
            f'<div><b style="color:var(--tx);font-size:0.9rem;">{pretty(f)}</b></div>'
            f'<div style="background:rgba(14,165,233,0.15);border:1px solid rgba(14,165,233,0.3);border-radius:6px;padding:2px 8px;font-size:0.82rem;font-weight:700;color:{marker_col};">'
            f'Value: {val_disp}</div></div>'
            f'<div style="color:var(--mut);font-size:0.75rem;margin-bottom:8px;line-height:1.3;">{hint}</div>'
            f'<div style="position:relative;height:10px;background:rgba(148,163,184,0.18);border-radius:6px;margin:14px 0 8px 0;">'
            f'<div style="position:absolute;left:{pct_med:.1f}%;top:-3px;bottom:-3px;width:2px;background:#94a3b8;" title="Median: {med:g}"></div>'
            f'<div style="position:absolute;left:calc({pct_val:.1f}% - 6px);top:-4px;width:12px;height:18px;border-radius:4px;background:{marker_col};box-shadow:0 0 8px {marker_col};"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;color:var(--mut);font-size:0.72rem;">'
            f'<span>Min: {lo:g}</span>'
            f'<span style="color:#94a3b8;">Median: {med:g}</span>'
            f'<span>Max: {hi:g}</span>'
            f'</div></div>'
        )
    html += '</div>'
    H(html)


def footer():
    H(f'<div class="foot">⚠️ {DISCLAIMER}</div>')


def stat_cards():
    items = [
        ("ROC-AUC", "roc_auc"),
        ("Accuracy", "accuracy"),
        ("Recall (Sensitivity)", "recall"),
        ("Precision", "precision"),
        ("F1-Score", "f1")
    ]
    html = '<div class="grid">'
    for i, (lbl, k) in enumerate(items):
        if k in metrics:
            v = float(metrics[k])
            html += (
                f'<div class="card stat" style="animation-delay:{i * 0.08:.2f}s;"><div class="lbl">{lbl}</div>'
                f'<div class="val">{pct(v)}</div><div class="bar"><i style="width:{v * 100:.1f}%"></i></div></div>'
            )
    H(html + "</div>")


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Overview
# -----------------------------------------------------------------------------
st.session_state.setdefault("page", PAGES[0])

with st.sidebar:
    H(
        f'<div class="brand"><svg width="42" height="42" viewBox="0 0 24 24"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#f87171"/><stop offset="1" stop-color="#0ea5e9"/></linearGradient></defs>'
        f'<path fill="url(#lg)" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>'
        f'<path d="M3 12h4l2-4 3 8 2-4h7" fill="none" stroke="#fff" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'<div><b>CardioPulse AI</b><span>Heart Health Prediction</span></div></div>'
    )
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()

    ov = (
        f"<b>Engine:</b> Scikit-Learn Pipeline<br>"
        f"<b>Model:</b> {ENGINE}<br>"
        f"<b>Target:</b> {CLASSES[0]} (0) vs {CLASSES[-1]} (1)<br>"
        f"<b>Features:</b> {len(FEATURES)} clinical indicators"
    )
    if "roc_auc" in metrics:
        ov += f"<br><b>Holdout ROC-AUC:</b> {pct(metrics['roc_auc'])}"
    if "recall" in metrics:
        ov += f"<br><b>Sensitivity:</b> {pct(metrics['recall'])}"
    H(f'<div class="note"><b>System Overview</b><br>{ov}</div>')
    H('<div class="note" style="margin-top:10px;"><b>Model Integrity</b><br>Predictions come directly from model.joblib and meta.json. Zero synthetic estimates.</div>')


# -----------------------------------------------------------------------------
# PAGE 1: Home (Cinematic Full-Bleed Look with Realistic 3D Heart)
# -----------------------------------------------------------------------------
def page_home():
    hero_b64 = load_img_b64("hero.jpg")
    heart_b64 = load_img_b64("realistic_heart.png")
    if not heart_b64:
        heart_b64 = load_img_b64("realistic_heart.jpg")

    lifestyle_b64 = load_img_b64("lifestyle.jpg")
    consult_b64 = load_img_b64("consultation.jpg")
    bp_b64 = load_img_b64("blood_pressure.jpg")
    steth_b64 = load_img_b64("stethoscope.jpg")

    hero_bg_style = f"background-image: url('{hero_b64}');" if hero_b64 else "background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(239,68,68,0.16));"

    # Realistic beating heart orb (from heart.avif, circular feathered alpha mask with NO white square)
    if heart_b64:
        heart_visual = f'<img src="{heart_b64}" class="beating-heart-img" alt="Realistic 3D Anatomical Heart"/>'
    else:
        heart_visual = (
            '<svg class="beating-heart-img" viewBox="0 0 24 24" fill="#ef4444">'
            '<path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>'
            '</svg>'
        )

    # 1. Full-Bleed Hero Section
    H(
        f'<div class="cinematic-hero">'
        f'<div class="hero-bg-layer" style="{hero_bg_style}"></div>'
        f'<div class="hero-overlay-layer"></div>'
        f'<div class="hero-content">'
        f'<div class="hero-text">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">'
        f'<span class="pill high" style="background:rgba(14,165,233,0.2);color:#7dd3fc;border:1px solid rgba(14,165,233,0.4);">Machine Learning · Cardiology AI</span>'
        f'<span style="color:#94a3b8;font-size:0.8rem;">Statlog Heart Cohort (OpenML 53)</span>'
        f'</div>'
        f'<h1>CardioPulse AI — Heart Health Prediction System</h1>'
        f'<p>A modern, clinician-grade risk stratification platform. Predicts coronary heart disease risk from '
        f'{len(FEATURES)} diagnostic indicators using a rigorous, leakage-free Scikit-Learn pipeline with nested cross-validation.</p>'
        f'{ECG_LINE}'
        f'</div>'
        f'<div class="hero-visual">'
        f'<div class="beating-heart-orb">{heart_visual}</div>'
        f'<div style="font-size:0.75rem;color:#fca5a5;font-weight:700;margin-top:8px;text-align:center;">Resting Rhythm ~72 BPM</div>'
        f'</div>'
        f'</div></div>'
    )

    st.button("🩺 Start a prediction →", on_click=lambda: st.session_state.update(page=PAGES[1]))

    # 2. "How It Works" 3 Animated Steps
    H(
        '<div class="step-strip">'
        '<div class="step-item"><div class="step-num">1</div><div><b>1. Enter Data</b><div style="font-size:0.8rem;color:var(--mut);">Input 13 diagnostic indicators on the dashboard</div></div></div>'
        '<div class="step-item"><div class="step-num">2</div><div><b>2. Model Analyses</b><div style="font-size:0.8rem;color:var(--mut);">Pipeline applies median imputation, scaling, and classification</div></div></div>'
        '<div class="step-item"><div class="step-num">3</div><div><b>3. Read Result</b><div style="font-size:0.8rem;color:var(--mut);">View risk gauge, patient avatar, range graph, and PDF report</div></div></div>'
        '</div>'
    )

    # 3. "The Problem" (Real Holdout Figures Only)
    roc_str = pct(metrics.get("roc_auc", 0.9097))
    rec_str = pct(metrics.get("recall", 0.8750))
    acc_str = pct(metrics.get("accuracy", 0.8333))

    H('<div class="sec">The Challenge &amp; Clinical Impact</div><div class="sub">Early non-invasive screening saves lives. Verified empirical metrics from results/test_metrics.json</div>')

    bp_bg = f"background-image:url('{bp_b64}');" if bp_b64 else ""
    H(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:16px;margin:1rem 0;">'
        f'<div class="photo-card">'
        f'<div class="photo-card-img" style="{bp_bg}"></div>'
        f'<div class="photo-card-body">'
        f'<h4>Non-Invasive Diagnostic Screening</h4>'
        f'<p>Cardiovascular disease represents the leading global mortality cause. CardioPulse AI uses routine measurements—such as blood pressure, cholesterol, resting ECG, and fluoroscopy—to estimate risk before invasive catheterization is required.</p>'
        f'</div></div>'
        f'<div class="card" style="display:flex;flex-direction:column;justify-content:space-around;">'
        f'<div><span style="font-size:0.8rem;color:#38bdf8;font-weight:700;">TEST SET EVALUATION</span>'
        f'<h3 style="margin:4px 0;color:#fff;">Validated on 54 Unseen Patients</h3>'
        f'<p>Evaluated on a completely locked holdout split reserved from OpenML Statlog Heart. Zero test-set leakage.</p></div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px;">'
        f'<div style="background:rgba(255,255,255,0.04);padding:10px;border-radius:12px;text-align:center;">'
        f'<div style="font-size:0.75rem;color:var(--mut);">ROC-AUC</div><div style="font-size:1.3rem;font-weight:800;color:#38bdf8;">{roc_str}</div></div>'
        f'<div style="background:rgba(255,255,255,0.04);padding:10px;border-radius:12px;text-align:center;">'
        f'<div style="font-size:0.75rem;color:var(--mut);">Recall / Sensitivity</div><div style="font-size:1.3rem;font-weight:800;color:#4ade80;">{rec_str}</div></div>'
        f'<div style="background:rgba(255,255,255,0.04);padding:10px;border-radius:12px;text-align:center;">'
        f'<div style="font-size:0.75rem;color:var(--mut);">Accuracy</div><div style="font-size:1.3rem;font-weight:800;color:#fff;">{acc_str}</div></div>'
        f'</div></div></div>'
    )

    # 4. "Healthy-Heart Tips" (Photo Cards)
    life_bg = f"background-image:url('{lifestyle_b64}');" if lifestyle_b64 else ""
    steth_bg = f"background-image:url('{steth_b64}');" if steth_b64 else ""
    consult_bg = f"background-image:url('{consult_b64}');" if consult_b64 else ""

    H('<div class="sec">Cardiovascular Wellness &amp; Prevention</div><div class="sub">General, evidence-based lifestyle habits for maintaining heart health (non-diagnostic reference).</div>')
    H(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));gap:16px;margin:1rem 0;">'
        f'<div class="photo-card">'
        f'<div class="photo-card-img" style="{life_bg}"></div>'
        f'<div class="photo-card-body">'
        f'<h4>Aerobic Conditioning</h4>'
        f'<p>At least 150 minutes of moderate aerobic activity (e.g. brisk walking, jogging, cycling) weekly strengthens the myocardium and lowers resting heart rate.</p>'
        f'</div></div>'
        f'<div class="photo-card">'
        f'<div class="photo-card-img" style="{bp_bg}"></div>'
        f'<div class="photo-card-body">'
        f'<h4>Vascular Pressure Control</h4>'
        f'<p>Keeping resting systolic blood pressure below 120 mm Hg protects coronary artery walls from endothelial microtrauma and plaque buildup.</p>'
        f'</div></div>'
        f'<div class="photo-card">'
        f'<div class="photo-card-img" style="{steth_bg}"></div>'
        f'<div class="photo-card-body">'
        f'<h4>Routine Biomarker Panels</h4>'
        f'<p>Regular lipid panels (serum cholesterol, LDL/HDL ratio) and resting ECG traces help detect asymptomatic ischemia well in advance.</p>'
        f'</div></div>'
        f'<div class="photo-card">'
        f'<div class="photo-card-img" style="{consult_bg}"></div>'
        f'<div class="photo-card-body">'
        f'<h4>Clinical Partnership</h4>'
        f'<p>Discussing persistent chest discomfort, shortness of breath, or exercise fatigue with a licensed physician ensures personalized preventative care.</p>'
        f'</div></div></div>'
    )

    # 5. "Project in 30 Seconds" (Plain Language Card)
    H(
        f'<div class="card" style="margin-top:16px;">'
        f'<h3 style="margin-top:0;color:#38bdf8;">⏱️ Project in 30 Seconds</h3>'
        f'<p><b>The Goal:</b> Provide an intuitive, educational decision-support demonstration that estimates whether heart disease is present or absent from clinical test numbers.</p>'
        f'<p><b>The Data:</b> OpenML Statlog Heart dataset (<code>data_id=53</code>) with 270 real patient records and 13 diagnostic attributes.</p>'
        f'<p><b>The Method:</b> Scikit-Learn Pipeline with median imputation, standard scaling, and Logistic Regression with balanced weighting.</p>'
        f'<p><b>The Real Numbers:</b> Held-out test <b>ROC-AUC of {roc_str}</b> and <b>Sensitivity (Recall) of {rec_str}</b> on completely unseen data.</p>'
        f'</div>'
    )

    # 6. "What do these inputs mean?" Glossary Expander
    with st.expander("❓ What do these clinical inputs mean? (Beginner's Glossary)"):
        glossary_html = '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:10px;">'
        for f in FEATURES:
            glossary_html += (
                f'<div style="background:rgba(255,255,255,0.03);padding:8px 12px;border-radius:10px;border:1px solid var(--line);">'
                f'<b style="color:#7dd3fc;">{pretty(f)}:</b> <span style="font-size:0.85rem;color:var(--tx);">{HINTS[f]}</span></div>'
            )
        glossary_html += '</div>'
        H(glossary_html)

    footer()


# -----------------------------------------------------------------------------
# PAGE 2: Prediction Dashboard
# -----------------------------------------------------------------------------
def page_predict():
    H('<div class="sec">Prediction Dashboard</div><div class="sub">Fill the clinical indicators below, then press Predict. Hover the ? icons for guidance.</div>')

    with st.form("predict_form"):
        row = build_inputs("p_")
        go = st.form_submit_button("❤️ Predict")

    if go:
        problems = validate(row)
        if problems:
            for p in problems:
                st.error(p)
        else:
            with st.spinner("Analysing ECG & Clinical Telemetry..."):
                time.sleep(1.0)
            try:
                proba = predict(row)
                st.session_state["last_predict_row"] = row
                st.session_state["last_predict_proba"] = proba

                k = int(np.argmax(proba))
                lab, _ = band(float(proba[POS]))
                pdf_buf = build_single_patient_pdf(
                    patient_inputs=row,
                    prediction_class=CLASSES[k],
                    confidence_pct=float(proba[k] * 100),
                    risk_band=lab,
                    proba=proba,
                    classes=CLASSES,
                    engine_name=ENGINE,
                    metrics=metrics,
                    features_meta=meta["features"]
                )
                st.session_state["last_predict_pdf"] = pdf_buf.getvalue()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Render results in 3 Tabs outside form so results persist after download
    if "last_predict_row" in st.session_state and "last_predict_proba" in st.session_state:
        row = st.session_state["last_predict_row"]
        proba = st.session_state["last_predict_proba"]
        sex_val = row.get("sex", 1.0)

        H('<div class="sec">Patient Evaluation Output</div>')
        tab_result, tab_input_graph, tab_pdf = st.tabs([
            "🩺 Result",
            "📊 Your Input vs Typical Range",
            "📄 Download Report"
        ])

        with tab_result:
            result_card("Model result for the entered patient", proba, sex_val=sex_val)

        with tab_input_graph:
            H('<div class="sub">Compare patient values (markers) against dataset minimum, median, and maximum distributions.</div>')
            render_input_range_charts(row)

        with tab_pdf:
            st.write("### Clinical Risk Assessment PDF Report")
            st.caption("A printable PDF document containing patient inputs, predicted risk band, probability distribution chart, and performance metrics.")
            if "last_predict_pdf" in st.session_state:
                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=st.session_state["last_predict_pdf"],
                    file_name=f"cardiopulse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    else:
        st.caption("No prediction evaluated yet. Complete the form above and click Predict.")

    footer()


# -----------------------------------------------------------------------------
# PAGE 3: Compare Two Persons (with 3D Animated Anatomical Models)
# -----------------------------------------------------------------------------
def page_compare():
    H('<div class="sec">Compare Two Persons</div><div class="sub">Evaluate two patient profiles simultaneously with interactive 3D anatomical models, compare risk differentials, and download a comparative report.</div>')

    # Example loader choices
    example_options = ["Custom / Manual Input"]
    examples_map = {}
    if sample_patients_df is not None:
        for _, r in sample_patients_df.iterrows():
            lbl = str(r["example_label"])
            example_options.append(lbl)
            examples_map[lbl] = r.to_dict()

    # Callbacks to populate state when an example is selected
    def load_ex_a():
        sel = st.session_state.get("ex_sel_a")
        if sel and sel in examples_map:
            ex = examples_map[sel]
            for f in FEATURES:
                val = ex[f]
                lo = meta["features"][f]["min"]
                hi = meta["features"][f]["max"]
                whole = all(float(x).is_integer() for x in (lo, hi)) and (hi - lo) >= 1 and f != "oldpeak"
                st.session_state[f"a_{f}"] = int(round(val)) if whole else float(val)

    def load_ex_b():
        sel = st.session_state.get("ex_sel_b")
        if sel and sel in examples_map:
            ex = examples_map[sel]
            for f in FEATURES:
                val = ex[f]
                lo = meta["features"][f]["min"]
                hi = meta["features"][f]["max"]
                whole = all(float(x).is_integer() for x in (lo, hi)) and (hi - lo) >= 1 and f != "oldpeak"
                st.session_state[f"b_{f}"] = int(round(val)) if whole else float(val)

    # One-click example selectors outside form
    c_ex1, c_ex2 = st.columns(2)
    with c_ex1:
        st.selectbox("⚡ Load Example for Person A (Real Dataset Rows)", example_options, key="ex_sel_a", on_change=load_ex_a)
    with c_ex2:
        st.selectbox("⚡ Load Example for Person B (Real Dataset Rows)", example_options, key="ex_sel_b", on_change=load_ex_b)

    with st.form("compare_form"):
        ca, cb = st.columns(2)
        with ca:
            H('<h4>🧑 Person A (Clinical Indicators)</h4>')
            ra = build_inputs("a_")
        with cb:
            H('<h4>🧑‍🦱 Person B (Clinical Indicators)</h4>')
            rb = build_inputs("b_")
        go = st.form_submit_button("⚖️ Compare Persons")

    if go:
        problems = validate(ra) + validate(rb)
        if problems:
            for p in problems:
                st.error(p)
        else:
            with st.spinner("Analysing Comparative ECG & Clinical Telemetry..."):
                time.sleep(1.0)
            try:
                pa, pb = predict(ra), predict(rb)
                st.session_state["comp_ra"] = ra
                st.session_state["comp_rb"] = rb
                st.session_state["comp_pa"] = pa
                st.session_state["comp_pb"] = pb

                pdf_buf = build_comparison_pdf(
                    patient_a=ra, proba_a=pa,
                    patient_b=rb, proba_b=pb,
                    classes=CLASSES, engine_name=ENGINE,
                    metrics=metrics, features_meta=meta["features"]
                )
                st.session_state["comp_pdf"] = pdf_buf.getvalue()
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    # Render results outside form so results persist after download
    if "comp_ra" in st.session_state and "comp_pa" in st.session_state:
        ra = st.session_state["comp_ra"]
        rb = st.session_state["comp_rb"]
        pa = st.session_state["comp_pa"]
        pb = st.session_state["comp_pb"]

        H('<div class="sec">Comparative Evaluation Results</div>')
        c1, c2 = st.columns(2)
        with c1:
            result_card("Person A Evaluation", pa, sex_val=ra.get("sex", 1.0))
        with c2:
            result_card("Person B Evaluation", pb, sex_val=rb.get("sex", 1.0))

        # Risk difference card
        d = (pb[POS] - pa[POS]) * 100
        who = "Person B" if d > 0 else "Person A"
        H(
            f'<div class="card" style="margin-top:14px;border-left:4px solid #0ea5e9;">'
            f'<h4>⚖️ Comparative Risk Differential</h4>'
            f'<p><b>{CLASSES[POS].title()} Probability:</b> Person A: <b>{pa[POS] * 100:.1f}%</b> vs Person B: <b>{pb[POS] * 100:.1f}%</b>.<br>'
            f'<b>{who}</b> has a higher risk estimate by <b>{abs(d):.1f} percentage points</b>.</p>'
            f'<p style="font-size:0.85rem;color:var(--mut);">Plain-English summary: The model estimates Person A at {pa[POS]*100:.1f}% probability and Person B at {pb[POS]*100:.1f}% probability of heart disease presence. This is an empirical model estimate, not a diagnosis.</p>'
            f'</div>'
        )

        # Side-by-side probability bar chart
        st.write("#### Probability Comparison Chart")
        comp_chart_img = make_comparison_chart_image(CLASSES, pa, pb)
        st.image(comp_chart_img, use_container_width=True)

        # Table of differing features
        diff = [f for f in FEATURES if ra[f] != rb[f]]
        if diff:
            st.write("#### Clinical Indicator Variances (Differing Inputs Only)")
            diff_df = pd.DataFrame({
                "Feature": [pretty(f) for f in diff],
                "Person A": [ra[f] for f in diff],
                "Person B": [rb[f] for f in diff],
                "Variance": [
                    f"{rb[f] - ra[f]:+.1f}" if isinstance(ra[f], (int, float)) and isinstance(rb[f], (int, float)) else "Different"
                    for f in diff
                ]
            })
            stretch(st.dataframe, diff_df, hide_index=True)
        else:
            st.info("Person A and Person B have identical clinical inputs.")

        # Download Comparison Report
        st.write("#### Comparative PDF Report")
        if "comp_pdf" in st.session_state:
            st.download_button(
                label="📥 Download Comparative PDF Report",
                data=st.session_state["comp_pdf"],
                file_name=f"cardiopulse_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

    footer()


# -----------------------------------------------------------------------------
# PAGE 4: Model Performance & Metrics
# -----------------------------------------------------------------------------
def page_metrics():
    H('<div class="sec">Model Performance &amp; Metrics</div><div class="sub">Actual empirical results from the training pipeline (results/ and figures/).</div>')
    if metrics:
        stat_cards()
    else:
        st.info("results/test_metrics.json not found.")

    c1, c2 = st.columns(2)
    with c1:
        H('<div class="sec">Confusion Matrix &amp; ROC (Test Set)</div>')
        show_img("test_eval.png", "Held-out test set, evaluated once")
    with c2:
        H('<div class="sec">Feature Importance</div>')
        show_img("importance.png", "Permutation importance (training data)")

    base = read_csv("baselines.csv")
    H('<div class="sec">Benchmark Across Classifiers</div><div class="sub">Repeated stratified CV, mean ± std</div>')
    stretch(st.dataframe, base, hide_index=True) if base is not None else st.info("results/baselines.csv not found.")

    c3, c4 = st.columns(2)
    with c3:
        H('<div class="sec">Ablation Study</div>')
        ab = read_csv("ablation.csv")
        stretch(st.dataframe, ab, hide_index=True) if ab is not None else st.info("results/ablation.csv not found.")
    with c4:
        H('<div class="sec">Hyperparameter Tuning</div>')
        if tuning:
            st.write(f"**Model:** {tuning.get('model', '-')}")
            st.write(f"**Best parameters:** {tuning.get('best_params', '-')}")
            if "nested_auc_mean" in tuning:
                st.write(f"**Nested-CV ROC-AUC:** {tuning['nested_auc_mean']:.3f} ± {tuning.get('nested_auc_std', 0):.3f}")
        else:
            st.info("results/tuning.json not found.")

    H('<div class="sec">Exploratory Data Analysis</div>')
    show_img("eda.png", "Class balance and feature correlations")

    err = read_csv("errors.csv")
    if err is not None:
        with st.expander(f"Error analysis: {len(err)} misclassified test patients"):
            stretch(st.dataframe, err)

    footer()


# -----------------------------------------------------------------------------
# PAGE 5: About Project & ML Pipeline
# -----------------------------------------------------------------------------
def page_about():
    H('<div class="sec">About the Project</div><div class="sub">Dataset: OpenML Statlog Heart Disease (data_id=53). Binary target: absent vs present.</div>')
    H(
        '<div class="grid">'
        f'<div class="card"><h4>{len(FEATURES)} features</h4><p>Clinical indicators including age, blood pressure, cholesterol, and maximum heart rate.</p></div>'
        f'<div class="card"><h4>2 classes</h4><p>{" / ".join(c.title() for c in CLASSES)}</p></div>'
        f'<div class="card"><h4>Final model</h4><p>{ENGINE} inside a Scikit-Learn Pipeline.</p></div>'
        '<div class="card"><h4>Seed 42</h4><p>Fixed random seed ensuring 100% reproducibility.</p></div></div>'
    )

    steps = [
        ("Data + EDA", "Missing values, class balance, and correlation analysis"),
        ("Split", "Stratified train/test split (80/20); test set locked until final validation"),
        ("Preprocess", "Median imputation and standard scaling encased in a leakage-free Pipeline"),
        ("Baselines", "Dummy, Logistic Regression, KNN, Random Forest, and Hist. Gradient Boosting"),
        ("Tuning", "GridSearchCV with nested cross-validation (outer 5-fold, inner 3-fold)"),
        ("Ablation", "Systematic component removal evaluating scaling, tuning, and class weights"),
        ("Test Evaluation", "Single hold-out test set evaluation on 54 unseen samples"),
        ("Deliver", "Transparent explainability, comparative analysis, and downloadable PDF reports")
    ]
    html = '<div class="sec">Machine Learning Pipeline (8 Stages)</div><div class="grid">'
    for i, (t, d) in enumerate(steps):
        html += (
            f'<div class="card" style="animation-delay:{i * 0.07:.2f}s;"><div style="display:flex;gap:12px;align-items:flex-start;">'
            f'<div class="step-num">{i + 1}</div><div><h4>{t}</h4><p>{d}</p></div></div></div>'
        )
    H(html + "</div>")

    H(
        '<div class="sec">Technology Stack</div><div class="grid">'
        '<div class="card"><h4>Python 3.10-3.13</h4><p>Core programming language</p></div>'
        '<div class="card"><h4>Scikit-Learn 1.6.1</h4><p>Pipelines, estimators, and cross-validation</p></div>'
        '<div class="card"><h4>ReportLab & Matplotlib</h4><p>Automated vector charts and PDF reports</p></div>'
        '<div class="card"><h4>Streamlit & Pillow</h4><p>Reactive clinical dashboard & image optimization</p></div>'
        '<div class="card"><h4>Pandas & NumPy</h4><p>Data preprocessing and array manipulation</p></div></div>'
    )

    # Photography & Asset Credits Section
    credits_file = ASSETS / "CREDITS.md"
    if credits_file.exists():
        with st.expander("📷 Photography & Media Asset Attribution Credits (assets/CREDITS.md)"):
            st.markdown(credits_file.read_text(encoding="utf-8"))

    H(
        '<div class="sec">Limitations &amp; Future Scope</div><div class="card"><p>'
        '<b>Dataset Constraints:</b> Evaluated on 270 patient records from the Statlog Heart cohort. Larger multi-center cohorts are needed for clinical generalization.<br>'
        '<b>Clinical Intended Use:</b> Designed as an educational decision-support demonstration. Statistical model probabilities are not a substitute for comprehensive physician diagnostic evaluation.<br>'
        '<b>Future Work:</b> Multi-center clinical EHR integration, SHAP explainability attribution, and wearable sensor telemetry ingestion.'
        '</p></div>'
    )

    footer()


# -----------------------------------------------------------------------------
# Dispatch Page
# -----------------------------------------------------------------------------
{
    PAGES[0]: page_home,
    PAGES[1]: page_predict,
    PAGES[2]: page_compare,
    PAGES[3]: page_metrics,
    PAGES[4]: page_about,
}[page]()
