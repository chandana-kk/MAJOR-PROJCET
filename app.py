"""
EnergyPulse — Home Electricity Dashboard
==================================================
Smart home energy monitoring, predictions, cost tracking,
appliance breakdown, optimization tips, and usage trends.

Launch:
    streamlit run app.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import copy
import calendar as cal
import pickle
import re
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from replay import ReplaySimulator
from appliances import get_appliance_breakdown
from cost import daily_cost, weekly_cost, next_month_cost
from optimize import detect_anomalies, generate_anomaly_tips, calendar_tips, whatif_simulator
from model import (
    load_data, predict_next_period,
    FEATURE_COLS, TARGET, WINDOW_SIZE,
)
from data import remap_to_current_dates
from i18n import T, TLIST, LANGS, HOME_TYPES, home_type_label


def init_language():
    """Ensure the selected language exists in session state (persists across tabs)."""
    if "lang" not in st.session_state:
        st.session_state.lang = "en"


def render_language_selector(key="lang_select", centered=False):
    """Language dropdown bound to st.session_state['lang'] so it survives reruns."""
    init_language()
    codes = list(LANGS.keys())
    current = st.session_state.lang if st.session_state.lang in codes else "en"
    names = [LANGS[c] for c in codes]
    container = st.container()
    if centered:
        c1, c2, c3 = container.columns([1, 1.4, 1])
        container = c2
    with container:
        chosen_name = st.selectbox(
            T("lang_label"),
            names,
            index=codes.index(current),
            key=key,
        )
    chosen = codes[names.index(chosen_name)] if chosen_name in names else current
    if chosen != st.session_state.lang:
        st.session_state.lang = chosen
        st.rerun()

# ── Paths ───────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.pkl")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="EnergyPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── City tariff defaults ───────────────────────────────────────
CITY_TARIFF_DEFAULTS = {
    "Mumbai": 9.5, "Delhi": 8.5, "Bangalore": 6.5,
    "Chennai": 7.0, "Kolkata": 7.5, "Hyderabad": 6.0,
    "Pune": 8.0, "Ahmedabad": 5.5,
}

APPLIANCE_PRESETS = [
    ("Air Conditioner", "❄️"),
    ("Refrigerator", "🧊"),
    ("Washing Machine", "🧴"),
    ("Water Heater", "💨"),
    ("Television", "📺"),
    ("Microwave", "🍴"),
    ("Lights & Fans", "💡"),
    ("Other", "⚙️"),
]

ACCENT = "#4A6741"
ACCENT2 = "#6B8F5E"
LIGHT = "#9CA3AF"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    .stApp, section[data-testid="stSidebar"] {
        font-family: 'Inter', -apple-system, sans-serif !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    .section-header {
        display: flex; align-items: center; gap: 12px;
        margin: 1.4rem 0 0.6rem 0;
    }
    .section-header h2 {
        color: #1A1A1A; font-size: 1.05rem; font-weight: 700;
        margin: 0; white-space: nowrap; display: flex; align-items: center; gap: 8px;
    }
    .section-line {
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(74,103,65,0.25), transparent 80%);
    }
    .metric-card {
        background: #FFFFFF; border: 1px solid #E8E5E0;
        border-radius: 16px; padding: 1.2rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative;
        min-height: 120px; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .mc-green::before { background: linear-gradient(90deg, #4A6741, #6B8F5E); }
    .mc-teal::before { background: linear-gradient(90deg, #6B8F5E, #8FB882); }
    .mc-orange::before { background: linear-gradient(90deg, #C4944A, #D4B06A); }
    .mc-purple::before { background: linear-gradient(90deg, #7B6B8A, #9B8BAA); }
    .mc-rose::before { background: linear-gradient(90deg, #B07070, #D09090); }
    .metric-card:hover {
        border-color: rgba(74,103,65,0.2); transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.07), 0 2px 6px rgba(0,0,0,0.04);
    }
    .mc-label {
        color: #6B7280; font-size: 0.68rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.7px;
        margin-bottom: 0.4rem;
    }
    .mc-value {
        color: #1A1A1A; font-size: 1.65rem; font-weight: 800;
        line-height: 1.15; white-space: normal; display: flex; align-items: baseline; gap: 6px;
    }
    .mc-unit { font-size: 0.85rem; font-weight: 500; color: #9CA3AF; }
    .mc-trend {
        display: inline-flex; align-items: center; gap: 3px;
        font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
        border-radius: 6px; margin-top: 0.3rem;
    }
    .mc-trend.up { background: #E8F5E9; color: #2E7D32; }
    .mc-trend.down { background: #FFF3E0; color: #E65100; }
    .mc-trend.neutral { background: #E8F0E6; color: #4A6741; }
    .mc-sub {
        color: #9CA3AF; font-size: 0.68rem; margin-top: 0.3rem;
        white-space: normal;
    }
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricDelta"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
    }
    .live-bar {
        display: flex; align-items: center; gap: 10px;
        background: #FFFFFF; border: 1px solid #E8E5E0;
        border-radius: 14px; padding: 0.65rem 1.3rem; margin-bottom: 1rem;
        font-size: 0.82rem; color: #6B7280;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    .live-dot {
        width: 8px; height: 8px; background: #22C55E; border-radius: 50%;
        animation: gp 2s ease-in-out infinite; flex-shrink: 0;
        box-shadow: 0 0 6px rgba(34,197,94,0.4);
    }
    @keyframes gp {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
        50% { opacity: 0.6; box-shadow: 0 0 0 5px rgba(34,197,94,0); }
    }
    .live-label { color: #22C55E; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.3px; }
    .disc {
        background: #FFFFFF; border: 1px solid #E8E5E0;
        border-left: 3px solid #4A6741; border-radius: 0 14px 14px 0;
        padding: 1.1rem 1.5rem; margin-top: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    .disc p { color: #6B7280; font-size: 0.8rem; margin: 0; line-height: 1.65; }
    .disc strong { color: #4A6741; }
    .tip-box {
        background: #FFFFFF; border: 1px solid #E8E5E0;
        border-radius: 14px; padding: 1rem 1.3rem; margin-bottom: 0.7rem;
        font-size: 0.85rem; color: #6B7280; line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .tip-box:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04); }
    .tip-box strong { color: #4A6741; }
    .cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
    .cal-header {
        text-align: center; font-size: 0.7rem; font-weight: 700;
        color: #6B7280; padding: 4px 0; text-transform: uppercase;
    }
    .cal-day {
        text-align: center; border-radius: 8px; padding: 8px 4px;
        font-size: 0.75rem; font-weight: 600; cursor: pointer;
        transition: all 0.2s; border: 1px solid transparent;
    }
    .cal-day:hover { border-color: rgba(74,103,65,0.3); transform: scale(1.05); }
    .cal-day.low { background: rgba(74,103,65,0.1); color: #4A6741; }
    .cal-day.medium { background: rgba(196,148,74,0.12); color: #C4944A; }
    .cal-day.high { background: rgba(180,80,80,0.1); color: #B45050; }
    .cal-day.empty { background: transparent; cursor: default; }
    .cal-day.future { background: rgba(74,103,65,0.06); color: #6B8F5E; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2E3B2F 0%, #232E24 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span { color: rgba(255,255,255,0.75) !important; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small { color: rgba(255,255,255,0.45) !important; }
    .sb-section-label {
        color: rgba(255,255,255,0.5); font-size: 0.62rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.2px; margin: 0.9rem 0 0.35rem 0;
        display: flex; align-items: center; gap: 6px;
    }
    .sb-card {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 0.65rem 0.8rem; margin-bottom: 0.4rem;
        transition: background 0.2s;
    }
    .sb-card:hover { background: rgba(255,255,255,0.1); }
    .sb-card-title {
        color: rgba(255,255,255,0.95); font-size: 0.82rem; font-weight: 600;
        display: flex; align-items: center; justify-content: space-between;
    }
    .sb-card-sub {
        color: rgba(255,255,255,0.55); font-size: 0.7rem; margin-top: 2px;
    }
    .sb-appliance-tag {
        display: inline-flex; align-items: center; gap: 4px;
        background: rgba(74,103,65,0.2); border: 1px solid rgba(74,103,65,0.3);
        color: rgba(255,255,255,0.85); font-size: 0.68rem; font-weight: 500;
        padding: 3px 8px; border-radius: 6px; margin: 2px;
    }
    .login-container {
        max-width: 420px; margin: 0 auto; padding: 3rem 2.5rem;
        background: #FFFFFF; border: 1px solid #E8E5E0; border-radius: 20px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06), 0 1px 4px rgba(0,0,0,0.04);
    }
    .login-title {
        font-size: 1.6rem; font-weight: 800; color: #1A1A1A;
        text-align: center; margin-bottom: 0.3rem;
    }
    .login-subtitle {
        font-size: 0.85rem; color: #6B7280; text-align: center;
        margin-bottom: 1.8rem; line-height: 1.5;
    }
    .login-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 1.2rem 0; color: #9CA3AF; font-size: 0.75rem;
    }
    .login-divider::before, .login-divider::after {
        content: ''; flex: 1; height: 1px; background: #E8E5E0;
    }
    .onboard-container { max-width: 680px; margin: 0 auto; }
    .data-progress {
        height: 4px; background: rgba(74,103,65,0.1); border-radius: 4px;
        overflow: hidden; margin-top: 6px;
    }
    .data-progress-fill {
        height: 100%; background: linear-gradient(90deg, #4A6741, #6B8F5E);
        border-radius: 4px; transition: width 0.5s ease;
    }
    [data-testid="stHorizontalBlock"] { gap: 0.8rem !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #FFFFFF; border-radius: 12px;
        padding: 4px; border: 1px solid #E8E5E0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 10px 20px; font-weight: 600;
        font-size: 0.85rem; color: #6B7280; border: none; background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #4A6741; }
    .stTabs [aria-selected="true"] {
        background: #4A6741 !important; color: #FFFFFF !important; border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 0.82rem !important; transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .error-card {
        background: #FFFFFF; border: 1px solid #E8E5E0;
        border-left: 4px solid #C4944A; border-radius: 0 16px 16px 0;
        padding: 2rem 2.5rem; max-width: 520px; margin: 3rem auto;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        text-align: center;
    }
    .error-card-icon { font-size: 2.2rem; margin-bottom: 0.8rem; }
    .error-card-title { font-size: 1.1rem; font-weight: 700; color: #1A1A1A; margin-bottom: 0.4rem; }
    .error-card-msg { font-size: 0.85rem; color: #6B7280; line-height: 1.6; margin-bottom: 1.2rem; }
    .skeleton {
        background: linear-gradient(90deg, #E8E5E0 25%, #F0EDE8 50%, #E8E5E0 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 10px;
    }
    .skeleton-chart { height: 340px; width: 100%; }
    .skeleton-row { height: 16px; width: 80%; margin-bottom: 8px; }
    .skeleton-card { height: 120px; width: 100%; }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    .empty-state {
        text-align: center; padding: 3rem 2rem;
        color: #6B7280; font-size: 0.9rem;
    }
    .empty-state-icon { font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.6; }
    .empty-state-title { font-size: 1rem; font-weight: 700; color: #1A1A1A; margin-bottom: 0.3rem; }
    .hero {
        position: relative; padding: 1.8rem 2.5rem; margin-bottom: 1rem;
        border-radius: 20px; overflow: hidden;
        background: linear-gradient(135deg,#2E3B2F 0%,#3D4E3F 50%,#4A6741 100%);
        border: 1px solid rgba(74,103,65,0.2);
    }
    .hero-glow {
        position: absolute; top: 0; right: 0; width: 300px; height: 300px;
        background: radial-gradient(circle,rgba(107,143,94,0.15) 0%,transparent 70%);
        transform: translate(30%,-30%);
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
        color: #FFFFFF; padding: 5px 14px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; margin-bottom: 0.8rem;
    }
    .hero-title {
        color: #FFFFFF; font-size: 1.9rem; font-weight: 800;
        margin: 0 0 0.25rem 0; letter-spacing: -0.8px;
    }
    .hero-title span {
        background: linear-gradient(135deg,#C8E6C0,#A8D49E);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub { color: rgba(255,255,255,0.7); font-size: 0.88rem; margin: 0; }
    .bill-card {
        background: #FFFFFF; border: 1px solid #E8E5E0; border-radius: 14px;
        padding: 1rem 1.1rem; margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    }
    /* ── Responsive tweaks: laptop → tablet → mobile ── */
    @media (max-width: 1024px) {
        .hero { padding: 1.4rem 1.6rem; }
        .hero-title { font-size: 1.55rem; }
    }
    @media (max-width: 820px) {
        .metric-card { min-height: auto; padding: 1rem 0.85rem; }
        .mc-value { font-size: 1.35rem; flex-wrap: wrap; }
        .section-header h2 { font-size: 0.95rem; white-space: normal; }
        .live-bar { flex-wrap: wrap; padding: 0.6rem 0.9rem; row-gap: 6px; }
        .live-bar .data-progress { margin-left: auto; }
        .disc { padding: 1rem 1.1rem; }
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding-left: 0.9rem; padding-right: 0.9rem; }
        .hero { padding: 1.15rem 1.05rem; border-radius: 16px; }
        .hero-title { font-size: 1.3rem; letter-spacing: -0.4px; }
        .hero-sub { font-size: 0.78rem; }
        .metric-card { padding: 0.85rem 0.75rem; border-radius: 13px; }
        .mc-label { font-size: 0.62rem; }
        .mc-value { font-size: 1.22rem; }
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none;
        }
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        .stTabs [data-baseweb="tab"] { padding: 10px 14px; white-space: nowrap; }
        .cal-day { padding: 7px 2px; font-size: 0.68rem; }
        .cal-header { font-size: 0.62rem; }
        .stButton > button { min-height: 44px; }
        section[data-testid="stSidebar"] .stButton > button { min-height: 40px; }
    }
</style>
""", unsafe_allow_html=True)

_PLOTLY_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#4A5568"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.08)"),
    yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.08)"),
)


def PLOTLY_LAYOUT(**overrides):
    layout = copy.deepcopy(_PLOTLY_BASE)
    for k, v in overrides.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict) and k in layout:
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


def section(title, icon=""):
    icon_html = f'<span style="font-size:1rem;">{icon}</span>' if icon else ""
    st.markdown(f"""
    <div class="section-header">
        <h2>{icon_html} {title}</h2>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(col, label, value, unit="", sub="", trend_text="", trend_dir="neutral",
                css_class="mc-green"):
    parts = [
        f'<div class="metric-card {css_class}">',
        f'<div class="mc-label">{label}</div>',
        f'<div class="mc-value">{value}<span class="mc-unit"> {unit}</span></div>',
    ]
    if trend_text:
        arrow = "&#9650;" if trend_dir == "up" else ("&#9660;" if trend_dir == "down" else "&#9644;")
        parts.append(f'<div class="mc-trend {trend_dir}">{arrow} {trend_text}</div>')
    if sub:
        parts.append(f'<div class="mc-sub">{sub}</div>')
    parts.append('</div>')
    html = "".join(parts)
    with col:
        st.markdown(html, unsafe_allow_html=True)


def fmt_rs(val):
    return f"{val:,.2f}"


def fmt_kwh(val):
    return f"{val:.2f}"


@st.cache_data
def load_data_cached():
    return load_data()

@st.cache_resource
def load_xgb():
    with open(XGB_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_lstm():
    from tensorflow import keras
    return keras.models.load_model(LSTM_PATH)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_forecast():
    path = os.path.join(DATA_DIR, "next_month_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["datetime"])
    return pd.DataFrame()


def compute_scaling_factor(home_details):
    occupants = home_details.get("occupants", 4)
    appliances = home_details.get("appliances", [])
    num_appliances = len(appliances) if appliances else home_details.get("num_appliances", 3)
    baseline_occupants = 4
    baseline_appliances = 3
    scale = occupants / baseline_occupants
    scale *= 1 + (num_appliances - baseline_appliances) * 0.05
    return max(0.3, min(scale, 3.0))


def get_simulator():
    if "simulator" not in st.session_state:
        sim = ReplaySimulator(delay=2)
        sim.start()
        st.session_state.simulator = sim
    return st.session_state.simulator


def render_login_screen():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "email": None, "mode": "login"}
    if "user_db" not in st.session_state:
        st.session_state.user_db = {}

    auth = st.session_state.auth
    if auth.get("logged_in"):
        return True

    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
        <div style="display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#2E3B2F,#4A6741);
             color:#FFF;padding:8px 18px;border-radius:14px;margin-bottom:1rem;">
            <span style="font-size:1.2rem;">&#9889;</span>
            <span style="font-weight:800;font-size:0.85rem;letter-spacing:0.5px;">ENERGYPULSE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_language_selector(key="lang_select_login", centered=True)

    mode = auth.get("mode", "login")
    is_signup = (mode == "signup")

    title = T("login_create_title") if is_signup else T("login_welcome_back")
    subtitle = T("login_sub_signup") if is_signup else T("login_sub_login")

    st.markdown(f"""
    <div class="login-container">
        <div class="login-title">{title}</div>
        <div class="login-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

    login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
    with login_col2:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input(T("email_label"), placeholder=T("ph_email"), key="login_email")
            password = st.text_input(T("password_label"), type="password",
                                     placeholder=T("ph_password"), key="login_pw")
            if is_signup:
                name = st.text_input(T("name_label"), placeholder=T("ph_name"), key="login_name")
            btn_label = T("btn_create_account") if is_signup else T("btn_sign_in")
            submitted = st.form_submit_button(btn_label, width="stretch", type="primary")
            if submitted:
                if not email or not password:
                    st.error(T("err_missing"))
                elif len(password) < 4:
                    st.error(T("err_short_pw"))
                else:
                    if is_signup:
                        st.session_state.user_db[email] = {
                            "name": name if name else email.split("@")[0],
                            "password": password,
                        }
                    else:
                        existing = st.session_state.user_db.get(email)
                        if not existing or existing["password"] != password:
                            st.error(T("err_invalid"))
                            return False
                    auth["logged_in"] = True
                    auth["email"] = email
                    auth["name"] = st.session_state.user_db[email]["name"]
                    st.rerun()

        st.markdown(f'<div class="login-divider">{T("divider_or")}</div>', unsafe_allow_html=True)
        if is_signup:
            if st.button(T("switch_to_login"), width="stretch", key="switch_login"):
                auth["mode"] = "login"
                st.rerun()
        else:
            if st.button(T("switch_to_signup"), width="stretch", key="switch_signup"):
                auth["mode"] = "signup"
                st.rerun()
        st.markdown("")
        if st.button(T("btn_guest"), width="stretch", key="guest_btn"):
            auth["logged_in"] = True
            auth["email"] = "guest@demo.local"
            auth["name"] = "Guest"
            auth["guest"] = True
            st.rerun()

    return False


def render_onboarding():
    if "home_details" in st.session_state and st.session_state.home_details is not None:
        return True
    init_language()
    if "onboard_step" not in st.session_state:
        st.session_state.onboard_step = 1
    if "onboard_data" not in st.session_state:
        st.session_state.onboard_data = {
            "home_type": "Apartment", "occupants": 4, "appliances": [],
            "tariff_rate": 8.0, "city": "", "home_size": 0,
        }
    step = st.session_state.onboard_step
    od = st.session_state.onboard_data
    user_name = st.session_state.auth.get("name", "there")

    st.markdown(f"""
    <div class="onboard-container">
        <div style="text-align:center;margin-bottom:1.5rem;">
            <div style="font-size:1.5rem;font-weight:800;color:#1A1A1A;margin-bottom:0.2rem;">
                {T('ob_welcome', name=user_name)}
            </div>
            <div style="color:#6B7280;font-size:0.88rem;">
                {T('ob_setup_msg')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_language_selector(key="lang_select_onboard", centered=True)

    step_labels = [T("step_home_info"), T("step_appliances"), T("step_rate")]
    step_icons = ["🏠", "🔌", "💰"]
    cols = st.columns(3)
    for i, (label, icon) in enumerate(zip(step_labels, step_icons)):
        with cols[i]:
            active = (i + 1 == step)
            done = (i + 1 < step)
            color = ACCENT if active else ("#22C55E" if done else LIGHT)
            border = f"2px solid {color}" if active else "1px solid #E8E5E0"
            bg = "rgba(74,103,65,0.05)" if active else "#FFFFFF"
            check = " &#10003;" if done else ""
            st.markdown(f"""
            <div style="text-align:center;padding:0.7rem;border-radius:12px;border:{border};
                 background:{bg};margin-bottom:0.5rem;">
                <div style="font-size:1.2rem;">{icon}</div>
                <div style="font-size:0.72rem;font-weight:600;color:{color};margin-top:2px;">
                    {label}{check}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    if step == 1:
        _render_onboard_step1(od)
    elif step == 2:
        _render_onboard_step2(od)
    elif step == 3:
        _render_onboard_step3(od)
    return False


def _render_onboard_step1(od):
    section(T("ob_about_home"), "🏠")
    c1, c2 = st.columns(2)
    with c1:
        _ht = TLIST("home_types")
        home_type_name = st.selectbox(T("home_type_label"),
            _ht,
            index=_ht.index(home_type_label(od["home_type"])) if home_type_label(od["home_type"]) in _ht else 0,
            key="ob_home_type")
        od["home_type"] = HOME_TYPES[_ht.index(home_type_name)]
    with c2:
        occupants = st.number_input(T("occupants_label"),
            min_value=1, max_value=20, value=od["occupants"], step=1, key="ob_occupants")
        od["occupants"] = occupants
    c3, c4 = st.columns(2)
    with c3:
        home_size = st.number_input(T("size_label"),
            min_value=0, max_value=10000, value=od.get("home_size", 0), step=50, key="ob_size")
        od["home_size"] = home_size
    with c4:
        city = st.text_input(T("city_label"),
            value=od.get("city", ""), key="ob_city", placeholder=T("ph_city"))
        od["city"] = city
    st.markdown("")
    c_back, c_next = st.columns([1, 1])
    with c_next:
        if st.button(T("btn_next_appliances"), width="stretch", type="primary", key="ob_next1"):
            st.session_state.onboard_step = 2
            st.rerun()
    with c_back:
        st.button(T("btn_back"), width="stretch", disabled=True, key="ob_back1")


def _render_onboard_step2(od):
    section(T("ob_your_appliances"), "🔌")
    st.markdown(f'<div style="color:#6B7280;font-size:0.82rem;margin-bottom:0.8rem;">'
                f'{T("ob_app_hint")}</div>',
                unsafe_allow_html=True)
    appliances = od.get("appliances", [])
    if appliances:
        for i, app in enumerate(appliances):
            cols = st.columns([0.3, 2, 2, 0.3])
            with cols[0]:
                st.markdown(f'<div style="text-align:center;font-size:1.2rem;margin-top:0.3rem;">{app.get("icon", "⚙️")}</div>', unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f'<div style="font-size:0.85rem;font-weight:600;color:#1A1A1A;">{app["name"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.72rem;color:#6B7280;">{app["type"]}</div>', unsafe_allow_html=True)
            with cols[2]:
                usage = st.select_slider(T("typical_usage"), options=["Low", "Medium", "High"],
                    value=app.get("usage", "Medium"), key=f"app_usage_{i}")
                od["appliances"][i]["usage"] = usage
            with cols[3]:
                if st.button("✕", key=f"app_del_{i}"):
                    od["appliances"].pop(i)
                    st.rerun()
        st.markdown("---")
    with st.expander(T("add_appliance_expander"), expanded=not appliances):
        add_cols = st.columns([2, 2, 1])
        with add_cols[0]:
            preset_names = [p[0] for p in APPLIANCE_PRESETS]
            chosen = st.selectbox(T("appliance_label"), preset_names + [T("custom_option")], key="ob_app_preset")
            custom_name = ""
            if chosen == T("custom_option"):
                custom_name = st.text_input(T("name_label"), placeholder=T("name_ph"), key="ob_app_custom")
        with add_cols[1]:
            app_type = st.selectbox(T("category_label"),
                ["Cooling", "Heating", "Kitchen", "Laundry", "Electronics", "Lighting", "Other"],
                key="ob_app_type")
        with add_cols[2]:
            st.markdown('<div style="margin-top:1.8rem;"></div>', unsafe_allow_html=True)
            if st.button(T("btn_add"), key="ob_add_app", type="primary"):
                name = custom_name if chosen == T("custom_option") and custom_name else chosen
                icon = "⚙️"
                for p_name, p_icon in APPLIANCE_PRESETS:
                    if p_name == chosen:
                        icon = p_icon
                        break
                od["appliances"].append({"name": name, "type": app_type, "usage": "Medium", "icon": icon})
                st.rerun()
    st.markdown("")
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button(T("btn_back"), width="stretch", key="ob_back2"):
            st.session_state.onboard_step = 1
            st.rerun()
    with c_next:
        if st.button(T("btn_next_rate"), width="stretch", type="primary", key="ob_next2"):
            st.session_state.onboard_step = 3
            st.rerun()


def _render_onboard_step3(od):
    section(T("rate_section_title"), "💰")
    st.markdown(f'<div style="color:#6B7280;font-size:0.82rem;margin-bottom:0.8rem;">'
                f'{T("rate_hint")}</div>',
                unsafe_allow_html=True)
    city = od.get("city", "")
    tariff_val = od.get("tariff_rate", 8.0)
    if city and tariff_val == 8.0:
        suggested = CITY_TARIFF_DEFAULTS.get(city.strip().title(), None)
        if suggested:
            st.info(T("avg_rate_for", city=city.strip().title(), rate=suggested))
            if st.button(T("use_city_rate", rate=suggested), key="ob_use_city_rate", type="secondary"):
                od["tariff_rate"] = suggested
                st.rerun()
    tariff = st.number_input(T("your_rate_label"),
        min_value=0.5, max_value=30.0, value=float(od.get("tariff_rate", 8.0)),
        step=0.5, key="ob_tariff")
    od["tariff_rate"] = tariff
    st.markdown(f"""
    <div style="background:#F5F3EE;border:1px solid #E8E5E0;border-radius:10px;padding:0.8rem 1rem;margin-top:0.3rem;">
        <div style="font-size:0.78rem;color:#6B7280;">
            {T("rate_hint_avg")}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button(T("btn_back"), width="stretch", key="ob_back3"):
            st.session_state.onboard_step = 2
            st.rerun()
    with c_next:
        if st.button(T("btn_start_dashboard"), width="stretch", type="primary", key="ob_finish"):
            st.session_state.home_details = {
                "home_type": od["home_type"],
                "occupants": od["occupants"],
                "appliances": od["appliances"],
                "num_appliances": len(od["appliances"]),
                "tariff_rate": od["tariff_rate"],
                "city": od.get("city", ""),
                "home_size": od.get("home_size", 0) if od.get("home_size", 0) > 0 else None,
                "peak_morning": (6, 10),
                "peak_evening": (18, 22),
            }
            for k in ["onboard_data", "onboard_step"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()


def render_sidebar():
    init_language()
    hd = st.session_state.home_details
    appliances = hd.get("appliances", [])
    with st.sidebar:
        user_name = st.session_state.auth.get("name", "Guest")
        is_guest = st.session_state.auth.get("guest", False)
        render_language_selector(key="lang_select")
        st.markdown(f"""
        <div style="padding:0.2rem 0 0.8rem 0;">
            <div style="font-size:1.1rem;font-weight:800;color:#FFFFFF;display:flex;align-items:center;gap:8px;">
                {T('sb_my_home')}
            </div>
            <div style="color:rgba(255,255,255,0.55);font-size:0.72rem;margin-top:2px;">
                {user_name}{"  ·  " + T('guest_tag') if is_guest else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="sb-section-label">{T("sb_section_home")}</div>', unsafe_allow_html=True)
        home_size_text = ("  ·  " + T("sqft_suffix", n=hd.get("home_size", ""))) if hd.get("home_size") else ""
        st.markdown(f"""
        <div class="sb-card">
            <div class="sb-card-title">
                {home_type_label(hd.get('home_type', 'Home'))}
                <span style="font-size:0.7rem;color:rgba(255,255,255,0.4);cursor:pointer;">&#9998;</span>
            </div>
            <div class="sb-card-sub">{T('people_suffix', n=hd.get('occupants', 0))}{home_size_text}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(T("sb_edit_home"), width="stretch", key="sb_edit_home"):
            st.session_state.home_details = None
            st.session_state.onboard_step = 1
            st.session_state.onboard_data = {
                "home_type": hd.get("home_type", "Apartment"), "occupants": hd.get("occupants", 1),
                "appliances": appliances, "tariff_rate": hd.get("tariff_rate", 8.0),
                "city": hd.get("city", ""), "home_size": hd.get("home_size", 0) or 0,
            }
            st.rerun()
        st.markdown(f'<div class="sb-section-label">{T("sb_section_appliances")}</div>', unsafe_allow_html=True)
        if appliances:
            tags_html = ""
            for app in appliances:
                tags_html += f'<span class="sb-appliance-tag">{app.get("icon", "")} {app["name"]}</span>'
            st.markdown(f"""
        <div class="sb-card">
            <div style="display:flex;flex-wrap:wrap;gap:3px;">{tags_html}</div>
        </div>
        """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="sb-card"><div class="sb-card-sub">{T("sb_no_appliances")}</div></div>',
                       unsafe_allow_html=True)
        st.markdown(f'<div class="sb-section-label">{T("sb_section_rate")}</div>', unsafe_allow_html=True)
        tariff_rate = st.number_input(T("sb_rate_label"), min_value=0.0,
            value=float(hd.get("tariff_rate", 8.0)), step=0.5,
            key="sb_tariff", label_visibility="collapsed")
        hd["tariff_rate"] = tariff_rate
        st.markdown(f'<div style="color:rgba(255,255,255,0.4);font-size:0.68rem;margin-top:-0.6rem;margin-bottom:0.4rem;">{T("sb_rate_hint")}</div>',
                   unsafe_allow_html=True)
        st.markdown(f'<div class="sb-section-label">{T("sb_section_peak")}</div>', unsafe_allow_html=True)
        peak_morning = st.slider(T("sb_peak_am"), 0, 23, (6, 10), key="sb_peak_am")
        peak_evening = st.slider(T("sb_peak_pm"), 0, 23, (18, 22), key="sb_peak_pm")
        st.markdown(f'<div class="sb-section-label">{T("sb_section_display")}</div>', unsafe_allow_html=True)
        display_count = st.slider(T("sb_trend_window"), 20, 500, 100, key="sb_display")
        st.markdown(f'<div class="sb-section-label">{T("sb_section_calendar")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:rgba(255,255,255,0.5);font-size:0.68rem;margin:0 0 0.3rem 0;">{T("sb_viewing_data")}</div>', unsafe_allow_html=True)
        cal_cols = st.columns(2)
        month_names = TLIST("months")
        with cal_cols[0]:
            cal_month_name = st.selectbox(T("month_label"), month_names,
                index=pd.Timestamp.now().month - 1, key="cal_month_sb")
            cal_month = month_names.index(cal_month_name) + 1
        with cal_cols[1]:
            cal_year = st.number_input(T("year_label"), 2006, 2030,
                value=pd.Timestamp.now().year, key="cal_year_sb")
        st.markdown("---")
        st.markdown(f'<div class="sb-section-label">{T("sb_section_simulation")}</div>', unsafe_allow_html=True)
        if st.button(T("btn_restart_demo"), width="stretch", key="sb_restart"):
            if "simulator" in st.session_state:
                st.session_state.simulator.reset()
                st.session_state.simulator.start()
            st.rerun()
        st.markdown(f'<div style="color:rgba(255,255,255,0.35);font-size:0.68rem;margin-top:-0.5rem;">{T("sb_restart_hint")}</div>',
                   unsafe_allow_html=True)
        st.markdown('<div style="height:1px;background:rgba(255,255,255,0.1);margin:0.6rem 0;"></div>', unsafe_allow_html=True)
        if st.button(T("btn_sign_out"), width="stretch", key="sb_signout"):
            if "simulator" in st.session_state:
                st.session_state.simulator.stop()
            for k in ["auth", "home_details", "simulator", "onboard_data", "onboard_step", "sidebar_settings"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    settings = {"tariff_rate": tariff_rate, "peak_morning": peak_morning, "peak_evening": peak_evening,
                "display_count": display_count, "cal_month": cal_month, "cal_year": cal_year}
    st.session_state.sidebar_settings = settings
    return settings


def render_calendar(df, forecast_df, year, month, tariff_rate):
    cal_obj = cal.Calendar(firstweekday=0)
    month_days = cal_obj.monthdayscalendar(year, month)
    daily_usage = {}
    if not df.empty:
        month_mask = (df["datetime"].dt.year == year) & (df["datetime"].dt.month == month)
        month_data = df.loc[month_mask].copy()
        if not month_data.empty:
            month_data["date"] = month_data["datetime"].dt.date
            daily_kwh = month_data.groupby("date")[TARGET].sum()
            for d, kwh in daily_kwh.items():
                daily_usage[d] = kwh * tariff_rate
    daily_forecast = {}
    if not forecast_df.empty:
        fc_month = forecast_df[
            (forecast_df["datetime"].dt.year == year) &
            (forecast_df["datetime"].dt.month == month)
        ]
        if not fc_month.empty:
            fc_daily = fc_month.groupby(fc_month["datetime"].dt.date)["predicted_kwh"].sum()
            for d, kwh in fc_daily.items():
                daily_forecast[d] = kwh * tariff_rate
    all_costs = list(daily_usage.values()) + list(daily_forecast.values())
    p33, p66 = (np.percentile(all_costs, 33), np.percentile(all_costs, 66)) if all_costs else (10, 30)

    def classify(cost):
        return "low" if cost <= p33 else ("medium" if cost <= p66 else "high")

    html = '<div style="background:#FFFFFF;border:1px solid #E8E5E0;border-radius:16px;padding:1.2rem;box-shadow:0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.03);">'
    html += f'<div style="color:#1A1A1A;font-size:1rem;font-weight:700;margin-bottom:0.8rem;text-align:center;">{TLIST("months")[month - 1]} {year}</div>'
    if not daily_usage and not daily_forecast:
        html += f'<div style="text-align:center;padding:2rem 1rem;color:#9CA3AF;font-size:0.85rem;">{T("cal_no_data")}</div>'
    else:
        html += '<div class="cal-grid">'
        for d in TLIST("weekdays"):
            html += f'<div class="cal-header">{d}</div>'
        for week in month_days:
            for day in week:
                if day == 0:
                    html += '<div class="cal-day empty"></div>'
                    continue
                dt = pd.Timestamp(year, month, day).date()
                if dt in daily_usage:
                    cost = daily_usage[dt]
                    html += f'<div class="cal-day {classify(cost)}" title="{dt}: Rs. {cost:.2f}">{day}<br><span style="font-size:0.6rem">Rs.{cost:.0f}</span></div>'
                elif dt in daily_forecast:
                    cost = daily_forecast[dt]
                    html += f'<div class="cal-day future" title="{dt}: Rs. {cost:.2f} (est.)">{day}<br><span style="font-size:0.6rem">Rs.{cost:.0f}</span></div>'
                else:
                    html += f'<div class="cal-day empty">{day}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Bill upload (photo / file) ────────────────────────────────
BILL_FILE_TYPES = ["jpg", "jpeg", "png", "webp", "heic", "heif", "pdf"]
IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "heic", "heif"}

UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)[ \t]*(?:kwh|units?)\b", re.IGNORECASE)
UNIT_PREFIX_RE = re.compile(
    r"(?:total[ \t]*units?|units?[ \t]*consumed|consumption|units?\b)[^0-9\n]{0,15}?(\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(r"(?:rs\.?|inr|\u20b9)\s*[:\-]?\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
PERIOD_RE = re.compile(
    r"(\d{1,2}[-/][A-Za-z0-9]{2,7}[-/]\d{2,4})\s*(?:to|-|\u2013|until)\s*(\d{1,2}[-/][A-Za-z0-9]{2,7}[-/]\d{2,4})"
    r"|([A-Za-z]{3,9}[\s'-]\d{2,4})\s*(?:to|-|\u2013)\s*([A-Za-z]{3,9}[\s'-]\d{2,4})"
)


def ocr_image_bytes(data: bytes):
    """
    Best-effort OCR on an image. Returns (text, status):
      status True  -> OCR ran successfully
      status False -> pytesseract/PIL not available (engine missing)
      status None  -> OCR attempted but failed on this image
    """
    try:
        import io
        import pytesseract
        from PIL import Image
    except ImportError:
        return "", False
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        return pytesseract.image_to_string(img), True
    except Exception:
        return "", None


def extract_bill_details(text: str) -> dict:
    """Pull units consumed, amount and billing period out of raw bill text."""
    details = {}
    if not text:
        return details
    m = UNIT_RE.search(text)
    if m:
        details["units"] = m.group(1)
    else:
        m = UNIT_PREFIX_RE.search(text)
        if m:
            details["units"] = m.group(1)
    amounts = AMOUNT_RE.findall(text)
    if amounts:
        biggest = max(amounts, key=lambda a: float(a.replace(",", "") or 0))
        details["amount"] = biggest
    m = PERIOD_RE.search(text)
    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            details["period"] = f"{groups[0]} - {groups[1]}"
    return details


def month_options():
    """Last 12 months as localized labels, newest first."""
    now = pd.Timestamp.now()
    opts = []
    for i in range(12):
        d = now - pd.DateOffset(months=i)
        opts.append(f"{TLIST('months')[d.month - 1]} {d.year}")
    return opts


def save_uploaded_bill(data: bytes, filename: str, ext: str, month_label: str):
    """OCR (images only), store against the chosen month, report what happened."""
    text, ocr_status = (("", False) if ext == "pdf" else ocr_image_bytes(data))
    details = extract_bill_details(text)
    st.session_state.setdefault("bills", []).append({
        "name": filename,
        "ext": ext,
        "data": data,
        "month": month_label,
        "details": details,
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    })

    st.success(T("bill_saved_ok", month=month_label))
    if ext != "pdf" and ocr_status is False:
        st.info(T("ocr_unavailable"))
    elif ext != "pdf" and ocr_status is None:
        st.warning(T("ocr_failed"))
    elif ext == "pdf":
        st.caption(T("pdf_stored_note"))
    if details:
        chips = "".join([
            f'<span><strong style="color:#4A6741;">{T("units_consumed")}:</strong> {details.get("units", "-")}</span>',
            f'<span><strong style="color:#4A6741;">{T("amount_paid")}:</strong> '
            + (f'Rs. {details.get("amount")}' if details.get("amount") else "-") + '</span>',
            f'<span><strong style="color:#4A6741;">{T("billing_period")}:</strong> {details.get("period", "-")}</span>',
        ])
        st.markdown(f"""
        <div class="bill-card">
            <div style="font-weight:700;font-size:0.85rem;color:#1A1A1A;margin-bottom:0.35rem;">
                {T("extracted_details")}
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:6px 18px;font-size:0.82rem;color:#374151;">
                {chips}
            </div>
        </div>
        """, unsafe_allow_html=True)


def bill_signature(data: bytes, name: str) -> str:
    return f"{name}:{hashlib.md5(data).hexdigest()}"


def render_saved_bills():
    bills = st.session_state.get("bills", [])
    if not bills:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-icon">\U0001f4c4</div>
            <div class="empty-state-title">{T('no_bills_yet')}</div>
        </div>
        """, unsafe_allow_html=True)
        return
    section(T("saved_bills_header", n=len(bills)), "\U0001f5c2")
    for idx in range(len(bills) - 1, -1, -1):
        b = bills[idx]
        with st.container():
            c_img, c_info, c_del = st.columns([1, 2.2, 0.6])
            with c_img:
                shown = False
                if b["ext"] in IMAGE_EXTS:
                    try:
                        st.image(b["data"], width=140)
                        shown = True
                    except Exception:
                        shown = False
                if not shown:
                    icon = "\U0001f4c4" if b["ext"] == "pdf" else "\U0001f5bc"
                    st.markdown(f'<div style="font-size:3rem;text-align:center;padding:0.6rem 0;">{icon}</div>',
                                unsafe_allow_html=True)
            with c_info:
                st.markdown(
                    f'<div style="font-weight:700;font-size:0.88rem;color:#1A1A1A;'
                    f'overflow-wrap:anywhere;">{b["name"]}</div>'
                    f'<div style="font-size:0.74rem;color:#6B7280;margin-top:2px;">'
                    f'{b["month"]} · {T("billing_period")}: '
                    f'{b["details"].get("period", "-")}<br>'
                    f'{T("units_consumed")}: {b["details"].get("units", "-")} · '
                    f'{T("amount_paid")}: Rs. {b["details"].get("amount", "-")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with c_del:
                st.write("")
                if st.button("🗑 " + T("btn_delete_bill"), key=f"bill_del_{idx}",
                             type="secondary"):
                    bills.pop(idx)
                    st.rerun()


def render_bills_tab():
    init_language()
    section(T("tab_bills"), "\U0001f4f8")
    st.markdown(f'<div style="color:#6B7280;font-size:0.85rem;margin-bottom:0.8rem;">'
                f'{T("bill_intro")}</div>', unsafe_allow_html=True)

    up_col, cam_col = st.columns(2)
    uploaded = None
    with up_col:
        uploaded = st.file_uploader(
            T("bill_uploader_label"),
            type=BILL_FILE_TYPES,
            help=T("bill_uploader_help"),
            key="bill_uploader",
        )
    with cam_col:
        camera_img = st.camera_input(
            T("bill_camera_label"),
            help=T("bill_camera_help"),
            key="bill_camera",
        )

    month_choice = st.selectbox(T("bill_month_label"), month_options(),
                                index=0, key="bill_month")

    if uploaded is not None:
        data = uploaded.getvalue()
        sig = bill_signature(data, uploaded.name)
        seen = st.session_state.setdefault("bill_seen", [])
        if sig not in seen:
            seen.append(sig)
            ext = uploaded.name.lower().split(".")[-1] or "bin"
            save_uploaded_bill(data, uploaded.name, ext, month_choice)
    elif camera_img is not None:
        data = camera_img.getvalue()
        sig = bill_signature(data, "camera")
        seen = st.session_state.setdefault("bill_seen", [])
        if sig not in seen:
            seen.append(sig)
            ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            save_uploaded_bill(data, f"bill_camera_{ts}.jpg", "jpeg", month_choice)

    render_saved_bills()


def main_dashboard():
    init_language()
    try:
        _main_dashboard_inner()
    except Exception as e:
        st.markdown(f"""
        <div class="error-card">
            <div class="error-card-icon">⚠️</div>
            <div class="error-card-title">{T('error_title')}</div>
            <div class="error-card-msg">
                {T('error_msg')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if e:
            st.caption(f"{type(e).__name__}: {str(e)[:100]}...")
        c1, c2 = st.columns(2)
        with c1:
            if st.button(T("btn_go_home"), type="primary", key="err_go_home"):
                st.session_state.home_details = None
                st.session_state.onboard_step = 1
                st.rerun()
        with c2:
            if st.button(T("btn_retry"), key="err_retry"):
                st.rerun()


def _main_dashboard_inner():
    home_details = st.session_state.home_details
    ss = st.session_state.get("sidebar_settings", {})
    tariff_rate = home_details.get("tariff_rate", 8.0)
    peak_morning = ss.get("peak_morning", (6, 10))
    peak_evening = ss.get("peak_evening", (18, 22))
    display_count = ss.get("display_count", 100)
    cal_month = ss.get("cal_month", pd.Timestamp.now().month)
    cal_year = ss.get("cal_year", pd.Timestamp.now().year)
    sf = compute_scaling_factor(home_details)

    st.markdown(f"""
    <div class="hero">
        <div class="hero-glow"></div>
        <div class="hero-badge">
            <div style="width:6px;height:6px;background:#8FB882;border-radius:50%;animation:gp 2s ease-in-out infinite;"></div>
            {T('hero_badge')}
        </div>
        <h1 class="hero-title">{T('hero_title_a')}
            <span>{T('hero_title_b')}</span>
        </h1>
        <p class="hero-sub">{T('hero_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(T("spinner_preparing")):
        raw_data = load_data_cached()
        xgb_model = load_xgb()
        lstm_model = load_lstm()
        scaler = load_scaler()
        meta = load_meta()
        forecast_df = load_forecast()

    full_data = remap_to_current_dates(raw_data, last_n_days=90)
    if not forecast_df.empty:
        forecast_df = forecast_df.copy()
        forecast_df["datetime"] = forecast_df["datetime"] + (
            full_data["datetime"].max() - raw_data["datetime"].max()
        )

    full_data = full_data.copy()
    full_data["Global_active_power"] = full_data["Global_active_power"] * sf
    if not forecast_df.empty:
        forecast_df["predicted_kw"] = forecast_df["predicted_kw"] * sf
        forecast_df["predicted_kwh"] = forecast_df["predicted_kwh"] * sf

    sim = get_simulator()
    hybrid_mae = meta.get("hybrid_mae", 0.15)

    live_row = sim.get_live_row_for_current_time(full_data)
    if live_row is None:
        live_row = sim.latest_row

    rows_played, total_rows = sim.progress
    progress_pct = rows_played / total_rows if total_rows > 0 else 0

    if live_row:
        gap = live_row.get(TARGET, 0)
        st.markdown(f"""
        <div class="live-bar">
            <div class="live-dot"></div>
            <span class="live-label">{T('live_label')}</span>
            <span style="color:rgba(0,0,0,0.3);">&middot;</span>
            <span style="color:#6B7280;">{T('live_updated')}</span>
            <div style="margin-left:auto;display:flex;align-items:center;gap:8px;">
                <div class="data-progress" style="width:80px;" title="{rows_played:,} / {total_rows:,}">
                    <div class="data-progress-fill" style="width:{progress_pct*100:.0f}%;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(T("getting_ready"))
        return

    scaled_live_row = live_row.copy()
    scaled_live_row[TARGET] = live_row.get(TARGET, 0) * sf

    replay_window_raw = sim.get_window(WINDOW_SIZE + 10)
    if not replay_window_raw.empty:
        replay_window = replay_window_raw.copy()
        replay_window["Global_active_power"] = replay_window["Global_active_power"] * sf
    else:
        replay_window = replay_window_raw

    pred_kw, xgb_p, lstm_p = predict_next_period(
        scaled_live_row, xgb_model, lstm_model, scaler, replay_window
    )
    xgb_p *= sf
    lstm_p *= sf
    uncertainty = hybrid_mae * sf
    lower = max(0, pred_kw - uncertainty)
    upper = pred_kw + uncertainty

    history = sim.get_full_history()
    if not history.empty:
        history = history.copy()
        history["Global_active_power"] = history["Global_active_power"] * sf
        latest_date = history["datetime"].dt.date.iloc[-1]
        today_info = daily_cost(history, str(latest_date), tariff_rate,
                                tuple(peak_morning), tuple(peak_evening))
        week_start_date = latest_date - pd.Timedelta(days=6)
        week_info = weekly_cost(history, str(week_start_date), tariff_rate,
                                tuple(peak_morning), tuple(peak_evening))
        nm_info = next_month_cost(forecast_df, tariff_rate,
                                  tuple(peak_morning), tuple(peak_evening))
    else:
        today_info = {"total_cost": 0, "total_kwh": 0}
        week_info = {"total_cost": 0, "pct_change": 0}
        nm_info = {"total_cost": 0, "month": "N/A"}

    tab_overview, tab_analysis, tab_optimize, tab_bills, tab_trends = st.tabs([
        T("tab_overview"), T("tab_analysis"), T("tab_save"), T("tab_bills"), T("tab_trends")
    ])

    with tab_overview:
        pct_change = week_info.get("pct_change", 0)
        if pct_change > 2:
            week_trend = (T("trend_vs_week", pct=f"+{pct_change:.1f}"), "up")
        elif pct_change < -2:
            week_trend = (T("trend_vs_week", pct=f"{pct_change:.1f}"), "down")
        else:
            week_trend = (T("steady"), "neutral")

        c1, c2, c3 = st.columns(3)
        metric_card(c1, label=T("card_current_usage"), value=f"{gap:.2f}", unit="kW",
                    sub=T("card_current_sub"), trend_text=T("steady"),
                    trend_dir="neutral", css_class="mc-green")
        metric_card(c2, label=T("card_next_hour"), value=f"{pred_kw:.2f}", unit="kW",
                    sub=T("card_next_hour_sub", lo=f"{lower:.2f}", hi=f"{upper:.2f}"),
                    trend_text=T("card_smart_forecast"), trend_dir="neutral",
                    css_class="mc-teal")
        metric_card(c3, label=T("card_today_cost"),
                    value=f"Rs. {fmt_rs(today_info['total_cost'])}",
                    sub=T("card_kwh_today", kwh=fmt_kwh(today_info['total_kwh'])),
                    trend_text="", trend_dir="neutral", css_class="mc-orange")

        c4, c5 = st.columns(2)
        metric_card(c4, label=T("card_this_week"),
                    value=f"Rs. {fmt_rs(week_info['total_cost'])}",
                    sub="", trend_text=week_trend[0], trend_dir=week_trend[1],
                    css_class="mc-purple")
        metric_card(c5, label=T("card_next_month"),
                    value=f"Rs. {fmt_rs(nm_info['total_cost'])}",
                    sub=T("card_projected_for", month=nm_info['month']),
                    trend_text="", trend_dir="neutral", css_class="mc-rose")

        st.markdown("")
        section(T("sec_how_much"), "\u26a1")

        c1, c2 = st.columns([1, 1])
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=pred_kw,
                number=dict(suffix=" kW", font=dict(size=36, color="#1A1A1A")),
                gauge=dict(
                    axis=dict(range=[0, max(2, pred_kw * 1.5)], tickfont=dict(color="#6B7280")),
                    bar=dict(color=ACCENT), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[
                        dict(range=[0, lower], color="rgba(74,103,65,0.08)"),
                        dict(range=[lower, upper], color="rgba(74,103,65,0.15)"),
                        dict(range=[upper, max(2, pred_kw * 1.5)], color="rgba(180,80,80,0.05)"),
                    ],
                    threshold=dict(line=dict(color="#C4944A", width=2), thickness=0.75, value=gap),
                ),
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT(height=260, margin=dict(l=30, r=30, t=10, b=10)))
            st.plotly_chart(fig_gauge, width="stretch")

        with c2:
            confidence = max(0, min(100, (1 - uncertainty / max(pred_kw, 0.01)) * 100))
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence,
                number=dict(suffix="%", font=dict(size=40, color="#1A1A1A")),
                title=dict(text=T("card_smart_forecast"), font=dict(size=14, color="#6B7280")),
                gauge=dict(
                    axis=dict(range=[0, 100]), bar=dict(color=ACCENT2),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[
                        dict(range=[0, 60], color="rgba(180,80,80,0.08)"),
                        dict(range=[60, 85], color="rgba(196,148,74,0.1)"),
                        dict(range=[85, 100], color="rgba(74,103,65,0.1)"),
                    ],
                    threshold=dict(line=dict(color=ACCENT, width=2), thickness=0.75, value=confidence),
                ),
            ))
            fig_conf.update_layout(**PLOTLY_LAYOUT(height=260, margin=dict(l=30, r=30, t=40, b=10)))
            st.plotly_chart(fig_conf, width="stretch")

    with tab_analysis:
        c_left, c_right = st.columns(2)
        with c_left:
            section(T("sec_power_goes"), "\U0001f4cb")
            period = st.radio(T("show_me"), [T("radio_now"), T("radio_7d"), T("radio_30d")],
                              horizontal=True, key="app_period")
            if period == T("radio_now") and not history.empty:
                app_start = history["datetime"].iloc[0]
                app_end = history["datetime"].iloc[-1]
            elif period == T("radio_7d") and not history.empty:
                app_end = history["datetime"].iloc[-1]
                app_start = app_end - pd.Timedelta(days=7)
            elif period == T("radio_30d") and not history.empty:
                app_end = history["datetime"].iloc[-1]
                app_start = app_end - pd.Timedelta(days=30)
            else:
                app_start = app_end = pd.Timestamp.now()
            app_df = get_appliance_breakdown(full_data, str(app_start.date()), str(app_end.date()))
            if not app_df.empty and app_df["Wh"].sum() > 0:
                fig_pie = px.pie(app_df, names="Appliance", values="Wh",
                    color_discrete_sequence=["#4A6741", "#6B8F5E", "#C4944A", "#9CA3AF"], hole=0.55)
                total_wh = app_df["Wh"].sum()
                fig_pie.update_layout(**PLOTLY_LAYOUT(
                    height=340, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                    annotations=[dict(text=f"{total_wh/1000:.1f}<br>kWh", x=0.5, y=0.5,
                        font=dict(size=16, color="#1A1A1A"), showarrow=False)]))
                st.plotly_chart(fig_pie, width="stretch")
                st.dataframe(app_df.style.format({"Wh": "{:.1f}", "Pct": "{:.1f}%"}),
                             width="stretch", hide_index=True)
            else:
                st.info(T("info_no_breakdown"))
        with c_right:
            section(T("sec_peak_offpeak"), "\U0001f4b0")
            if not history.empty:
                hd = history.copy()
                hd["hour"] = hd["datetime"].dt.hour
                hd["is_peak"] = hd["hour"].apply(
                    lambda h: (peak_morning[0] <= h < peak_morning[1]) or
                              (peak_evening[0] <= h < peak_evening[1])
                )
                hd["cost"] = hd[TARGET] * tariff_rate
                peak_cost = hd.loc[hd["is_peak"], "cost"].sum()
                off_cost = hd.loc[~hd["is_peak"], "cost"].sum()
                cost_df = pd.DataFrame({T("df_column_period"): [T("peak_hours_lbl"), T("offpeak_hours_lbl")], "Cost": [peak_cost, off_cost]})
                fig_cost = px.pie(cost_df, names="Period", values="Cost",
                    color_discrete_sequence=["#B45050", "#4A6741"], hole=0.55)
                fig_cost.update_layout(**PLOTLY_LAYOUT(height=340, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)))
                st.plotly_chart(fig_cost, width="stretch")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric(T("peak_hours_lbl"), f"Rs. {fmt_rs(peak_cost)}")
                pc2.metric(T("offpeak_hours_lbl"), f"Rs. {fmt_rs(off_cost)}")
                pc3.metric(T("total_lbl"), f"Rs. {fmt_rs(peak_cost + off_cost)}")
            else:
                st.info(T("info_no_peak"))

    with tab_optimize:
        section(T("sec_ways_save"), "\U0001f4a1")
        if not history.empty and len(history) > 100:
            anomalies = detect_anomalies(history)
            tips = generate_anomaly_tips(anomalies, tariff_rate, translate_fn=T)
            for tip in tips:
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
            if not anomalies.empty:
                with st.expander(T("expander_anomalies", n=len(anomalies))):
                    st.dataframe(anomalies[["datetime", "hour", "Global_active_power",
                                             "rolling_mean", "threshold", "top_appliance"]],
                                 width="stretch", hide_index=True)
        else:
            st.markdown(f'<div class="tip-box">{T("msg_still_collecting")}</div>',
                       unsafe_allow_html=True)
        if not forecast_df.empty:
            section(T("sec_plan_ahead"), "\U0001f4c5")
            cal_tip_list = calendar_tips(forecast_df, tariff_rate, translate_fn=T)
            for tip in cal_tip_list:
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
        section(T("sec_savings"), "\U0001f4b0")
        if not full_data.empty:
            reduction = st.slider(T("slider_reduction"), 0, 50, 0)
            wi = whatif_simulator(full_data, reduction, tariff_rate)
            daily_cost_val = wi['original_daily_kwh'] * tariff_rate
            reduced_daily_cost = wi['reduced_daily_kwh'] * tariff_rate
            daily_saved = daily_cost_val - reduced_daily_cost
            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric(T("m_daily_usage"), f"{fmt_kwh(wi['original_daily_kwh'])} kWh",
                       T("m_daily_cost_delta", rs=fmt_rs(daily_cost_val)), delta_color="off")
            if reduction > 0:
                wc2.metric(T("m_after_reduction", pct=reduction), f"{fmt_kwh(wi['reduced_daily_kwh'])} kWh",
                           T("m_saved_delta", rs=fmt_rs(daily_saved)))
            else:
                wc2.metric(T("m_after_reduction", pct=reduction), f"{fmt_kwh(wi['reduced_daily_kwh'])} kWh")
            wc3.metric(T("m_monthly_savings"), f"Rs. {fmt_rs(wi['monthly_saving'])}")
            wc4.metric(T("m_yearly_savings"), f"Rs. {fmt_rs(wi['monthly_saving'] * 12)}")
            compare_df = pd.DataFrame({
                T("df_column_scenario"): [T("scenario_current"), T("scenario_with_savings", pct=reduction)],
                T("yaxis_monthly_cost"): [wi["original_monthly_cost"], wi["reduced_monthly_cost"]],
            })
            fig_compare = go.Figure(go.Bar(
                x=compare_df[T("df_column_scenario")], y=compare_df[T("yaxis_monthly_cost")],
                marker=dict(color=["#B45050", "#4A6741"], line=dict(width=0)),
                text=[f"Rs. {v:,.0f}" for v in compare_df[T("yaxis_monthly_cost")]],
                textposition="outside", textfont=dict(color="#4A5568", size=13),
            ))
            fig_compare.update_layout(**PLOTLY_LAYOUT(height=300, yaxis_title=T("yaxis_monthly_cost")))
            st.plotly_chart(fig_compare, width="stretch")

    with tab_bills:
        render_bills_tab()

    with tab_trends:
        section(T("sec_trends"), "\U0001f4c8")
        # Build the plotted window straight from the full remapped dataset so
        # the chart is populated immediately (the replay simulator replays this
        # exact dataset, so its timestamps align — no dependency on how many
        # rows the simulator has reached).
        plot_df = pd.DataFrame()
        if not full_data.empty:
            plot_df = full_data.tail(display_count)[["datetime", "Global_active_power"]].copy()
            plot_df["Global_active_power"] = plot_df["Global_active_power"].astype(float) * sf
            plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
            # Bin to a clean hourly grid: one point per hour, no duplicate x labels.
            plot_df = (plot_df.assign(hour_bin=plot_df["datetime"].dt.floor("h"))
                              .groupby("hour_bin", as_index=False)["Global_active_power"]
                              .mean()
                              .rename(columns={"hour_bin": "datetime"})
                              .sort_values("datetime")
                              .reset_index(drop=True))
            # Guard against any nulls sneaking in from bad rows.
            plot_df = plot_df.dropna(subset=["datetime", "Global_active_power"])

        if len(plot_df) >= 2:
            sparse = len(plot_df) < 40
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=plot_df["datetime"], y=plot_df["Global_active_power"],
                name=T("legend_power"), mode="lines+markers" if sparse else "lines",
                line=dict(color=ACCENT, width=2, shape="spline", smoothing=0.6),
                marker=dict(size=6, color=ACCENT) if sparse else None,
                fill="tozeroy", fillcolor="rgba(74,103,65,0.08)",
                hovertemplate="%{x|%a, %b %d %I:%M %p}<br>%{y:.2f} kW<extra>" + T("legend_power") + "</extra>",
            ))
            if len(plot_df) >= 5:
                rolling = plot_df["Global_active_power"].rolling(5, min_periods=1).mean()
                fig_trend.add_trace(go.Scatter(
                    x=plot_df["datetime"], y=rolling,
                    name=T("legend_avg"), mode="lines",
                    line=dict(color="#C4944A", width=2, dash="dash"),
                    hovertemplate="%{x|%a, %b %d %I:%M %p}<br>%{y:.2f} kW<extra>" + T("legend_avg") + "</extra>",
                ))
            time_span = plot_df["datetime"].max() - plot_df["datetime"].min()
            if time_span <= pd.Timedelta(hours=24):
                tick_fmt = "%I %p"          # single day: plain hour labels
            elif time_span <= pd.Timedelta(days=10):
                tick_fmt = "%b %d %I %p"    # crosses midnight: keep date in label (unique)
            else:
                tick_fmt = "%b %d"
            fig_trend.update_layout(**PLOTLY_LAYOUT(
                height=340, xaxis_title=T("x_time"), yaxis_title="kW",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(tickformat=tick_fmt, nticks=10,
                           gridcolor="rgba(0,0,0,0.05)")))
            st.plotly_chart(fig_trend, width="stretch", key="trend_chart")
        else:
            st.markdown(f"""
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div class="empty-state-title">{T('trends_empty_title')}</div>
                <div>{T('trends_empty_msg')}</div>
            </div>
            """, unsafe_allow_html=True)
        section(T("sec_calendar"), "\U0001f4c5")
        cal_year_int = int(cal_year)
        cal_month_int = int(cal_month)
        render_calendar(full_data, forecast_df, cal_year_int, cal_month_int, tariff_rate)
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:12px 16px;margin-top:0.8rem;font-size:0.75rem;color:#6B7280;">
            <span><span style="color:#4A6741;">&#9632;</span> {T('cal_legend_low')}</span>
            <span><span style="color:#C4944A;">&#9632;</span> {T('cal_legend_avg')}</span>
            <span><span style="color:#B45050;">&#9632;</span> {T('cal_legend_high')}</span>
            <span><span style="color:#6B8F5E;">&#9632;</span> {T('cal_legend_future')}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""<div class="disc">
        <p>{T('about_demo')}</p>
    </div>""", unsafe_allow_html=True)


def main():
    if not render_login_screen():
        return
    if not render_onboarding():
        return
    render_sidebar()
    main_dashboard()


if __name__ == "__main__":
    main()
