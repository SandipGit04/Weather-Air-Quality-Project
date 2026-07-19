import streamlit as st
import pandas as pd
import numpy as np
from prophet.serialize import model_from_json
from datetime import date, timedelta
import os
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

from Weather_Logic import (
    feels_like, cloud_coverage, rain_chance, rain_perception,
    weather_condition, wind_gust, uv_index, weather_description, sunrise_sunset
)

# The server this app runs on may be in any timezone (often UTC on cloud
# hosts). Since both forecasting systems are India-focused, "today"/"now"
# must always reflect IST (UTC+5:30), not the server's local clock, or
# default forecast dates, chart windows, and sun-position displays could
# all be off by a day or several hours near midnight.
IST = "Asia/Kolkata"

def today_ist():
    return pd.Timestamp.now(tz="UTC").tz_convert(IST).date()

def now_ist():
    return pd.Timestamp.now(tz="UTC").tz_convert(IST)

# Icon map: condition key -> emoji (kept small and consistent, not oversized)
ICONS = {
    "storm": "\U0001F329", "rain": "\U0001F327", "drizzle": "\U0001F326",
    "haze": "\U0001F32B", "cloudy": "\U00002601", "partly_cloudy": "\U000026C5",
    "sunny": "\U00002600", "clear": "\U0001F31E",
}

# ─────────────────────────────────────────
# ONE Page Config for the whole unified app
# (Streamlit only allows this to be called once per app run, so it is
# centralized here rather than inside either forecasting system.)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="India Forecasting Hub",
    page_icon="\U0001F30D",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Navigation state
# ─────────────────────────────────────────
if "active_system" not in st.session_state:
    st.session_state.active_system = None  # None = show the landing/selector screen

def go_to(system_name):
    st.session_state.active_system = system_name

# ─────────────────────────────────────────
# Shared chrome CSS (landing screen + top switcher only).
# This is purely additive — it does not touch, override, or remove any
# class used inside the AQI or Weather apps' own CSS blocks below; it
# only styles the new landing page and the small switcher bar.
# ─────────────────────────────────────────
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">',
    unsafe_allow_html=True
)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0f1e !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="block-container"] {
    padding: 0.5rem 3rem !important;
    max-width: 1400px;
}
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility: hidden; display: none !important; }

/* ── Landing screen ── */
.hub-hero {
    text-align: center;
    padding: 0.5rem 1.5rem 2.5rem;
}
.hub-hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255, 122, 26, 0.12); border: 1px solid rgba(255, 122, 26, 0.3);
    color: #ff7a1a; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.3rem 0.9rem; border-radius: 999px; margin-bottom: 1.2rem;
}
.hub-hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem; font-weight: 700; color: #f0f9ff;
    margin: 0 0 0.6rem 0; line-height: 1.6;
    text-align: center;
}
.hub-hero-title span { color: #ff7a1a; }
.hub-hero-sub {
    font-size: 1.05rem !important;
    color: #94a3b8 !important;
    max-width: 640px !important;
    margin: 0 auto !important;
    line-height: 1.6 !important;
    text-align: center !important;
    display: block !important;
}

.hub-card {
    background: linear-gradient(160deg, #131a2c 0%, #0d1220 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 2.2rem 2rem 1.8rem;
    height: 100%;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.hub-card::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    pointer-events: none;
}
.hub-card.aqi::before   { background: radial-gradient(circle, rgba(255, 122, 26, 0.10) 0%, transparent 70%); }
.hub-card.weather::before { background: radial-gradient(circle, rgba(255, 122, 26, 0.12) 0%, transparent 70%); }

.hub-card-icon { font-size: 2.6rem; margin-bottom: 1rem; }
.hub-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem; font-weight: 700; color: #f0f9ff; margin-bottom: 0.5rem;
}
.hub-card-desc {
    font-size: 0.9rem; color: #94a3b8; line-height: 1.65; margin-bottom: 1.2rem;
    min-height: 84px;
}
.hub-card-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.4rem; }
.hub-tag {
    font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.7rem;
    border-radius: 999px; color: #94a3b8;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
}
.hub-card.aqi .hub-tag.accent { color: #ff7a1a; border-color: rgba(255, 122, 26, 0.3); background: rgba(255, 122, 26, 0.08); }
.hub-card.weather .hub-tag.accent { color: #ff7a1a; border-color: rgba(255, 122, 26, 0.3); background: rgba(255, 122, 26, 0.08); }

/* Buttons on the landing cards reuse Streamlit's own button. Streamlit
   renders each st.button in its own container that sits AS A SIBLING of
   the card's markdown HTML above it, not nested inside it — so we target
   each button directly via its widget key instead of a CSS ancestor
   relationship that doesn't exist in the real DOM. */
            
div[data-testid="stVerticalBlockBorderWrapper"] { height: 100%; }

.stButton > button {
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 1.4rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    background: linear-gradient(135deg, #ff7a1a, #ea580c) !important;
}

/* Distinct colors for the two landing-page "Enter" buttons and the two
   switcher buttons. Each is wrapped in its own st.container(key=...),
   which Streamlit renders with a matching "st-key-<key>" class — the
   reliable way to target one specific button without relying on a
   parent/child relationship that doesn't exist in the real DOM. */
.st-key-enter_aqi .stButton > button {
    background: linear-gradient(135deg, #ff7a1a, #14b8a6) !important;
}
.st-key-enter_weather .stButton > button {
    background: linear-gradient(135deg, #ff7a1a, #7c3aed) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(255, 122, 26, 0.25) !important;
}

/* ── Top switcher bar (shown inside either forecasting environment) ── */
.switcher-wrap {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; margin-bottom: 0.5rem; flex-wrap: wrap;
}
.switcher-label {
    font-size: 1.5rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #5b6478; white-space: nowrap;
}

/* Segmented-control look for the two switcher buttons, plus vertical
   alignment so "All Systems" sits flush with the two buttons beside it
   (Streamlit columns can otherwise let one button sit slightly higher
   if its container renders with different top spacing). */
.switcher-wrap + div[data-testid="stHorizontalBlock"] {
    align-items: flex-end;
}
.st-key-switch_home,
.st-key-switch_aqi,
.st-key-switch_weather {
    display: flex;
    align-items: flex-end;
    height: 100%;
}
.st-key-switch_aqi .stButton > button,
.st-key-switch_weather .stButton > button {
    background: #1b2942 !important;
    border: 3px solid rgba(255, 122, 26, .20)!important;
    color: #c3c9d6 !important;
    font-size: 0.85rem !important;
    padding: 0.70rem 1.1rem !important;
    box-shadow: inset 0 2px 0 rgba(255,255,255,0.05), 0 4px 12px rgba(0,0,0,0.2) !important;
}
.st-key-switch_aqi .stButton > button:hover,
.st-key-switch_weather .stButton > button:hover {
    border-color: #ffb84d !important;
    color: #dbe1ec !important;
    transform: none !important;
    box-shadow: none !important;
}

@media (max-width: 768px) {
    .hub-hero-title { font-size: 2rem; }
    [data-testid="block-container"] { padding: 1.2rem 1rem !important; }
}
</style>
""", unsafe_allow_html=True)


def render_switcher(current):
    """Small top bar shown inside either environment, letting the user
    jump straight to the other forecasting system or back to the
    landing/selector screen — purely additive navigation chrome."""
    st.markdown('<div class="switcher-wrap"><div class="switcher-label">India Forecasting Hub</div></div>', unsafe_allow_html=True)

    # st.container(key=...) renders its own wrapper div with a matching
    # "st-key-<key>" class, but any markup we st.markdown() *inside* that
    # container becomes a separate sibling element, not part of the same
    # class list — so a plain "active" marker div can't combine classes
    # with the container. Instead, inject one tiny conditional style rule
    # naming whichever key is currently active.
    active_key = f"switch_{current}" if current in ("aqi", "weather") else None
    if active_key:
        active_color = (
            "linear-gradient(135deg, #ff7a1a, #14b8a6)" if current == "aqi"
            else "linear-gradient(135deg, #ff7a1a, #7c3aed)"
        )
        st.markdown(f"""
        <style>
        .st-key-{active_key} .stButton > button {{
            background: {active_color} !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
        }}
        </style>
        """, unsafe_allow_html=True)

    c_home, c_aqi, c_wx = st.columns([1, 1.6, 1.6])
    with c_home:
        with st.container(key="switch_home"):
            if st.button("\u2190 All Systems", key="switch_home_btn", use_container_width=True):
                go_to(None)
                st.rerun()
    with c_aqi:
        with st.container(key="switch_aqi"):
            if st.button("\U0001F30D AQI Forecasting", key="switch_aqi_btn", use_container_width=True):
                go_to("aqi")
                st.rerun()
    with c_wx:
        with st.container(key="switch_weather"):
            if st.button("\U0001F326\uFE0F Weather Forecasting", key="switch_weather_btn", use_container_width=True):
                go_to("weather")
                st.rerun()


def render_landing_page():
    st.markdown("""
    <div class="hub-hero">
        <div class="hub-hero-badge">\U0001F1EE\U0001F1F3 India Environmental Intelligence</div>
        <h1 class="hub-hero-title">Choose Your <span>Forecasting System</span></h1>
        <p class="hub-hero-sub">
            Two Prophet-powered time-series forecasting environments for Indian cities.
            Pick a system below to get started — you can switch between them anytime.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="hub-card aqi">
            <div class="hub-card-icon">\U0001F30D</div>
            <div class="hub-card-title">AQI Forecasting System</div>
            <div class="hub-card-desc">
                Predict air quality index up to 365 days ahead for Indian cities.
                Includes pollution category breakdown, confidence bands, and
                monthly seasonal AQI patterns.
            </div>
            <div class="hub-card-tags">
                <span class="hub-tag accent">Air Quality Index</span>
                <span class="hub-tag">PM2.5</span>
                <span class="hub-tag">365-day horizon</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(key="enter_aqi"):
            if st.button("Enter AQI Forecasting \u2192", key="enter_aqi_btn", use_container_width=True):
                go_to("aqi")
                st.rerun()

    with col_b:
        st.markdown("""
        <div class="hub-card weather">
            <div class="hub-card-icon">\U0001F326\uFE0F</div>
            <div class="hub-card-title">Weather Forecasting System</div>
            <div class="hub-card-desc">
                A 6-day rolling weather outlook with hourly temperature curves,
                rain probability, wind, pressure, UV index, and real-time
                sunrise/sunset tracking for Indian cities.
            </div>
            <div class="hub-card-tags">
                <span class="hub-tag accent">Temperature</span>
                <span class="hub-tag">Rain &amp; Wind</span>
                <span class="hub-tag">6-day rolling</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(key="enter_weather"):
            if st.button("Enter Weather Forecasting \u2192", key="enter_weather_btn", use_container_width=True):
                go_to("weather")
                st.rerun()


# =====================================
# AQI Forecasting
# =====================================

def render_aqi_app():
    # Read the dataset from the specified path
    weather_df = pd.read_csv("Datasets/Forecasting_Data.csv")

    # ─────────────────────────────────────────
    # Page Config
    # ─────────────────────────────────────────

    # ─────────────────────────────────────────
    # Custom CSS — Dark Orange Theme
    # ─────────────────────────────────────────
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0f1e !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] > .main {
        background-color: #0a0f1e !important;
    }

    [data-testid="block-container"] {
        padding: 2rem 3rem !important;
        max-width: 1400px;
    }

    /* ── Header ── */
    .hero-header {
        background: linear-gradient(135deg, #0d2137 0%, #0a1628 50%, #071020 100%);
        border: 1px solid rgba(255, 122, 26, 0.15);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(255, 122, 26, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -80px; left: 30%;
        width: 300px; height: 200px;
        background: radial-gradient(ellipse, rgba(234, 88, 12, 0.05) 0%, transparent 70%);
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #f0f9ff;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-title .accent { color: #ff7a1a; }

    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 0.01em;
        margin: 0;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255, 122, 26, 0.12);
        border: 1px solid rgba(255, 122, 26, 0.3);
        color: #ff7a1a;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }

    /* ── Control Panel ── */
    /* ============================================================ */
    /* ── NEW GLASSMORPHISM DASHBOARD METRIC CARDS ADDED HERE ── */
    /* ============================================================ */
    .glass-grid{
        display:grid;
        grid-template-columns:repeat(6,1fr);
        gap:18px;
        margin-top:10px;
    }

    .glass-card {
        background:linear-gradient(180deg,#13233d,#0d182c);
        border:3px solid rgba(255,255,255,.08);
        border-radius:16px;
        padding:22px 16px;
        transition:.3s;
        height:220px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        box-shadow:0 0 18px rgba(0,0,0,.25);
    }

    .glass-card:hover{
        transform:translateY(-8px);
        box-shadow:0 0 30px rgba(255, 122, 26, .15);
        border-color:#ff7a1a;
    }

    .icon-circle {
        width: 50px; height: 50px;
        border-radius: 50%;
        display: flex; justify-content: center; align-items: center;
        font-size: 1rem;
        margin-bottom: 20px;
        background: transparent;
        border: 2px solid;
    }

    .glass-title {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.5px; margin-bottom: 12px;
    }

    .glass-value {
        font-size: 1.7rem; font-weight: 700; margin-bottom: 16px; color: #ffffff;
    }

    .glass-badge {
        font-size: 0.8rem; padding: 4px 12px; border-radius: 20px;
        margin-bottom: 12px; font-weight: 600;
    }

    .glass-footer {
        margin-top: auto; font-size: 0.8rem; padding: 6px 12px;
        border-radius: 20px; display: flex; align-items: center; gap: 6px;
        background-color: rgba(0, 0, 0, 0.2); border: 1.2px solid;
    }

    /* Theme Colors for Glass Cards */
    .g-teal { border-color: rgba(255,105,0,0.3); box-shadow: 0 0 20px rgba(255,105,0,0.05), inset 0 0 10px rgba(255,105,0,0.05); }
    .g-teal .icon-circle { border-color: #ff6900; color: #ff6900; box-shadow: 0 0 10px rgba(255,105,0,0.2); }
    .g-teal .glass-title, .g-teal .glass-footer { color: #ff6900; }
    .g-teal .glass-footer { border-color: rgba(255,105,0,0.2); }

    .g-blue { border-color: rgba(52, 152, 219, 0.3); box-shadow: 0 0 20px rgba(52, 152, 219, 0.05), inset 0 0 10px rgba(52, 152, 219, 0.05); }
    .g-blue .icon-circle { border-color: #3498db; color: #3498db; box-shadow: 0 0 10px rgba(52, 152, 219, 0.2); }
    .g-blue .glass-title, .g-blue .glass-footer { color: #3498db; }
    .g-blue .glass-footer { border-color: rgba(52, 152, 219, 0.2); }

    .g-purple { border-color: rgba(155, 89, 182, 0.4); box-shadow: 0 0 20px rgba(155, 89, 182, 0.05), inset 0 0 10px rgba(155, 89, 182, 0.1); }
    .g-purple .icon-circle { border-color: #9b59b6; color: #9b59b6; box-shadow: 0 0 10px rgba(155, 89, 182, 0.2); }
    .g-purple .glass-title, .g-purple .glass-footer { color: #9b59b6; }
    .g-purple .glass-footer { border-color: rgba(155, 89, 182, 0.2); }

    .g-orange { border-color: rgba(243, 156, 18, 0.4); box-shadow: 0 0 20px rgba(243, 156, 18, 0.05), inset 0 0 10px rgba(243, 156, 18, 0.1); }
    .g-orange .icon-circle { border-color: #f39c12; color: #f39c12; box-shadow: 0 0 10px rgba(243, 156, 18, 0.2); }
    .g-orange .glass-title, .g-orange .glass-footer { color: #f39c12; }
    .g-orange .glass-value { margin-bottom: 8px; }
    .g-orange .glass-footer { border-color: rgba(243, 156, 18, 0.2); }

    .g-green { border-color: rgba(46, 204, 113, 0.3); box-shadow: 0 0 20px rgba(46, 204, 113, 0.05), inset 0 0 10px rgba(46, 204, 113, 0.05); }
    .g-green .icon-circle { border-color: #2ecc71; color: #2ecc71; box-shadow: 0 0 10px rgba(46, 204, 113, 0.2); }
    .g-green .glass-title { color: #2ecc71; }
    .g-green .glass-value { margin-bottom: 12px; }
    .g-green .glass-badge { background-color: rgba(46, 204, 113, 0.15); color: #2ecc71; border: 1px solid rgba(46, 204, 113, 0.3); margin-top: auto;}

    .g-lblue { border-color: rgba(9, 132, 227, 0.3); box-shadow: 0 0 20px rgba(9, 132, 227, 0.05), inset 0 0 10px rgba(9, 132, 227, 0.05); }
    .g-lblue .icon-circle { border-color: #0984e3; color: #0984e3; box-shadow: 0 0 10px rgba(9, 132, 227, 0.2); }
    .g-lblue .glass-title, .g-lblue .glass-footer { color: #0984e3; }
    .g-lblue .glass-footer { border-color: rgba(9, 132, 227, 0.2); }

    .panel-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 1rem;
    }

    /* ==================
        SELECTBOX
       ==================*/

    .stSelectbox div[data-baseweb="select"]{
        border-radius:16px !important;
        overflow:hidden !important;
    }

    /* Main Box */
    .stSelectbox div[data-baseweb="select"] > div{
        background:#1b2942 !important;
        border:3px solid rgba(255, 122, 26, .20) !important;
        border-radius:16px !important;
        min-height:54px !important;
        padding:0 14px !important;
        display:flex !important;
        align-items:center !important;
        transition:all .25s ease;
    }

    /* Hover */
    .stSelectbox div[data-baseweb="select"] > div:hover{
        border:3px solid #ff7a1a !important;
        box-shadow:
            0 0 0 2px rgba(255, 122, 26, .18),
            0 0 16px rgba(255, 122, 26, .30);

    }

    /* Selected */
    .stSelectbox div[data-baseweb="select"] > div:focus-within{
        border:2px solid #ff7a1a !important;
        box-shadow:
            0 0 0 2px rgba(255, 122, 26, .18),
            0 0 18px rgba(255, 122, 26, .35);
    }

    /* Text */
    .stSelectbox div[data-baseweb="select"] input{
        color:#ffffff !important;
        font-size:15px !important;
        font-weight:600 !important;
        caret-color:#ff7a1a !important;
    }

    /* Dropdown Arrow */
    .stSelectbox svg{
        color:#ff7a1a !important;
        transition:.25s;
    }

    /* Optional */
    .stSelectbox svg:hover{color:#ffb066 !important;}

    /* Selected Value */
    .stSelectbox div[data-baseweb="select"] span{
        font-weight:700 !important;
        font-size:20px !important;
        color:#ffffff !important;
    }

    /* Backup selector for newer BaseWeb versions */
    .stSelectbox div[data-baseweb="select"] div{
        font-weight:700 !important;
    }

    /* ==================
        DATE INPUT
       ================== */

    .stDateInput > div{
        border-radius:16px !important;
        overflow:hidden !important;
    }

    /* Main Box */
    .stDateInput > div > div{
        background:#1b2942 !important;
        border:3px solid rgba(255, 122, 26, .20) !important;
        border-radius:16px !important;
        min-height:54px !important;
        padding:0 14px !important;
        display:flex !important;
        align-items:center !important;
        transition:all .25s ease;
    }

    /* Hover */
    .stDateInput > div > div:hover{
        border:3px solid #ff7a1a !important;
        box-shadow:
            0 0 0 2px rgba(255, 122, 26, .18),
            0 0 16px rgba(255, 122, 26, .30);
    }

    /* Focus */
    .stDateInput > div > div:focus-within{
        border:2px solid #ff7a1a !important;
        box-shadow:
            0 0 0 2px rgba(255, 122, 26, .18),
            0 0 16px rgba(255, 122, 26, .35);
    }

    /* Text */
    .stDateInput input{
        background:transparent !important;
        color:#ffffff !important;
        font-size:15px !important;
        font-weight:600 !important;
        padding:0 !important;
        caret-color:#ff7a1a !important;
    }

    /* Calendar Button */
    .stDateInput button{
        background:transparent !important;
        color:#ff7a1a !important;
        border:none !important;
        transition:.25s;
    }

    .stDateInput button:hover{
        color:#ffb066 !important;
    }

    /* ==================
        CALENDAR POPUP
       ================== */

    div[role="dialog"]{
        background:#121d30 !important;
        border:1px solid rgba(255,255,255,.08) !important;
        border-radius:18px !important;
    }

    div[role="dialog"] *{
        color:white !important;
    }

    div[role="dialog"] button{
        background:transparent !important;
        color:white !important;
    }

    div[role="dialog"] button:hover{
        background:#1d2d48 !important;
    }

    div[role="dialog"] [aria-selected="true"]{
        background:#ff7a1a !important;
        border-radius:50% !important;
    }

    /* ── Button ── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ff7a1a, #ea580c) !important;
        color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.04em !important;
        text-shadow: 0 1px 2px rgba(0,0,0,.25) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        cursor: pointer !important;
        transition: all 0.25s ease !important;
        margin-top: 0.5rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255, 122, 26, 0.3) !important;
    }

    /* ── AQI Result Banner ── */
    .aqi-banner {
        border-radius: 14px;
        padding: 1.2rem 1.8rem;
        margin: 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #0d1b2e;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 8px;
        border-radius: 16px 16px 0 0;
    }

    .metric-card.teal::before  { background: linear-gradient(90deg, #ff6900, #f54900); }
    .metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #ef4444); }
    .metric-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
    .metric-card.green::before { background: linear-gradient(90deg, #22c55e, #10b981); }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #f0f9ff;
        line-height: 1;
        margin: 0.3rem 0 0.2rem;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.2rem;
    }

    .metric-sub {
        font-size: 0.82rem;
        color: #475569;
    }

    /* ── Section Title ── */
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #cbd5e1;
        margin: 1.8rem 0 1rem;
        padding-left: 0.8rem;
        border-left: 3px solid #ff7a1a;
    }

    /* ── Info Cards ── */
    .info-strip {
        background: rgba(255, 122, 26, 0.05);
        border: 1px solid rgba(255, 122, 26, 0.12);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-top: 1.5rem;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* ── AQI Legend ── */
    .legend-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }

    .legend-chip {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: #94a3b8;
        font-weight: 500;
    }

    .legend-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── stMetric override ── */
    [data-testid="metric-container"] {
        background: transparent !important;
    }

    /* ── Streamlit success/error overrides ── */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    /* ── Hide default streamlit UI chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
        margin: 2rem 0;
    }

    /* Shown only on small/touch screens, right under charts that scroll */
    .swipe-hint {
        display: none;
        font-size: 11px;
        color: #64748b;
        text-align: center;
        margin: -8px 0 16px;
    }

    /* Prevent any oversized element from pushing the whole page sideways
       on phones/tablets — content that needs extra width scrolls within
       its own box instead. */
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
    }

    /* ═══════════════════════════════════════════════════════
       RESPONSIVE — Laptops, Tablets, Smartphones
       (Additive only — does not modify any rule above)
       ═══════════════════════════════════════════════════════ */

    /* Safety net: force Streamlit's native columns (city/date/button,
       the 4 metric cards, etc.) to stack cleanly on narrow screens,
       even on older Streamlit versions that don't do this on their own. */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            row-gap: 0.8rem !important;
        }
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }

    /* Small laptops / large tablets */
    @media (max-width: 1200px) {
        [data-testid="block-container"] {
            padding: 1.6rem 1.6rem !important;
        }
        .glass-grid {
            grid-template-columns: repeat(3, 1fr);
        }
    }

    /* Tablets (portrait) */
    @media (max-width: 900px) {
        .hero-header { padding: 1.8rem 1.6rem; }
        .hero-title  { font-size: 1.8rem; }
        .glass-grid  { grid-template-columns: repeat(2, 1fr); gap: 14px; }
        .glass-card  { height: auto; min-height: 180px; padding: 18px 12px; }
        .metric-value { font-size: 2.1rem; }

        /* Charts scroll instead of squeezing their labels together */
        [data-testid="stPlotlyChart"] { overflow-x: auto; }
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {
            min-width: 680px;
        }
        .swipe-hint { display: block; }
    }

    /* Smartphones */
    @media (max-width: 640px) {
        [data-testid="block-container"] {
            padding: 1.1rem 0.9rem !important;
        }
        .hero-header   { padding: 1.4rem 1.1rem; border-radius: 14px; }
        .hero-title    { font-size: 1.4rem; }
        .hero-subtitle { font-size: 0.85rem; }
        .hero-badge    { font-size: 0.62rem; padding: 0.2rem 0.6rem; }

        .glass-grid  { grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .glass-card  { height: auto; min-height: 150px; padding: 14px 8px; }
        .glass-value { font-size: 1.3rem; margin-bottom: 8px; }
        .glass-title { font-size: 0.62rem; }
        .icon-circle { width: 38px; height: 38px; margin-bottom: 12px; }

        .metric-card  { padding: 1.1rem 1rem; }
        .metric-value { font-size: 1.7rem; }

        .aqi-banner    { flex-wrap: wrap; padding: 1rem 1.2rem; }
        .section-title { font-size: 0.95rem; }
        .legend-row    { gap: 0.4rem; }

        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {
            min-width: 740px;
        }
    }

    /* Small phones */
    @media (max-width: 380px) {
        .glass-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.2rem; }
    }
    </style>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # AQI Helper Functions
    # ─────────────────────────────────────────
    def get_aqi_category(aqi):
        if aqi <= 50:   return "Good",        "#22c55e", "🟢", "Air quality is satisfactory."
        elif aqi <= 100: return "Satisfactory","#84cc16", "🟡", "Acceptable air quality."
        elif aqi <= 200: return "Moderate",    "#f59e0b", "🟠", "Sensitive groups may be affected."
        elif aqi <= 300: return "Poor",        "#ef4444", "🔴", "Health effects for everyone."
        else:            return "Severe",      "#9333ea", "🟣", "Emergency conditions."

    AQI_THRESHOLDS = [
        ("Good",         50,  "#22c55e"),
        ("Satisfactory", 100, "#84cc16"),
        ("Moderate",     200, "#f59e0b"),
        ("Poor",         300, "#ef4444"),
        ("Severe",       500, "#9333ea"),
    ]

    # ─────────────────────────────────────────
    # Load Cities
    # ─────────────────────────────────────────
    model_folder = "aqi_models"
    cities = sorted([
        f.replace("_forecast.json", "")
        for f in os.listdir(model_folder)
        if f.endswith("_forecast.json")
    ])

    city = st.session_state.get('city', cities[0] if cities else "Demo City")

    @st.cache_resource(show_spinner=False)
    def load_city_model(city_name):
        """Load a city's trained Prophet model (cached — each city's model
        file is only read from disk once per app session)."""
        model_path = f"{model_folder}/{city_name}_forecast.json"
        with open(model_path, "r") as fin:
            return model_from_json(fin.read())

    # ─────────────────────────────────────────
    # HERO HEADER
    # ─────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">🛰️ India Air Quality Intelligence</div>
        <h1 class="hero-title">🌍 AQI <span class="accent">Forecasting</span> System</h1>
        <p class="hero-subtitle">
            Prophet-powered time-series forecasting · """ + str(len(cities)) + """ Indian cities · Up to 365-day predictions
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # CONTROL PANEL
    # ─────────────────────────────────────────
    # Dynamic City Information Dashboard
    st.markdown('<div class="panel-label">📍 Configure your forecast</div>', unsafe_allow_html=True)

    col_city, col_date, col_btn = st.columns([2, 2, 1])

    with col_city:
        city = st.selectbox("Select City", cities, label_visibility="visible")

    # Anchor the selectable forecast range to THIS CITY'S actual last known
    # data point (the model's own training history), not to the server's
    # system clock. A model last trained through, say, 15 July can only
    # meaningfully forecast from 16 July onward — regardless of what
    # "today" happens to be if the data hasn't been refreshed since.
    last_known_date = load_city_model(city).history["ds"].max().date()

    with col_date:
        min_date = last_known_date + timedelta(days=1)
        max_date = last_known_date + timedelta(days=365)
        # Default to the real current IST date when it's actually forecastable
        # (i.e. the model isn't so stale that "today" is still in the past
        # relative to its training data); otherwise fall back to the first
        # date the model can forecast at all.
        default_date = today_ist()
        real_today = today_ist()
        if not (min_date <= default_date <= max_date):
            default_date = min_date
        selected_date = st.date_input(
            "Select Forecast Date",
            value=default_date,
            min_value=min_date,
            max_value=max_date
        )
        st.caption(f" Today: {real_today.strftime('%d %b %Y')} · "f"📡 Latest available data for {city}: {last_known_date.strftime('%d %b %Y')} · "
                   f"showing forecast for {default_date.strftime('%d %b %Y')}" + (" (next available forecast day)" if default_date != real_today else ""))

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        forecast_clicked = st.button("💨 Forecast AQI", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================================
    # CITY ANALYTICS DASHBOARD
    # ==========================================================

    city_info = weather_df[weather_df["City"] == city]

    avg_temp = city_info["Temperature"].mean()
    avg_humidity = city_info["Humidity"].mean()
    avg_pm25 = city_info["PM25"].mean()
    avg_aqi = city_info["AQI"].mean()
    records = len(city_info)

    dominant_weather = city_info["WeatherCondition"].mode().iloc[0]
    category, cat_color, _, _ = get_aqi_category(avg_aqi)

    st.markdown(f"""
    <div style="
    background:linear-gradient(180deg,#0f1c30,#0b1627);
    padding:22px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,.08);
    margin-top:25px;
    margin-bottom:30px;
    ">

    <h3 style="
    margin-top:0;
    margin-bottom:22px;
    color:#ff7a1a;
    font-family:'Space Grotesk';
    font-weight:700;
    ">
    📊 {city} Historical Overview 🌐
    </h3>

    <div class="glass-grid">

    <div class="glass-card g-teal">
    <div class="icon-circle"><i class="fas fa-temperature-high"></i></div>
    <div class="glass-title">Temperature</div>
    <div class="glass-value">{avg_temp:.1f}°C</div>
    <div class="glass-footer">
    <i class="fas fa-chart-line"></i> Historical Average
    </div>
    </div>

    <div class="glass-card g-blue">
    <div class="icon-circle"><i class="fas fa-droplet"></i></div>
    <div class="glass-title">Humidity</div>
    <div class="glass-value">{avg_humidity:.1f}%</div>
    <div class="glass-footer">
    <i class="fas fa-chart-line"></i> Historical Average
    </div>
    </div>

    <div class="glass-card g-purple">
    <div class="icon-circle"><i class="fas fa-smog"></i></div>
    <div class="glass-title">PM2.5</div>
    <div class="glass-value">{avg_pm25:.1f}</div>
    <div class="glass-footer">
    <i class="fas fa-wind"></i> µg/m³
    </div>
    </div>

    <div class="glass-card g-orange">
    <div class="icon-circle">
    <i class="fas fa-gauge-high"></i>
    </div>
    <div class="glass-title">Average AQI</div>
    <div class="glass-value">{avg_aqi:.0f}</div>

    <div class="glass-badge"
    style="
    background:{cat_color};
    color:white;
    ">
    {category}
    </div>

    </div>

    <div class="glass-card g-green">
    <div class="icon-circle">
    <i class="fas fa-cloud"></i>
    </div>

    <div class="glass-title">Weather</div>
    <div class="glass-value"
    style="font-size:1.2rem;">
    {dominant_weather}
    </div>

    <div class="glass-badge">
    Most Frequent
    </div>

    </div>

    <div class="glass-card g-lblue">

    <div class="icon-circle">
    <i class="fas fa-database"></i>
    </div>

    <div class="glass-title">
    Records
    </div>

    <div class="glass-value">
    {records}
    </div>

    <div class="glass-footer">
    <i class="fas fa-calendar"></i>
    {records} Days
    </div>

    </div>

    </div>

    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # AQI SCALE LEGEND
    # ─────────────────────────────────────────
    legend_html = '<div class="legend-row">'
    for name, _, color in AQI_THRESHOLDS:
        legend_html += f'<div class="legend-chip"><div class="legend-dot" style="background:{color}"></div>{name}</div>'
    legend_html += '</div>'
    st.markdown(f'<div class="info-strip">AQI Scale Reference {legend_html}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # FORECAST LOGIC
    # ─────────────────────────────────────────
    if forecast_clicked:
        try:
            model = load_city_model(city)

            with st.spinner(f"Running forecast model for {city}..."):
                future = model.make_future_dataframe(periods=365, freq='D')
                forecast = model.predict(future)

            result = forecast[forecast["ds"].dt.date == selected_date]

            if len(result) > 0:
                predicted_aqi = max(0, float(result["yhat"].iloc[0]))
                lower_bound   = max(0, float(result["yhat_lower"].iloc[0]))
                upper_bound   = max(0, float(result["yhat_upper"].iloc[0]))
                category, color, emoji, desc = get_aqi_category(predicted_aqi)

                # Measured from the model's own last known data point (not the
                # system clock) — always a positive, meaningful "how far out is
                # this prediction" figure, even when the data hasn't been
                # refreshed in the last day or two.
                days_ahead = (selected_date - last_known_date).days

                # ==============================
                # AUTO SCROLL TO RESULTS
                # ==============================
                st.markdown(
                    '<div id="forecast-results"></div>',
                    unsafe_allow_html=True
                )

                components.html("""
                    <script>
                    setTimeout(function(){
                        const result = window.parent.document.getElementById("forecast-results");

                        if(result){
                            result.scrollIntoView({
                                behavior:"smooth",
                                block:"start"
                            });
                        }
                    },50);
                    </script>
                    """,
                    height=0,
                )

                # ── Result Banner ──
                st.markdown(f"""
                <div class="aqi-banner" style="background: linear-gradient(135deg, {color}18, {color}08);
                     border: 1px solid {color}35;">
                    <span style="font-size:2rem">{emoji}</span>
                    <div>
                        <div style="font-size:1.05rem; font-weight:600; color:#f0f9ff;">
                            {city} · {selected_date.strftime('%d %B %Y')}
                        </div>
                        <div style="font-size:0.85rem; color:#94a3b8; margin-top:0.2rem">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── 4 Metric Cards ──
                mc1, mc2, mc3, mc4 = st.columns(4)

                with mc1:
                    st.markdown(f"""
                    <div class="metric-card teal">
                        <div class="metric-label">Predicted AQI</div>
                        <div class="metric-value" style="color:{color}">{predicted_aqi:.0f}</div>
                        <div class="metric-sub">index value</div>
                    </div>""", unsafe_allow_html=True)

                with mc2:
                    st.markdown(f"""
                    <div class="metric-card amber">
                        <div class="metric-label">Category</div>
                        <div class="metric-value" style="font-size:1.5rem; color:{color}">{category}</div>
                        <div class="metric-sub">{emoji} air quality</div>
                    </div>""", unsafe_allow_html=True)

                with mc3:
                    st.markdown(f"""
                    <div class="metric-card blue">
                        <div class="metric-label">Confidence Range</div>
                        <div class="metric-value" style="font-size:1.6rem">{lower_bound:.0f}–{upper_bound:.0f}</div>
                        <div class="metric-sub">lower · upper bound</div>
                    </div>""", unsafe_allow_html=True)

                with mc4:
                    st.markdown(f"""
                    <div class="metric-card green">
                        <div class="metric-label">Forecast Horizon</div>
                        <div class="metric-value">{days_ahead}</div>
                        <div class="metric-sub">days beyond last data</div>
                    </div>""", unsafe_allow_html=True)

                # ── Divider ──
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # ── Plotly Chart ──
                st.markdown(f'<div class="section-title">📈 AQI Forecast Trend — {city}</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="legend-row" style="margin-top:-0.3rem; margin-bottom:0.9rem;">
                    <div class="legend-chip"><div class="legend-dot" style="background:#ff7a1a"></div>Predicted AQI</div>
                    <div class="legend-chip"><div class="legend-dot" style="background:{color}"></div>Selected: {selected_date}</div>
                </div>
                """, unsafe_allow_html=True)

                chart_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                chart_df = chart_df[chart_df["ds"] >= pd.Timestamp(today_ist() - timedelta(days=30))]
                chart_df["yhat"]       = chart_df["yhat"].clip(lower=0)
                chart_df["yhat_lower"] = chart_df["yhat_lower"].clip(lower=0)
                chart_df["yhat_upper"] = chart_df["yhat_upper"].clip(lower=0)

                # Selected date marker
                sel_ts = pd.Timestamp(selected_date)

                # Default visible window: center the view around the selected date
                # instead of letting Plotly auto-scale across the full data range
                # (which is what was squeezing the line into a thin sliver).
                window_start = max(chart_df["ds"].min(), sel_ts - timedelta(days=21))
                window_end   = min(chart_df["ds"].max(), sel_ts + timedelta(days=21))
                min_days = 30
                if (window_end - window_start).days < min_days:
                    window_start = max(chart_df["ds"].min(), window_end - timedelta(days=min_days))
                    if (window_end - window_start).days < min_days:
                        window_end = min(chart_df["ds"].max(), window_start + timedelta(days=min_days))

                fig = go.Figure()

                # Confidence band
                fig.add_trace(go.Scatter(
                    x=pd.concat([chart_df["ds"], chart_df["ds"][::-1]]),
                    y=pd.concat([chart_df["yhat_upper"], chart_df["yhat_lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(255, 122, 26, 0.07)",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="Confidence Band"
                ))

                # AQI line
                fig.add_trace(go.Scatter(
                    x=chart_df["ds"],
                    y=chart_df["yhat"],
                    mode="lines",
                    line=dict(color="#ff7a1a", width=2.5),
                    name="Predicted AQI",
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>AQI: %{y:.0f}<extra></extra>"
                ))

                # AQI zone bands (background reference lines)
                zone_colors = ["#22c55e", "#84cc16", "#f59e0b", "#ef4444", "#9333ea"]
                zone_vals   = [50, 100, 200, 300, 500]
                zone_names  = ["Good", "Satisfactory", "Moderate", "Poor", "Severe"]
                for val, col, name in zip(zone_vals, zone_colors, zone_names):
                    fig.add_hline(
                        y=val,
                        line=dict(color=col, width=1, dash="dot"),
                        opacity=0.3,
                        annotation_text=name,
                        annotation_position="right",
                        annotation_font=dict(color=col, size=10)
                    )

                # Selected date vertical line
                fig.add_vline(
                    x=sel_ts,
                    line=dict(color=color, width=2, dash="dash"),
                    opacity=0.8
                )

                # Selected date marker point
                sel_row = chart_df[chart_df["ds"] == sel_ts]
                if not sel_row.empty:
                    fig.add_trace(go.Scatter(
                        x=[sel_ts],
                        y=[predicted_aqi],
                        mode="markers",
                        marker=dict(color=color, size=12, symbol="circle",
                                    line=dict(color="#ffffff", width=2)),
                        name=f"Selected: {selected_date}",
                        hovertemplate=f"<b>{selected_date}</b><br>AQI: {predicted_aqi:.0f} ({category})<extra></extra>"
                    ))

                fig.update_layout(
                    paper_bgcolor="#0a0f1e",
                    plot_bgcolor="#0d1b2e",
                    font=dict(family="Inter", color="#94a3b8", size=12),
                    xaxis=dict(
                        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                        tickfont=dict(color="#64748b"),
                        zeroline=False,
                        title=None,
                        range=[window_start, window_end],   # focused default view
                        rangeslider=dict(
                            visible=True,
                            thickness=0.08,
                            bgcolor="#0d1b2e",
                            bordercolor="rgba(255,255,255,0.08)",
                            borderwidth=1,
                        ),
                        rangeselector=dict(
                            buttons=[
                                dict(count=7,   label="7d",   step="day",   stepmode="backward"),
                                dict(count=30,  label="30d",  step="day",   stepmode="backward"),
                                dict(count=90,  label="90d",  step="day",   stepmode="backward"),
                                dict(count=180, label="6m",   step="day",   stepmode="backward"),
                                dict(step="all", label="All"),
                            ],
                            bgcolor="#131f35",
                            activecolor="#ff7a1a",
                            bordercolor="rgba(255,255,255,0.08)",
                            borderwidth=1,
                            font=dict(color="#94a3b8", size=11),
                            y=1.1,
                        ),
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                        tickfont=dict(color="#64748b"),
                        title=dict(text="AQI Value", font=dict(color="#475569")),
                        fixedrange=False,
                    ),
                    showlegend=False,
                    hovermode="x unified",
                    hoverlabel=dict(
                        bgcolor="#0d1b2e",
                        bordercolor="rgba(255, 122, 26, 0.3)",
                        font=dict(color="#e2e8f0", family="Inter")
                    ),
                    margin=dict(l=10, r=55, t=65, b=10),
                    height=460,
                    dragmode="pan",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "scrollZoom": True,
                        "modeBarButtonsToAdd": ["pan2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                    }
                )
                st.html('<div class="swipe-hint">&#8596; Swipe to see the full chart · pinch or drag the mini-bar below to zoom</div>')

                # ── Seasonal Breakdown ──
                st.markdown('<div class="section-title">🗓️ Monthly AQI Pattern</div>', unsafe_allow_html=True)

                monthly = forecast[["ds", "yhat"]].copy()
                monthly["yhat"] = monthly["yhat"].clip(lower=0)
                monthly["month"] = monthly["ds"].dt.strftime("%b %Y")
                monthly_avg = monthly.groupby(monthly["ds"].dt.to_period("M"))["yhat"].mean().reset_index()
                monthly_avg["ds_str"] = monthly_avg["ds"].astype(str)
                monthly_avg = monthly_avg.tail(13)

                bar_colors = [get_aqi_category(v)[1] for v in monthly_avg["yhat"]]

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=monthly_avg["ds_str"],
                    y=monthly_avg["yhat"].round(0),
                    marker=dict(
                        color=bar_colors,
                        line=dict(width=0),
                        opacity=0.85
                    ),
                    hovertemplate="<b>%{x}</b><br>Avg AQI: %{y:.0f}<extra></extra>",
                    name="Monthly Avg AQI"
                ))

                fig2.update_layout(
                    paper_bgcolor="#0a0f1e",
                    plot_bgcolor="#0d1b2e",
                    font=dict(family="Inter", color="#94a3b8", size=12),
                    xaxis=dict(
                        showgrid=False,
                        tickfont=dict(color="#64748b", size=11),
                        title=None
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                        tickfont=dict(color="#64748b"),
                        title=dict(text="Avg AQI", font=dict(color="#475569"))
                    ),
                    hoverlabel=dict(
                        bgcolor="#0d1b2e",
                        bordercolor="rgba(255, 122, 26, 0.3)",
                        font=dict(color="#e2e8f0", family="Inter")
                    ),
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=300,
                    bargap=0.25
                )

                st.plotly_chart(fig2, use_container_width=True)
                st.html('<div class="swipe-hint">&#8596; Swipe to see the full chart</div>')

            else:
                st.warning("⚠️ Selected date is outside the forecast range. Try a date within the next 365 days.")

        except FileNotFoundError:
            st.error(f"❌ No forecast model found for **{city}**. Please check your `aqi_models/` folder.")
        except Exception as e:
            st.error(f"❌ Forecast error: {str(e)}")

    else:
        # ── Empty State ──
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #334155;">
            <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.7;">🌫️</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; color:#475569; font-weight:500;">
                Select a city and date, then click <span style="color:#ff7a1a">Forecast AQI</span>
            </div>
            <div style="font-size:0.85rem; color:#334155; margin-top:0.5rem;">
                Predictions powered by Meta Prophet time-series model
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================================
# Weather Forecast
# =================================
def render_weather_app():

    # Icon map: condition key -> emoji (kept small and consistent, not oversized)
    ICONS = {
        "storm": "\U0001F329", "rain": "\U0001F327", "drizzle": "\U0001F326",
        "haze": "\U0001F32B", "cloudy": "\U00002601", "partly_cloudy": "\U000026C5",
        "sunny": "\U00002600", "clear": "\U0001F31E",
    }

    st.html("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    * { box-sizing: border-box; }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    section.main > div {
        background: #0b0e17 !important;
        color: #dbe1ec !important;
        font-family: 'Inter', sans-serif;
    }
    /* Prevent any oversized element (long headings, wide charts, etc.)
       from pushing the whole page sideways on phones/tablets — content
       that needs extra width scrolls within its own box instead. */
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
    }
    [data-testid="block-container"] {
        padding: 1.6rem 2.4rem 3rem !important;
        max-width: 1240px !important;
    }

    /* top bar */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
    }
    .topbar-brand {
        display: flex; align-items: center; gap: 10px;
        font-size: 15px; font-weight: 700; color: #e7ebf3;
        letter-spacing: -0.01em;
    }
    .topbar-brand .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #ff7a1a; box-shadow: 0 0 8px #ff7a1aaa;
    }
    .topbar-sub { font-size: 12px; color: #4b5468; }

    /* BIG page header (matches AQI app hero) */
    .page-hero {
        background: linear-gradient(135deg, #0d2137 0%, #0a1628 50%, #071020 100%);
        border: 1px solid rgba(255, 122, 26, 0.15);
        border-radius: 20px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .page-hero::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255, 122, 26, 0.10) 0%, transparent 70%);
        border-radius: 50%; pointer-events: none;
    }
    .page-hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255, 122, 26, 0.12); border: 1px solid rgba(255, 122, 26, 0.3);
        color: #ff7a1a; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 0.3rem 0.9rem; border-radius: 999px; margin-bottom: 1rem;
    }
    .page-hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem; font-weight: 800; color: #f0f9ff;
        margin: 0 0 0.5rem 0; line-height: 1.15;
        display: flex; align-items: center; gap: 12px;
        flex-wrap: wrap;
        row-gap: 4px;
    }
    .page-hero-title .accent { color: #ff7a1a; }
    .page-hero-sub { font-size: 0.95rem; color: #64748b; margin: 0; }

    /* weather icon row above the 6-day chart */
    .icon-day-row {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        margin: 14px 0 4px;
        padding: 14px 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
    }
    .icon-day-col {
        text-align: center;
        padding: 8px 4px;
        border-radius: 12px;
        transition: background 0.2s;
    }
    .icon-day-col.icon-day-active {
        background: rgba(255,255,255,0.06);
    }
    .icon-day-label {
        font-size: 12.5px;
        font-weight: 600;
        color: #7c86a0;
        margin-bottom: 8px;
    }
    .icon-day-active .icon-day-label { color: #f4f6fb; }
    .icon-day-emoji {
        font-size: 26px;
        line-height: 1;
    }

    /* chart section card header (matches reference: white rounded card w/ title+subtitle) */
    .chart-card-head {
        background: linear-gradient(160deg, #131a2c 0%, #0d1220 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 4px;
    }
    .chart-card-title {
        font-size: 21px; font-weight: 800; color: #f4f6fb;
        margin-bottom: 5px; letter-spacing: -0.01em;
    }
    .chart-card-sub { font-size: 13px; color: #7c86a0; }
    .chart-card-note {
        font-size: 11px; color: #4b5468; font-style: italic;
        margin-top: 8px; padding-top: 8px;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* hero current-weather glass card */
    .hero-glass {
        position: relative;
        border-radius: 24px;
        padding: 30px 32px;
        margin-bottom: 18px;
        overflow: hidden;
        background:
            radial-gradient(120% 140% at 8% 0%, rgba(255, 122, 26, 0.16) 0%, transparent 55%),
            radial-gradient(90% 120% at 95% 100%, rgba(194, 65, 12, 0.14) 0%, transparent 55%),
            linear-gradient(150deg, #131a2c 0%, #0d1220 60%, #0a0d16 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 20px 50px -20px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 22px;
    }
    .hero-place { font-size: 25px; font-weight: 600; color: #cfd6e4; display:flex; align-items:center; gap:3px;}
    .hero-date  { font-size: 14px; color: #5b6478; margin-top: 3px; }
    .hero-cond-pill {
        background: rgba(255, 122, 26, 0.12);
        border: 1px solid rgba(255, 122, 26, 0.3);
        color: #ff7a1a;
        font-size: 12px; font-weight: 600;
        padding: 5px 13px; border-radius: 999px;
        letter-spacing: 0.02em;
    }
    .hero-main-row {
        display: flex; align-items: center; gap: 26px; margin-bottom: 6px;
    }
    .hero-icon { font-size: 58px; line-height: 1; filter: drop-shadow(0 4px 14px rgba(255, 122, 26, 0.25)); }
    .hero-temp {
        font-family: 'Inter', sans-serif;
        font-size: 76px; font-weight: 800; color: #f4f6fb;
        line-height: 1; letter-spacing: -0.04em;
    }
    .hero-temp sup { font-size: 32px; font-weight: 600; color: #8a93a8; top: -0.5em; }
    .hero-meta-col { display: flex; flex-direction: column; gap: 4px; padding-bottom: 6px;}
    .hero-feels { font-size: 13.5px; color: #8a93a8; }
    .hero-condition-text { font-size: 20px; font-weight: 700; color: #eef1f7; }
    .hero-desc { font-size: 13px; color: #6b7488; margin-top: 12px; max-width: 560px; line-height: 1.55; }

    /* metric strip - standard sized, evenly fit */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
        margin-top: 24px;
    }
    .metric-cell {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 12px 8px;
        text-align: center;
    }
    .metric-cell .mi { font-size: 15px; margin-bottom: 5px; opacity: 0.85;}
    .metric-cell .ml { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #5b6478; margin-bottom: 3px; }
    .metric-cell .mv { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; color: #dbe1ec; }

    /* section header */
    .sec-head {
        display: flex; align-items: center; gap: 8px;
        font-size: 13px; font-weight: 700; color: #8a93a8;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 26px 0 12px;
    }
    .sec-head i { color: #ff7a1a; font-size: 12px; }

    /* forecast day-row list (standard, compact like real weather apps) */
    .forecast-list {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 6px 18px;
    }
    .f-row {
        display: grid;
        grid-template-columns: 92px 34px 1fr 100px 46px;
        align-items: center;
        gap: 14px;
        padding: 13px 0;
        border-bottom: 1px solid rgba(255,255,255,0.045);
    }
    .f-row:last-child { border-bottom: none; }
    .f-row.is-today .f-day { color: #ff7a1a; }
    .f-day { font-size: 13.5px; font-weight: 600; color: #cfd6e4; }
    .f-date { font-size: 11px; color: #4b5468; display: block; margin-top: 1px; }
    .f-icon { font-size: 19px; text-align: center; }
    .f-cond { font-size: 12.5px; color: #8a93a8; }
    .f-rain { font-size: 12px; color: #fb923c; text-align: right; font-weight: 600; font-family: 'JetBrains Mono', monospace;}
    .f-temp-bar-wrap { display: flex; align-items: center; gap: 8px; }
    .f-temp-track { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,0.07); position: relative; overflow: hidden;}
    .f-temp-fill { position:absolute; top:0; bottom:0; border-radius: 2px; }
    .f-temp-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #eef1f7; min-width: 34px; text-align: right;}

    /* Shown only on small/touch screens, right under charts that scroll */
    .swipe-hint {
        display: none;
        font-size: 11px;
        color: #5b6478;
        text-align: center;
        margin: -8px 0 16px;
    }

    /* detailed daily breakdown table with headers */
    .bd-wrap {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        overflow: hidden;
    }
    .bd-head-row {
        display: grid;
        grid-template-columns: 118px 1.4fr 1fr 1fr 1fr 1fr 1fr;
        gap: 10px;
        align-items: center;
        padding: 12px 18px;
        background: rgba(255,255,255,0.03);
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .bd-head-cell {
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.07em; color: #5b6478; text-align: center;
    }
    .bd-head-cell.left { text-align: left; }
    .bd-row {
        display: grid;
        grid-template-columns: 118px 1.4fr 1fr 1fr 1fr 1fr 1fr;
        gap: 10px;
        align-items: center;
        padding: 14px 18px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        transition: background 0.15s;
    }
    .bd-row:last-child { border-bottom: none; }
    .bd-row:hover { background: rgba(255,255,255,0.025); }
    .bd-row.bd-today { background: rgba(255, 122, 26, 0.05); }
    .bd-day-cell { text-align: center; }
    .bd-day-cell .bd-day-name { font-size: 13.5px; font-weight: 700; color: #cfd6e4; display:block; }
    .bd-row.bd-today .bd-day-cell .bd-day-name { color: #ff7a1a; }
    .bd-day-cell .bd-day-date { font-size: 12px; color: #4b5468; }
    .bd-cond-cell { font-size: 14px; font-weight: 600; color: #e8ecf5; text-align: center; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .bd-cond-cell .bd-icon { font-size: 18px; filter: drop-shadow(0 0 6px rgba(255,209,102,0.45)); }
    .bd-metric-cell { text-align: center; }
    .bd-metric-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #dbe1ec; display: block; }
    .bd-metric-sub { font-size: 11.3px; color: #4b5468; }
    .bd-temp-cell { display: flex; align-items: center; justify-content: center; gap: 6px; }
    .bd-temp-high { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; color: #f4f6fb; }
    .bd-temp-low { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: #7c8598; }

    /* detail panels */
    .panel-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 6px; }
    .panel {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 16px 18px;
    }
    .panel-title {
        font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
        color: #5b6478; margin-bottom: 12px; display:flex; align-items:center; gap:6px;
    }
    .panel-title i { color: #ff7a1a; }
    .p-row { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04);}
    .p-row:last-child { border-bottom: none; }
    .p-label { font-size: 12.5px; color: #6b7488; }
    .p-value { font-family: 'JetBrains Mono', monospace; font-size: 13.5px; font-weight: 600; color: #dbe1ec; }

    /* sun card */
    .sun-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 16px 18px;
    }
    .sun-times-row { display: flex; justify-content: space-between; margin-top: 6px; }
    .sun-block { text-align: center; }
    .sun-lab { font-size: 13px; color: #5b6478; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px;}
    .sun-val { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; color: #f0b95c; }

    /* footer / app description block (matches AQI app closing style) */
    .app-footer {
        text-align: center;
        padding: 3rem 2rem 2rem;
        margin-top: 1.5rem;
    }
    .app-footer-icon {
        font-size: 2.6rem;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    .app-footer-text {
        font-size: 1rem;
        color: #7c86a0;
        font-weight: 400;
    }
    .app-footer-text .accent { color: #ff7a1a; font-weight: 700; }
    .app-footer-sub {
        font-size: 0.8rem;
        color: #4b5468;
        margin-top: 0.5rem;
    }

    /* select widget */
    .stSelectbox div[data-baseweb="select"] > div{
        background:#1b2942 !important;
        border:3px solid rgba(255,137,4,.20) !important;
        border-radius:16px !important;
        min-height:54px !important;
        padding:0 14px !important;
        display:flex !important;
        align-items:center !important;
        transition:all .25s ease;
    }
    .stSelectbox div[data-baseweb="select"] > div:hover{
        border:3px solid #ff8904 !important;
        box-shadow:
            0 0 0 2px rgba(255,137,4,.18),
            0 0 16px rgba(255,137,4,.30);
    }
    .stSelectbox div[data-baseweb="select"] > div:focus-within{
        border:3px solid #ff8904 !important;
        box-shadow:
            0 0 0 2px rgba(255,137,4,.18),
            0 0 18px rgba(255,137,4,.35);
    }
    .stSelectbox div[data-baseweb="select"] input{
        color:#ffffff !important;
        font-size:15px !important;
        font-weight:600 !important;
        caret-color:#ff8904 !important;
    }
    .stSelectbox svg{
        color:#ff8904 !important;
        transition:.25s;
    }
    .stSelectbox svg:hover{
        color:#ffb84d !important;
    }
    .stSelectbox div[data-baseweb="select"] span{
        font-weight:700 !important;
        font-size:20px !important;
        color:#ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] div{
        font-weight:700 !important;
    }
 
    #MainMenu, footer, header,
    [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

    /* ═══════════════════════════════════════════════════════
       RESPONSIVE — Laptops, Tablets, Smartphones
       (Additive only — does not modify any rule above)
       ═══════════════════════════════════════════════════════ */

    /* Safety net: force Streamlit's native columns (city selector /
       jump-to-day, the 3 detail panels, etc.) to stack cleanly on
       narrow screens, even on older Streamlit versions that don't
       do this on their own. */
    @media (max-width: 768px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            row-gap: 0.8rem !important;
        }
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }
    }

    /* Small laptops / large tablets */
    @media (max-width: 1200px) {
        [data-testid="block-container"] {
            padding: 1.4rem 1.6rem 2.4rem !important;
        }
        .metric-strip  { grid-template-columns: repeat(3, 1fr); }
        .icon-day-row  { grid-template-columns: repeat(3, 1fr); row-gap: 10px; }
    }

    /* Tablets (portrait) */
    @media (max-width: 900px) {
        .page-hero       { padding: 1.6rem 1.6rem; }
        .page-hero-title { font-size: 1.9rem; }
        .hero-temp       { font-size: 56px; }
        .hero-icon       { font-size: 44px; }
        .hero-main-row   { gap: 16px; }

        /* Wide data tables scroll horizontally instead of squeezing
           illegibly — content and columns stay unchanged. */
        .bd-wrap                  { overflow-x: auto; }
        .bd-head-row, .bd-row     { min-width: 640px; }

        /* Line charts (24-Hour / 6-Day Forecast): same idea — scroll
           instead of squeezing point labels into each other. */
        [data-testid="stPlotlyChart"] { overflow-x: auto; }
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {
            min-width: 640px;
        }
        .swipe-hint { display: block; }
    }

    /* Smartphones */
    @media (max-width: 640px) {
        [data-testid="block-container"] {
            padding: 1rem 0.9rem 2rem !important;
        }
        .page-hero        { padding: 1.3rem 1.1rem; border-radius: 14px; }
        .page-hero-title  { font-size: 1.5rem; gap: 8px; }
        .page-hero-sub    { font-size: 0.8rem; }
        .page-hero-badge  { font-size: 0.62rem; padding: 0.25rem 0.7rem; }

        .hero-glass          { padding: 20px 18px; }
        .hero-top-row        { flex-wrap: wrap; gap: 10px; }
        .hero-place          { font-size: 19px; }
        .hero-main-row       { flex-wrap: wrap; gap: 14px; }
        .hero-temp           { font-size: 46px; }
        .hero-icon           { font-size: 36px; }
        .hero-condition-text { font-size: 16px; }

        .metric-strip  { grid-template-columns: repeat(2, 1fr); gap: 8px; }
        .icon-day-row  { grid-template-columns: repeat(3, 1fr); gap: 6px; padding: 10px 6px; }
        .icon-day-emoji { font-size: 20px; }
        .icon-day-label { font-size: 11px; }

        .chart-card-title { font-size: 17px; }
        .sec-head         { font-size: 12px; margin: 20px 0 10px; }

        .bd-head-row, .bd-row { min-width: 560px; }

        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container {
            min-width: 720px;
        }
    }

    /* Small phones */
    @media (max-width: 400px) {
        .hero-temp     { font-size: 38px; }
        .metric-strip  { grid-template-columns: repeat(2, 1fr); }
        .icon-day-row  { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """)


    # --------------------------------------------------------------
    # Model loading
    # --------------------------------------------------------------
    MODEL_DIR = "weather_models"
    VARIABLES = ["Temperature", "Humidity", "Pressure", "WindSpeed"]

    @st.cache_resource(show_spinner=False)
    def load_models(city):
        models = {}
        for var in VARIABLES:
            path = os.path.join(MODEL_DIR, var, f"{city}.pkl")
            models[var] = joblib.load(path) if os.path.exists(path) else None
        return models

    def predict_variable(model, real_today, days=7):
        """Return `days` consecutive daily predictions starting from the REAL
        current IST date (real_today), not from wherever the model's training
        data happens to end. Prophet's make_future_dataframe() only knows how
        to count forward from its last trained date, so if the model is a few
        days stale we have to ask it for enough extra periods to actually reach
        today, then pick today (and the following days) out by date - never by
        row position."""
        if model is None:
            return pd.DataFrame({"ds": [], "yhat": []})

        last_hist_date = model.history["ds"].max().date()
        stale_days = (real_today - last_hist_date).days  # >0 if model is behind

        # Make sure the forecast horizon reaches at least real_today + (days-1).
        periods = max(1, stale_days + days - 1)

        future = model.make_future_dataframe(periods=periods, freq="D")
        forecast = model.predict(future)
        forecast["cal_date"] = forecast["ds"].dt.date

        end_date = real_today + timedelta(days=days - 1)
        mask = (forecast["cal_date"] >= real_today) & (forecast["cal_date"] <= end_date)
        return forecast.loc[mask, ["ds", "yhat"]].reset_index(drop=True)

    def predict_hourly_today(model, target_date, hours=24):
        """Interpolate an hourly curve for `target_date` (a real calendar date,
        picked out of daily[] rather than a row offset) by blending that day's
        and the next day's Prophet prediction."""
        if model is None:
            return None

        last_hist_date = model.history["ds"].max().date()
        stale_days = (target_date - last_hist_date).days
        periods = max(1, stale_days + 1)  # ensure we reach target_date + 1 day

        future = model.make_future_dataframe(periods=periods, freq="D")
        forecast = model.predict(future)
        forecast["cal_date"] = forecast["ds"].dt.date

        row_day = forecast.loc[forecast["cal_date"] == target_date]
        row_next = forecast.loc[forecast["cal_date"] == target_date + timedelta(days=1)]
        if row_day.empty or row_next.empty:
            return None

        v_day = float(row_day["yhat"].iloc[0])
        v_next = float(row_next["yhat"].iloc[0])
        hours_arr = np.arange(hours)
        daily_curve = np.sin((hours_arr - 6) / 24 * 2 * np.pi - np.pi/2) * 0.5 + 0.5
        blended = v_day + (v_next - v_day) * (hours_arr / hours) * 0.3
        variation = (daily_curve - 0.5) * (v_day * 0.12)
        return blended + variation

    cities = sorted([
        f.replace(".pkl", "")
        for f in os.listdir(os.path.join(MODEL_DIR, "Temperature"))
        if f.endswith(".pkl")
    ])

    # --------------------------------------------------------------
    # BIG PAGE HERO
    # --------------------------------------------------------------
    st.html(f"""
    <div class="page-hero">
        <div class="page-hero-badge">&#127749; India Weather Intelligence</div>
        <h1 class="page-hero-title">🌦️ Weather <span class="accent">Forecasting</span> System</h1>
        <p class="page-hero-sub">
            Prophet-powered time-series forecasting &middot; {len(cities)} Indian cities &middot; Temperature, Humidity, Pressure &amp; Wind
        </p>
    </div>
    """)

    col_city, col_jump = st.columns([2, 1])
    with col_city:
        city = st.selectbox("Select City", cities, label_visibility="visible")

    # --------------------------------------------------------------
    # Predict
    # --------------------------------------------------------------
    with st.spinner("Loading forecast..."):
        models = load_models(city)
        days_ahead = 6
        real_today = now_ist().date()

        temp_fc = predict_variable(models["Temperature"], real_today, days_ahead)
        hum_fc  = predict_variable(models["Humidity"], real_today, days_ahead)
        pres_fc = predict_variable(models["Pressure"], real_today, days_ahead)
        wind_fc = predict_variable(models["WindSpeed"], real_today, days_ahead)

    daily = []
    for i in range(days_ahead):
        t = max(0.0, float(temp_fc["yhat"].iloc[i]))
        h = min(100.0, max(0.0, float(hum_fc["yhat"].iloc[i])))
        p = float(pres_fc["yhat"].iloc[i])
        w = max(0.0, float(wind_fc["yhat"].iloc[i]))
        ds = temp_fc["ds"].iloc[i].date()

        rc = rain_chance(h, p)
        cond, icon_key = weather_condition(t, h, p, rc)
        fl = feels_like(t, h)
        rp = rain_perception(rc)
        cl_name, cl_pct = cloud_coverage(h)
        gust = wind_gust(w)
        uv = uv_index(t, cl_pct)
        desc = weather_description(cond, t, fl, h, rc, w)
        sr, ss = sunrise_sunset(city)

        daily.append({
            "date": ds, "temp": t, "humidity": h, "pressure": p, "wind": w,
            "rain_chance": rc, "condition": cond, "icon_key": icon_key,
            "feels_like": fl, "rain_perception": rp, "cloud_name": cl_name,
            "cloud_pct": cl_pct, "gust": gust, "uv": uv, "desc": desc,
            "sunrise": sr, "sunset": ss,
        })

    # Optional, additive-only touch: a quiet data-freshness note using the
    # app's existing subtle-text style (same look as the topbar subtitle).
    # Purely informational - doesn't alter layout, navigation, or any
    # existing feature; safe to delete if not wanted.
    if models["Temperature"] is not None:
        model_last_date = models["Temperature"].history["ds"].max().date()
        freshness_note = (
            "Model data current as of today"
            if model_last_date >= real_today
            else f"Model trained through {model_last_date.strftime('%d %b')} &middot; forecast projected forward to today"
        )
        st.html(f'<div class="topbar-sub" style="margin: -8px 0 14px 2px;">{freshness_note}</div>')

    # Jump-to-day selector uses the REAL dates from daily[] (not independently
    # computed date.today() offsets), so labels always match the actual forecast
    # regardless of how far the model's training data extends.
    jump_labels = ["Today"] + [d["date"].strftime("%A, %d %b") for d in daily[1:]]
    with col_jump:
        jump_choice = st.selectbox("Jump to day", jump_labels, label_visibility="visible")
    selected_day_idx = jump_labels.index(jump_choice)

    hourly_temp = predict_hourly_today(models["Temperature"], daily[selected_day_idx]["date"], 24)
    today_data = daily[selected_day_idx]
    today_str = today_data["date"].strftime("%A, %d %B %Y")
    is_showing_today = (selected_day_idx == 0)

    # --------------------------------------------------------------
    # HERO CARD
    # --------------------------------------------------------------
    icon = ICONS.get(today_data["icon_key"], "\U00002600")

    st.html(f"""
    <div class="hero-glass">
        <div class="hero-top-row">
            <div>
                <div class="hero-place"><i class="fas fa-location-dot" style="font-size:18px;color:#ff7a1a"></i> {city}</div>
                <div class="hero-date">{today_str}{'' if is_showing_today else ' &middot; forecast'}</div>
            </div>
            <div class="hero-cond-pill">{today_data['condition']}</div>
        </div>
        <div class="hero-main-row">
            <div class="hero-icon">{icon}</div>
            <div class="hero-temp">{today_data['temp']:.0f}<sup>&deg;</sup></div>
            <div class="hero-meta-col">
                <div class="hero-condition-text">{today_data['condition']}</div>
                <div class="hero-feels">Feels like {today_data['feels_like']:.0f}&deg;C</div>
            </div>
        </div>
        <div class="hero-desc">{today_data['desc']}</div>

        <div class="metric-strip">
            <div class="metric-cell"><div class="mi">&#128167;</div><div class="ml">Humidity</div><div class="mv">{today_data['humidity']:.0f}%</div></div>
            <div class="metric-cell"><div class="mi">&#128168;</div><div class="ml">Wind</div><div class="mv">{today_data['wind']:.1f} km/h</div></div>
            <div class="metric-cell"><div class="mi">&#127786;</div><div class="ml">Gust</div><div class="mv">{today_data['gust']:.1f} km/h</div></div>
            <div class="metric-cell"><div class="mi">&#127777;&#65039;</div><div class="ml">Pressure</div><div class="mv">{today_data['pressure']:.0f} hPa</div></div>
            <div class="metric-cell"><div class="mi">&#127782;</div><div class="ml">Rain</div><div class="mv">{today_data['rain_chance']}%</div></div>
            <div class="metric-cell"><div class="mi">&#9728;&#65039;</div><div class="ml">UV Index</div><div class="mv">{today_data['uv']:.0f}</div></div>
        </div>
    </div>
    """)

    # --------------------------------------------------------------
    # 24-HOUR FORECAST CHART (dual-axis: temp line + rain bars)
    # --------------------------------------------------------------
    st.html("""
    <div class="chart-card-head">
        <div class="chart-card-title">24-Hour Forecast</div>
        <div class="chart-card-sub">Temperature curve with rain probability for the selected date.</div>
        <div class="chart-card-note">Hourly curve interpolated from daily forecast for visualization &mdash; not an independently modeled hourly prediction.</div>
    </div>
    """)

    hours_labels = [f"{h:02d}:00" for h in range(24)]

    # Build hourly rain-chance curve from today's humidity/pressure, with a
    # realistic diurnal shape (rain risk usually peaks afternoon/evening)
    base_rc = today_data["rain_chance"]
    hour_arr = np.arange(24)
    rain_curve = base_rc * (0.55 + 0.45 * np.sin((hour_arr - 5) / 24 * 2 * np.pi - np.pi/2) ** 2)
    rain_curve = np.clip(rain_curve, 0, 95).round(0)

    now_hour = min(now_ist().hour, 23) if is_showing_today else None

    fig_hourly = go.Figure()

    # Rain-chance bars on secondary axis
    fig_hourly.add_trace(go.Bar(
        x=hours_labels, y=rain_curve,
        name="Rain Chance",
        marker=dict(color="rgba(234, 88, 12, 0.55)"),
        yaxis="y2",
        hovertemplate="%{x}<br>Rain %{y:.0f}%<extra></extra>",
    ))

    # Temperature line with point labels
    fig_hourly.add_trace(go.Scatter(
        x=hours_labels, y=hourly_temp,
        mode="lines+markers+text",
        line=dict(color="#f0b95c", width=3, shape="spline", smoothing=0.5),
        marker=dict(size=6, color="#f0b95c", line=dict(color="#0b0e17", width=1.5)),
        text=[f"{t:.0f}\u00b0" if i % 3 == 0 else "" for i, t in enumerate(hourly_temp)],
        textposition="top center",
        textfont=dict(size=11, color="#dbe1ec", family="Inter"),
        name="Temperature",
        hovertemplate="%{x}<br>Temp %{y:.1f}&deg;C<extra></extra>",
    ))

    # "Now" vertical marker
    if now_hour is not None:
        fig_hourly.add_vline(
            x=now_hour, line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"),
        )

    tick_positions = [0, 3, 6, 9, 12, 15, 18, 21]
    tick_text = [hours_labels[i] for i in tick_positions]

    fig_hourly.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8a93a8", size=12),
        xaxis=dict(
            tickmode="array", tickvals=tick_positions, ticktext=tick_text,
            showgrid=False, zeroline=False, showline=False,
            tickfont=dict(size=11.5, color="#8a93a8"),
        ),
        yaxis=dict(
            title=dict(text="\u00b0C", font=dict(size=11, color="#5b6478")),
            showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            zeroline=False, showline=False,
            tickfont=dict(size=11, color="#5b6478"),
        ),
        yaxis2=dict(
            title=dict(text="Rain %", font=dict(size=11, color="#5b6478")),
            overlaying="y", side="right",
            range=[0, 100], showgrid=False,
            tickfont=dict(size=11, color="#5b6478"),
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15,
                    font=dict(size=12, color="#c3c9d6"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=440,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#131a2c", bordercolor="rgba(255,255,255,0.1)",
                         font=dict(color="#dbe1ec", family="Inter", size=12)),
        bargap=0.35,
    )
    st.plotly_chart(fig_hourly, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    st.html('<div class="swipe-hint">&#8596; Swipe to see the full 24-hour chart</div>')

    # --------------------------------------------------------------
    # 5/6-DAY FORECAST CHART (High/Low dual line, matches reference)
    # --------------------------------------------------------------
    st.html("""
    <div class="chart-card-head">
        <div class="chart-card-title">6-Day Forecast</div>
        <div class="chart-card-sub">High and low temperature trend from today.</div>
    </div>
    """)

    day_full = [d["date"].strftime("%d %b") for d in daily]

    # --------------------------------------------------------------
    # WEATHER ICON ROW (above the chart, like reference image)
    # --------------------------------------------------------------
    icon_row_html = '<div class="icon-day-row">'
    for i, d in enumerate(daily):
        day_label = "Today" if i == 0 else d["date"].strftime("%a")
        icon = ICONS.get(d["icon_key"], "\u2600")
        active_cls = "icon-day-active" if i == 0 else ""
        icon_row_html += f"""
        <div class="icon-day-col {active_cls}">
            <div class="icon-day-label">{day_label}</div>
            <div class="icon-day-emoji">{icon}</div>
        </div>"""
    icon_row_html += "</div>"
    st.html(icon_row_html)


    # Prophet gives one daily value; derive a realistic daily LOW using the
    # typical diurnal swing (low tends to sit ~7-9C below the daily mean/high
    # for these Indian city climates, tightened by humidity)
    temps_all = [d["temp"] for d in daily]
    lows_all  = [max(0, t - (9 - (d["humidity"] / 100) * 3)) for t, d in zip(temps_all, daily)]

    today_idx = 0

    fig_week = go.Figure()

    fig_week.add_trace(go.Scatter(
        x=day_full, y=temps_all,
        mode="lines+markers+text",
        line=dict(color="#f4f6fb", width=2.5, shape="spline", smoothing=0.4),
        marker=dict(size=7, color="#f4f6fb", line=dict(color="#0b0e17", width=1.5)),
        text=[f"{t:.0f}\u00b0" for t in temps_all],
        textposition="top center",
        textfont=dict(size=12, color="#f4f6fb", family="Inter"),
        name="High",
        hovertemplate="%{x}<br>High %{y:.1f}&deg;C<extra></extra>",
    ))

    fig_week.add_trace(go.Scatter(
        x=day_full, y=lows_all,
        mode="lines+markers+text",
        line=dict(color="#7c8598", width=2.5, shape="spline", smoothing=0.4),
        marker=dict(size=7, color="#7c8598", line=dict(color="#0b0e17", width=1.5)),
        text=[f"{t:.0f}\u00b0" for t in lows_all],
        textposition="bottom center",
        textfont=dict(size=12, color="#9aa2b3", family="Inter"),
        name="Low",
        hovertemplate="%{x}<br>Low %{y:.1f}&deg;C<extra></extra>",
    ))

    # Vertical marker on today (selected date)
    fig_week.add_vline(
        x=today_idx, line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
    )

    fig_week.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8a93a8", size=12),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=12, color="#8a93a8")),
        yaxis=dict(
            title=dict(text="Temperature \u00b0C", font=dict(size=11, color="#5b6478")),
            showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showline=False,
            tickfont=dict(size=11, color="#5b6478"),
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15,
                    font=dict(size=12, color="#c3c9d6"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#131a2c", bordercolor="rgba(255,255,255,0.1)",
                         font=dict(color="#dbe1ec", family="Inter", size=12)),
    )
    st.plotly_chart(fig_week, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})
    st.html('<div class="swipe-hint">&#8596; Swipe to see the full 6-day chart</div>')

    # --------------------------------------------------------------
    # DAILY BREAKDOWN - detailed table with column headers
    # --------------------------------------------------------------
    st.html('<div class="sec-head"><i class="fas fa-table-list"></i> Daily Breakdown</div>')

    temp_min = min(temps_all)
    temp_max = max(temps_all)
    t_range = (temp_max - temp_min) or 1

    bd_html = '<div class="bd-wrap">'
    bd_html += """
    <div class="bd-head-row">
        <div class="bd-head-cell">Day</div>
        <div class="bd-head-cell">Condition</div>
        <div class="bd-head-cell">Temp</div>
        <div class="bd-head-cell">Humidity</div>
        <div class="bd-head-cell">Wind</div>
        <div class="bd-head-cell">Pressure</div>
        <div class="bd-head-cell">Rain</div>
    </div>
    """

    for i, d in enumerate(daily):
        is_today = i == 0
        day_name = "Today" if is_today else d["date"].strftime("%A")
        date_label = d["date"].strftime("%d %b")
        icon = ICONS.get(d["icon_key"], "\u2600")
        low_val = lows_all[i]
        row_cls = "bd-row bd-today" if is_today else "bd-row"

        bd_html += f"""
        <div class="{row_cls}">
            <div class="bd-day-cell">
                <span class="bd-day-name">{day_name}</span>
                <span class="bd-day-date">{date_label}</span>
            </div>
            <div class="bd-cond-cell"><span class="bd-icon">{icon}</span>{d['condition']}</div>
            <div class="bd-temp-cell">
                <span class="bd-temp-high">{d['temp']:.0f}&deg;</span>
                <span class="bd-temp-low">{low_val:.0f}&deg;</span>
            </div>
            <div class="bd-metric-cell">
                <span class="bd-metric-val">{d['humidity']:.0f}%</span>
            </div>
            <div class="bd-metric-cell">
                <span class="bd-metric-val">{d['wind']:.1f}</span>
                <span class="bd-metric-sub">km/h</span>
            </div>
            <div class="bd-metric-cell">
                <span class="bd-metric-val">{d['pressure']:.0f}</span>
                <span class="bd-metric-sub">hPa</span>
            </div>
            <div class="bd-metric-cell">
                <span class="bd-metric-val" style="color:{'#4ade80' if d['rain_chance']<30 else '#f0b95c' if d['rain_chance']<60 else '#f87171'}">{d['rain_chance']}%</span>
            </div>
        </div>"""

    bd_html += "</div>"
    st.html(bd_html)

    # --------------------------------------------------------------
    # DETAIL PANELS
    # --------------------------------------------------------------
    detail_label = "Today in Detail" if is_showing_today else f"{today_data['date'].strftime('%A')} in Detail"
    st.html(f'<div class="sec-head"><i class="fas fa-chart-simple"></i> {detail_label}</div>')

    d = today_data
    uv_label = "Low" if d["uv"] <= 2 else "Moderate" if d["uv"] <= 5 else "High" if d["uv"] <= 7 else "Very High"
    rc_color = "#4ade80" if d["rain_chance"] < 30 else "#f0b95c" if d["rain_chance"] < 60 else "#f87171"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html(f"""
        <div class="panel">
            <div class="panel-title"><i class="fas fa-cloud-rain"></i> Precipitation</div>
            <div class="p-row"><span class="p-label">Rain chance</span><span class="p-value" style="color:{rc_color}">{d['rain_chance']}%</span></div>
            <div class="p-row"><span class="p-label">Description</span><span class="p-value">{d['rain_perception']}</span></div>
            <div class="p-row"><span class="p-label">Cloud cover</span><span class="p-value">{d['cloud_name']}</span></div>
            <div class="p-row"><span class="p-label">Cloud %</span><span class="p-value">{d['cloud_pct']}%</span></div>
        </div>
        """)

    with col2:
        st.html(f"""
        <div class="panel">
            <div class="panel-title"><i class="fas fa-wind"></i> Wind &amp; Pressure</div>
            <div class="p-row"><span class="p-label">Wind speed</span><span class="p-value">{d['wind']:.1f} km/h</span></div>
            <div class="p-row"><span class="p-label">Wind gust</span><span class="p-value">{d['gust']:.1f} km/h</span></div>
            <div class="p-row"><span class="p-label">Pressure</span><span class="p-value">{d['pressure']:.0f} hPa</span></div>
            <div class="p-row"><span class="p-label">Humidity</span><span class="p-value">{d['humidity']:.0f}%</span></div>
        </div>
        """)

    with col3:
        st.html(f"""
        <div class="panel">
            <div class="panel-title"><i class="fas fa-temperature-half"></i> Temperature &amp; UV</div>
            <div class="p-row"><span class="p-label">Temperature</span><span class="p-value">{d['temp']:.1f}&deg;C</span></div>
            <div class="p-row"><span class="p-label">Feels like</span><span class="p-value">{d['feels_like']:.1f}&deg;C</span></div>
            <div class="p-row"><span class="p-label">UV index</span><span class="p-value">{d['uv']:.0f} &middot; {uv_label}</span></div>
            <div class="p-row"><span class="p-label">Condition</span><span class="p-value">{d['condition']}</span></div>
        </div>
        """)

    # --------------------------------------------------------------
    # SUN CARD - animated, real-time sun position
    # --------------------------------------------------------------
    st.html('<div class="sun-card">')
    st.html('<div class="sec-head"><i class="fas fa-sun"></i> Sunrise &amp; Sunset</div>')

    sr, ss = today_data["sunrise"], today_data["sunset"]

    # Compute real-time sun progress (0 = sunrise, 1 = sunset) only meaningful
    # for "today" - for future days we just show the static arc at rest.
    def to_minutes(t_str):
        h, m = map(int, t_str.split(":"))
        return h * 60 + m

    sr_min, ss_min = to_minutes(sr), to_minutes(ss)
    _now = now_ist()
    now_min = _now.hour * 60 + _now.minute
    day_length = max(1, ss_min - sr_min)
    peak_min = sr_min + day_length // 2
    peak_label = f"{peak_min // 60:02d}:{peak_min % 60:02d}"

    if is_showing_today:
        progress = (now_min - sr_min) / day_length
        progress = max(0.0, min(1.0, progress))
        is_daytime = sr_min <= now_min <= ss_min
    else:
        progress = 0.5
        is_daytime = True

    # Position along the arc: parametrize x 20->380, y follows a parabola peaking at y=8
    arc_x = 20 + progress * 360
    arc_y = 72 - (4 * 0.62 * progress * (1 - progress)) * 160  # matches path curvature
    sun_color = "#f0b95c" if is_daytime else "#5b6478"
    sun_glow  = "0.9" if is_daytime else "0.25"
    now_label = now_ist().strftime("%H:%M")


    # Smooth arc from sunrise (t=0) to sunset (t=1), peaking in the middle -
    # same shape as the old SVG path, just plotted with Plotly instead.
    t = np.linspace(0, 1, 60)
    arc_y_curve = np.sin(np.pi * t)  # 0 -> 1 -> 0
    
    fig_sun = go.Figure()
    
    # dashed baseline
    fig_sun.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode="lines",
        line=dict(color="rgba(255,255,255,0.08)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    
    # the arc itself
    fig_sun.add_trace(go.Scatter(
        x=t, y=arc_y_curve, mode="lines",
        line=dict(color="#f0b95c", width=2, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    
    # sunrise / sunset endpoint dots
    fig_sun.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode="markers",
        marker=dict(color="#f0b95c", size=8, opacity=0.7),
        hoverinfo="skip", showlegend=False,
    ))
    
    # current sun position (or resting mid-arc position for non-today views)
    fig_sun.add_trace(go.Scatter(
        x=[progress], y=[np.sin(np.pi * progress)], mode="markers",
        marker=dict(
            color=sun_color, size=26,
            line=dict(color=sun_color, width=1),
            opacity=float(sun_glow),
        ),
        hoverinfo="skip", showlegend=False,
    ))
    
    fig_sun.update_layout(
        height=90,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.05, 1.05]),
        yaxis=dict(visible=False, range=[-0.15, 1.25]),
        showlegend=False,
    )
    

    st.plotly_chart(fig_sun, use_container_width=True, config={"staticPlot": True, "displayModeBar": False})
    st.html(f"""
        <div class="sun-times-row">
            <div class="sun-block"><div class="sun-lab">Sunrise</div><div class="sun-val">{sr}</div></div>
            <div class="sun-block">
                <div class="sun-lab">{'Now' if is_showing_today else 'Peak'}</div>
                <div class="sun-val" style="color:{'#ff6900' if is_showing_today else "#dd1756"}; font-size: 20px;">
                    {now_label if is_showing_today else peak_label}
                </div>
            </div>
            <div class="sun-block"><div class="sun-lab">Sunset</div><div class="sun-val">{ss}</div></div>
        </div>
    </div>
    """)
    st.html('<div class="sun-card">')


    # --------------------------------------------------------------
    # APP FOOTER 
    # --------------------------------------------------------------
    st.html("""
    <div class="app-footer">
        <div class="app-footer-icon">&#9925;&#65039;</div>
        <div class="app-footer-text">
            Weather intelligence for <span class="accent">India</span>, powered by Prophet time-series forecasting
        </div>
        <div class="app-footer-sub">Predictions powered by Meta Prophet time-series model</div>
    </div>
    """)


# ─────────────────────────────────────────
# ROUTER — decides what to draw based on navigation state
# ─────────────────────────────────────────
_active = st.session_state.active_system

if _active is None:
    render_landing_page()
elif _active == "aqi":
    render_switcher("aqi")
    render_aqi_app()
elif _active == "weather":
    render_switcher("weather")
    render_weather_app()
