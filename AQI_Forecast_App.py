import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
from datetime import date, timedelta
import os
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

# The server this app runs on may be in any timezone (often UTC on cloud
# hosts). Since this is an India-focused app, "today" must always reflect
# IST (UTC+5:30), not the server's local clock, or the default forecast
# date and the 30-day chart window could be off by a day near midnight.
IST = "Asia/Kolkata"

def today_ist():
    return pd.Timestamp.now(tz="UTC").tz_convert(IST).date()

# Read the dataset from the specified path
weather_df = pd.read_csv("Datasets/Forecasting_Data.csv")

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AQI Forecasting System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# Custom CSS — Dark Teal Theme
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

/* ============================================================
   AQI HERO HEADER - UPDATED WITH CROSS-APP NAV BUTTONS
   ============================================================ */

.hero-header {
    background: linear-gradient(135deg, #0d2137 0%, #0a1628 50%, #071020 100%);
    border: 1px solid rgba(0, 200, 170, 0.15);
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
    background: radial-gradient(circle, rgba(0,200,170,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 300px; height: 200px;
    background: radial-gradient(ellipse, rgba(56,189,248,0.05) 0%, transparent 70%);
}


.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 200, 170, 0.12);
    border: 1px solid rgba(0, 200, 170, 0.3);
    color: #00c8aa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #f0f9ff;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
    position: relative;
    z-index: 1;
}
.hero-title .accent { color: #4fd1c5; }

.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.01em;
    margin: 0;
    position: relative;
    z-index: 1;
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
    box-shadow:0 0 30px rgba(0,210,211,.15);
    border-color:#00d2d3;
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
.g-teal { border-color: rgba(0, 210, 211, 0.3); box-shadow: 0 0 20px rgba(0, 210, 211, 0.05), inset 0 0 10px rgba(0, 210, 211, 0.05); }
.g-teal .icon-circle { border-color: #00d2d3; color: #00d2d3; box-shadow: 0 0 10px rgba(0, 210, 211, 0.2); }
.g-teal .glass-title, .g-teal .glass-footer { color: #00d2d3; }
.g-teal .glass-footer { border-color: rgba(0, 210, 211, 0.2); }

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
   FORECAST CONTROLS
   Select City + Date
   ================== */

/* Labels */
.stSelectbox label,
.stDateInput label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}

/* Select + Date same container size */
.stSelectbox div[data-baseweb="select"] > div,
.stDateInput div[data-baseweb="input"] {
    background: #1b2942 !important;
    border: 2px solid rgba(24,214,197,0.28) !important;
    border-radius: 16px !important;
    min-height: 58px !important;
    height: 58px !important;
    padding: 0 18px !important;
    display: flex !important;
    align-items: center !important;
    box-shadow: 0 0 8px rgba(24,214,197,0.08) !important;
    transition: all 0.22s ease !important;
}

/* Hover/focus glow - controlled, not huge */
.stSelectbox div[data-baseweb="select"] > div:hover,
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stDateInput div[data-baseweb="input"]:hover,
.stDateInput div[data-baseweb="input"]:focus-within {
    border-color: #18d6c5 !important;
    box-shadow:
        0 0 0 1px rgba(24,214,197,0.12),
        0 0 12px rgba(24,214,197,0.22) !important;
}

/* Select City text */
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] input {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Date text */
.stDateInput input {
    background: transparent !important;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 0 !important;
    caret-color: #18d6c5 !important;
}

/* Dropdown/calendar icon */
.stSelectbox svg,
.stDateInput button svg {
    color: #18d6c5 !important;
    fill: #18d6c5 !important;
}

/* ==================
   FORECAST BUTTON ONLY
   ================== */

.stButton {
    height: 58px !important;
    margin-top: 6px !important;   /* aligns with input boxes */
}

.stButton > button {
    width: 100% !important;
    height: 58px !important;
    min-height: 58px !important;

    background: linear-gradient(135deg, #00c8aa, #0284c7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;

    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: 0.01em !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    padding: 0 22px !important;
    margin: 0 !important;

    box-shadow: 0 8px 20px rgba(0, 200, 170, 0.22) !important;
    transition: all 0.22s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow:
        0 0 0 1px rgba(24,214,197,0.16),
        0 0 18px rgba(24,214,197,0.28),
        0 10px 24px rgba(2,132,199,0.24) !important;
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

.metric-card.teal::before  { background: linear-gradient(90deg, #00c8aa, #0284c7); }
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
    border-left: 3px solid #00c8aa;
}

/* ── Info Cards ── */
.info-strip {
    background: rgba(0, 200, 170, 0.05);
    border: 1px solid rgba(0, 200, 170, 0.12);
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
    if not (min_date <= default_date <= max_date):
        default_date = min_date
    selected_date = st.date_input(
        "Select Forecast Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date
    )
    st.caption(f" Today: {default_date.strftime('%d %b %Y')} · 📡 Latest available data for {city}: {last_known_date.strftime('%d %b %Y')} · forecasting {(default_date - last_known_date).days} days ahead to today")

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    forecast_clicked = st.button("⚡ Forecast AQI", use_container_width=True)

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
color:#00d2d3;
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
                <div class="legend-chip"><div class="legend-dot" style="background:#00c8aa"></div>Predicted AQI</div>
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
                fillcolor="rgba(0,200,170,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="Confidence Band"
            ))

            # AQI line
            fig.add_trace(go.Scatter(
                x=chart_df["ds"],
                y=chart_df["yhat"],
                mode="lines",
                line=dict(color="#00c8aa", width=2.5),
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
                        activecolor="#00c8aa",
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
                    bordercolor="rgba(0,200,170,0.3)",
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
                    bordercolor="rgba(0,200,170,0.3)",
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
            Select a city and date, then click <span style="color:#00c8aa">Forecast AQI</span>
        </div>
        <div style="font-size:0.85rem; color:#334155; margin-top:0.5rem;">
            Predictions powered by Meta Prophet time-series model
        </div>
    </div>
    """, unsafe_allow_html=True)