import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta
import plotly.graph_objects as go

from Weather_Logic import (
    feels_like, cloud_coverage, rain_chance, rain_perception,
    weather_condition, wind_gust, uv_index, weather_description, sunrise_sunset
)

# The server this app runs on may be in any timezone (often UTC on cloud
# hosts). Since this is an India-focused app, "now" must always reflect
# IST (UTC+5:30), not the server's local clock, or the sun position and
# "current time" displays will be hours off from reality.
IST = "Asia/Kolkata"

def now_ist():
    return pd.Timestamp.now(tz="UTC").tz_convert(IST)

st.set_page_config(
    page_title="Weather Forecast India",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    background: #4fd1c5; box-shadow: 0 0 8px #4fd1c5aa;
}
.topbar-sub { font-size: 12px; color: #4b5468; }

/* BIG page header (matches AQI app hero) */
.page-hero {
    background: linear-gradient(135deg, #0d2137 0%, #0a1628 50%, #071020 100%);
    border: 1px solid rgba(79,209,197,0.15);
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
    background: radial-gradient(circle, rgba(79,209,197,0.10) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.page-hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,209,197,0.12); border: 1px solid rgba(79,209,197,0.3);
    color: #4fd1c5; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 0.3rem 0.9rem; border-radius: 999px; margin-bottom: 1rem;
}
.page-hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.5rem; font-weight: 800; color: #f0f9ff;
    margin: 0 0 0.5rem 0; line-height: 1.15;
    display: flex; align-items: center; gap: 12px;
}
.page-hero-title .accent { color: #4fd1c5; }
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

/* hero current-weather glass card */
.hero-glass {
    position: relative;
    border-radius: 24px;
    padding: 30px 32px;
    margin-bottom: 18px;
    overflow: hidden;
    background:
        radial-gradient(120% 140% at 8% 0%, rgba(79,209,197,0.16) 0%, transparent 55%),
        radial-gradient(90% 120% at 95% 100%, rgba(99,124,255,0.14) 0%, transparent 55%),
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
    background: rgba(79,209,197,0.12);
    border: 1px solid rgba(79,209,197,0.3);
    color: #4fd1c5;
    font-size: 12px; font-weight: 600;
    padding: 5px 13px; border-radius: 999px;
    letter-spacing: 0.02em;
}
.hero-main-row {
    display: flex; align-items: center; gap: 26px; margin-bottom: 6px;
}
.hero-icon { font-size: 58px; line-height: 1; filter: drop-shadow(0 4px 14px rgba(79,209,197,0.25)); }
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
.sec-head i { color: #4fd1c5; font-size: 12px; }

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
.f-row.is-today .f-day { color: #4fd1c5; }
.f-day { font-size: 13.5px; font-weight: 600; color: #cfd6e4; }
.f-date { font-size: 11px; color: #4b5468; display: block; margin-top: 1px; }
.f-icon { font-size: 19px; text-align: center; }
.f-cond { font-size: 12.5px; color: #8a93a8; }
.f-rain { font-size: 12px; color: #5a9fd6; text-align: right; font-weight: 600; font-family: 'JetBrains Mono', monospace;}
.f-temp-bar-wrap { display: flex; align-items: center; gap: 8px; }
.f-temp-track { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,0.07); position: relative; overflow: hidden;}
.f-temp-fill { position:absolute; top:0; bottom:0; border-radius: 2px; }
.f-temp-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #eef1f7; min-width: 34px; text-align: right;}

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
.bd-row.bd-today { background: rgba(79,209,197,0.05); }
.bd-day-cell { text-align: center; }
.bd-day-cell .bd-day-name { font-size: 13.5px; font-weight: 700; color: #cfd6e4; display:block; }
.bd-row.bd-today .bd-day-cell .bd-day-name { color: #4fd1c5; }
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
.panel-title i { color: #4fd1c5; }
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
.app-footer-text .accent { color: #4fd1c5; font-weight: 700; }
.app-footer-sub {
    font-size: 0.8rem;
    color: #4b5468;
    margin-top: 0.5rem;
}

/* select widget */
div[data-baseweb="select"] > div {
    background-color: #131a2c !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 11px !important;
}
div[data-baseweb="select"] span { color: #dbe1ec !important; font-family: 'Inter', sans-serif !important; font-size: 13.5px !important;}
div[data-baseweb="select"] svg  { color: #5b6478 !important; }

#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
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

def predict_variable(model, days=7):
    if model is None:
        return pd.DataFrame({"ds": [], "yhat": []})
    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)
    return forecast[["ds", "yhat"]].tail(days).reset_index(drop=True)

def predict_hourly_today(model, hours=24, day_offset=0):
    """Interpolate an hourly curve for the selected day (day_offset days
    from today) by blending that day's and the next day's prediction."""
    if model is None:
        return None
    future = model.make_future_dataframe(periods=day_offset + 2, freq="D")
    forecast = model.predict(future)
    v_day  = float(forecast["yhat"].iloc[day_offset - 2])
    v_next = float(forecast["yhat"].iloc[day_offset - 1])
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

    temp_fc = predict_variable(models["Temperature"], days_ahead)
    hum_fc  = predict_variable(models["Humidity"], days_ahead)
    pres_fc = predict_variable(models["Pressure"], days_ahead)
    wind_fc = predict_variable(models["WindSpeed"], days_ahead)

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

# Jump-to-day selector uses the REAL dates from daily[] (not independently
# computed date.today() offsets), so labels always match the actual forecast
# regardless of how far the model's training data extends.
jump_labels = ["Today"] + [d["date"].strftime("%A, %d %b") for d in daily[1:]]
with col_jump:
    jump_choice = st.selectbox("Jump to day", jump_labels, label_visibility="visible")
selected_day_idx = jump_labels.index(jump_choice)

hourly_temp = predict_hourly_today(models["Temperature"], 24, day_offset=selected_day_idx)
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
            <div class="hero-place"><i class="fas fa-location-dot" style="font-size:18px;color:#4fd1c5"></i> {city}</div>
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
    marker=dict(color="rgba(58,90,140,0.55)"),
    yaxis="y2",
    hovertemplate="%{x}<br>Rain %{y:.0f}%<extra></extra>",
))

# Temperature line with point labels
fig_hourly.add_trace(go.Scatter(
    x=hours_labels, y=hourly_temp,
    mode="lines+markers+text",
    line=dict(color="#f0b95c", width=3, shape="spline", smoothing=0.5),
    marker=dict(size=6, color="#f0b95c", line=dict(color="#0b0e17", width=1.5)),
    text=[f"{t:.0f}\u00b0" for t in hourly_temp],
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

st.html(f"""
<div class="sun-card">
    <svg viewBox="0 0 400 90" xmlns="http://www.w3.org/2000/svg" width="100%" height="80">
        <defs>
            <linearGradient id="arc" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#f0b95c" stop-opacity="0.15"/>
                <stop offset="50%" stop-color="#f0b95c" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="#f0b95c" stop-opacity="0.15"/>
            </linearGradient>
            <radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="{sun_color}" stop-opacity="0.9"/>
                <stop offset="100%" stop-color="{sun_color}" stop-opacity="0"/>
            </radialGradient>
        </defs>
        <line x1="15" y1="72" x2="385" y2="72" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
        <path d="M 20 72 Q 200 5 380 72" fill="none" stroke="url(#arc)" stroke-width="2" stroke-dasharray="3 4"/>

        <!-- animated sun with rays, matching reference style -->
        <circle cx="{arc_x:.1f}" cy="{arc_y:.1f}" r="22" fill="url(#sunGlow)" opacity="{sun_glow}">
            <animate attributeName="r" values="20;24;20" dur="3s" repeatCount="indefinite"/>
        </circle>
        <g transform="translate({arc_x:.1f},{arc_y:.1f})">
            <g stroke="{sun_color}" stroke-width="2" stroke-linecap="round" opacity="{sun_glow}">
                <line x1="0" y1="-14" x2="0" y2="-10"/>
                <line x1="0" y1="10" x2="0" y2="14"/>
                <line x1="-14" y1="0" x2="-10" y2="0"/>
                <line x1="10" y1="0" x2="14" y2="0"/>
                <line x1="-9.9" y1="-9.9" x2="-7.1" y2="-7.1"/>
                <line x1="7.1" y1="7.1" x2="9.9" y2="9.9"/>
                <line x1="9.9" y1="-9.9" x2="7.1" y2="-7.1"/>
                <line x1="-7.1" y1="7.1" x2="-9.9" y2="9.9"/>
            </g>
            <circle cx="0" cy="0" r="8" fill="{sun_color}"/>
        </g>

        <circle cx="20" cy="72" r="4" fill="#f0b95c" opacity="0.7"/>
        <circle cx="380" cy="72" r="4" fill="#f0b95c" opacity="0.7"/>
    </svg>
    <div class="sun-times-row">
        <div class="sun-block"><div class="sun-lab">Sunrise</div><div class="sun-val">{sr}</div></div>
        <div class="sun-block">
            <div class="sun-lab">{'Now' if is_showing_today else 'Peak'}</div>
            <div class="sun-val" style="color:{'#4fd1c5' if is_showing_today else '#7c8598'}; font-size: 20px;">
                {now_label if is_showing_today else '&mdash;'}
            </div>
        </div>
        <div class="sun-block"><div class="sun-lab">Sunset</div><div class="sun-val">{ss}</div></div>
    </div>
</div>
""")


# --------------------------------------------------------------
# APP FOOTER (matches AQI app closing description style)
# --------------------------------------------------------------
st.html("""
<div class="app-footer">
    <div class="app-footer-icon">&#9925;&#65039;</div>
    <div class="app-footer-text">
        Weather intelligence for <span class="accent">India</span>, powered by real forecasting models
    </div>
    <div class="app-footer-sub">Predictions powered by Meta Prophet time-series model</div>
</div>
""")
