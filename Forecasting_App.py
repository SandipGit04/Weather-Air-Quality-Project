import streamlit as st
import pandas as pd
from prophet.serialize import model_from_json
from datetime import date
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AQI Forecasting System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Index Forecasting System")
st.markdown("Predict future AQI for Indian cities")

# -----------------------------
# Get Available Cities
# -----------------------------
model_folder = "forecast_models"

cities = []

for file in os.listdir(model_folder):
    if file.endswith("_forecast.json"):
        city = file.replace("_forecast.json", "")
        cities.append(city)

cities = sorted(cities)

# -----------------------------
# User Inputs
# -----------------------------
city = st.selectbox(
    "Select City",
    cities
)

selected_date = st.date_input(
    "Select Future Date",
    min_value=date.today()
)

# -----------------------------
# AQI Category Function
# -----------------------------
def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    else:
        return "Severe"

# -----------------------------
# Forecast Button
# -----------------------------
if st.button("Forecast AQI"):

    try:
        # Load Selected City Model
        model_path = f"{model_folder}/{city}_forecast.json"

        with open(model_path, "r") as fin:
            model = model_from_json(fin.read())

        # Create Future Dates
        future = model.make_future_dataframe(periods=365, freq='D')

        # Predict
        forecast = model.predict(future)

        # Match Selected Date
        result = forecast[forecast["ds"].dt.date == selected_date]


        if len(result) > 0:

            predicted_aqi = float(result["yhat"].iloc[0])

            category = get_aqi_category(predicted_aqi)

            st.success(f"Predicted AQI for {city} on {selected_date}: {predicted_aqi:.0f}")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Predicted AQI", f"{predicted_aqi:.0f}")

            with col2:
                st.metric("AQI Category", category)

            # Show Forecast Chart
            st.subheader(f"AQI Forecast Trend - {city}")

            chart_data = forecast[["ds", "yhat"]].tail(365)

            chart_data = chart_data.rename(
                columns={"ds": "Date", "yhat": "AQI"}
            )

            st.line_chart(chart_data.set_index("Date"))

        else:
            st.warning("Selected date is outside forecast range.")

    except (FileNotFoundError, ValueError) as e:
        st.error(f"Error: {str(e)}")
