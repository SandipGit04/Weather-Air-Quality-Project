import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("aqi_model.pkl")
st.title("AQI Prediction System")

temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind = st.number_input("WindSpeed")
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")

if st.button("Predict AQI"):

    input_data = pd.DataFrame({
        'Temperature': [temp],
        'Humidity': [humidity],
        'WindSpeed': [wind],
        'PM25': [pm25],
        'PM10': [pm10],
        'NO2': [no2],
        'SO2': [so2],
        'CO': [co]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted AQI: {prediction[0]:.2f}")