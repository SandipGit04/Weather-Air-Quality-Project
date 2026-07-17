"""
Converted from AQI_Forecasting_Model.ipynb
1) A single combined RandomForestRegressor across all cities, predicting AQI from other weather/pollutant readings.
    -> saved as aqi_model.pkl

2) A separate per-city Prophet time-series model, predicting AQI purely from historical AQI over time (like the weather models).
    -> saved as forecast_models/{City}_forecast.json

Input : Cleaned_Weather_Data.csv (produced by clean_data.py)
Output: aqi_model.pkl & forecast_models/{City}_forecast.json
"""

import os
import warnings

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from prophet import Prophet
from prophet.serialize import model_to_json

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Datasets" / "Forecasting_Data.csv"
RF_FEATURES = ["Temperature", "Humidity", "WindSpeed", "PM25", "PM10", "NO2", "SO2", "CO"]
RF_MODEL_PATH = ROOT_DIR / "Temporary Test Models" / "AQI" / "aqi_model.pkl"
PROPHET_MODEL_DIR = ROOT_DIR / "aqi_models"


def train_random_forest(df: pd.DataFrame):
    
    #Single combined AQI model across all cities (approach 1).
    X = df[RF_FEATURES]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"  RandomForest R^2: {r2_score(y_test, y_pred):.4f}")
    print(f"  RandomForest MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    if not os.path.exists(RF_MODEL_PATH.parent):
        os.makedirs(RF_MODEL_PATH.parent, exist_ok=True)
    joblib.dump(model, RF_MODEL_PATH)
    print(f"  Saved -> {RF_MODEL_PATH}")


def train_prophet_per_city(df: pd.DataFrame):

    # Separate per-city AQI time-series model (approach 2).
    if not os.path.exists(PROPHET_MODEL_DIR):
        os.makedirs(PROPHET_MODEL_DIR, exist_ok=True)

    cities = df["City"].unique()
    for city in cities:
        city_data = df[df["City"] == city][["Date", "AQI"]].copy()
        city_data.columns = ["ds", "y"]

        model = Prophet()
        model.fit(city_data)

        path = f"{PROPHET_MODEL_DIR}/{city}_forecast.json"
        with open(path, "w") as fout:
            fout.write(model_to_json(model))
        print(f"  Saved -> {path}")


def main():
    print(f"Reading cleaned data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print("\nTraining combined RandomForest AQI model...")
    train_random_forest(df)

    print("\nTraining per-city Prophet AQI models...")
    train_prophet_per_city(df)

    print("\nAll AQI models trained successfully.")


if __name__ == "__main__":
    main()