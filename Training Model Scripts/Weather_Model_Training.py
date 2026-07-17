"""
Converted from Weather_Forecasting_Model.ipynb - trains one Prophet
model per (city, parameter) combination and saves it as a .pkl,
exactly matching the notebook's final working approach (cells 48-52).

Input : Datasets/Forecasting_Data.csv  (produced by clean_data.py)
Output: weather_models/{Parameter}/{City}.pkl for Parameter in [Temperature, Humidity, Pressure, WindSpeed]
"""

import os
import warnings

import pandas as pd
import joblib
from pathlib import Path
from prophet import Prophet

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Datasets" / "Forecasting_Data.csv"
MODEL_ROOT = ROOT_DIR / "weather_models"
PARAMETERS = ["Temperature", "Humidity", "Pressure", "WindSpeed"]


def train_weather_model(df: pd.DataFrame, city: str, parameter: str) -> str:
    city_df = df[df["City"] == city].copy()
    prophet_df = city_df[["Date", parameter]].copy()
    prophet_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(prophet_df)

    path = f"{MODEL_ROOT}/{parameter}/{city}.pkl"
    joblib.dump(model, path)
    return path


def main():
    print(f"Reading cleaned data from {DATA_PATH} ...")
    weather_forecast = pd.read_csv(DATA_PATH)

    # Ensure output folders exist
    for parameter in PARAMETERS:
        if not os.path.exists(f"{MODEL_ROOT}/{parameter}"):
            os.makedirs(f"{MODEL_ROOT}/{parameter}", exist_ok=True)

    cities = sorted(weather_forecast["City"].unique())
    print(f"Training for {len(cities)} cities: {cities}")

    for parameter in PARAMETERS:
        print(f"\nTraining {parameter} models...")
        for city in cities:
            path = train_weather_model(weather_forecast, city, parameter)
            print(f"  Saved -> {path}")

    print("\nAll weather models trained successfully.")


if __name__ == "__main__":
    main()