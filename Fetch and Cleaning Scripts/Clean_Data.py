
import os
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT_DIR / "Datasets" / "Weather_Pollution_Data.csv"
CLEANED_PATH = ROOT_DIR / "Datasets" / "Forecasting_Data.csv"


def circular_mean(series):
    """Circular interpolation for wind direction (0 deg == 360 deg)."""
    vals = series.dropna()
    if len(vals) == 0:
        return series.fillna(0)
    rad = np.deg2rad(vals)
    sin_sum = np.sin(rad).mean()
    cos_sum = np.cos(rad).mean()
    mean_angle = np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360
    return series.fillna(mean_angle)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Parse timestamps (same format as source notebook)
    data["CreatedAt"] = pd.to_datetime(data["CreatedAt"], format="%d/%m/%Y, %I:%M:%S %p")
    data["UpdatedAt"] = pd.to_datetime(data["UpdatedAt"], format="%d/%m/%Y, %I:%M:%S %p")

    # Extract plain Date from UpdatedAt
    data["Date"] = data["UpdatedAt"].dt.date

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Sort by City and timestamp
    data = data.sort_values(["City", "UpdatedAt", "Date"])

    # --- WindSpeed: time-based interpolation per city, fallback to median ---
    data = data.set_index("UpdatedAt")
    data["WindSpeed"] = (
        data.groupby("City")["WindSpeed"]
            .transform(lambda x: x.interpolate(method="time").fillna(x.median()))
    )
    data = data.reset_index()
    data["WindSpeed"] = data["WindSpeed"].round(2)

    # --- WindDirection: circular mean per city+date ---
    data["WindDirection"] = (
        data.groupby(["City", "Date"])["WindDirection"].transform(circular_mean)
    )
    data["WindDirection"] = data["WindDirection"].round(0)

    # --- CloudCoverage: cascading median fallback ---
    data["CloudCoverage"] = data["CloudCoverage"].fillna(
        data.groupby(["City", "WeatherCondition"])["CloudCoverage"].transform("median")
    )
    data["CloudCoverage"] = data["CloudCoverage"].fillna(
        data.groupby(["City", "Date"])["CloudCoverage"].transform("median")
    )
    data["CloudCoverage"] = data["CloudCoverage"].fillna(
        data.groupby("City")["CloudCoverage"].transform("median")
    )
    data["CloudCoverage"] = data["CloudCoverage"].fillna(data["CloudCoverage"].median())
    data["CloudCoverage"] = data["CloudCoverage"].round(0)

    # --- NO: cascading median fallback ---
    data["NO"] = data["NO"].fillna(
        data.groupby(["City", "WeatherDescription"])["NO"].transform("median")
    )
    data["NO"] = data["NO"].fillna(
        data.groupby(["City", "Date"])["NO"].transform("median")
    )
    data["NO"] = data["NO"].fillna(data.groupby("City")["NO"].transform("median"))
    data["NO"] = data["NO"].fillna(data["NO"].median())
    data["NO"] = data["NO"].round(2)

    # --- NH3: cascading median fallback ---
    data["NH3"] = data["NH3"].fillna(
        data.groupby(["City", "WeatherDescription"])["NH3"].transform("median")
    )
    data["NH3"] = data["NH3"].fillna(
        data.groupby(["City", "Date"])["NH3"].transform("median")
    )
    data["NH3"] = data["NH3"].fillna(data.groupby("City")["NH3"].transform("median"))
    data["NH3"] = data["NH3"].fillna(data["NH3"].median())
    data["NH3"] = data["NH3"].round(2)

    # Final duplicate check (post-cleaning), same as notebook
    data = data.drop_duplicates()

    return data


def main():
    print(f"Reading raw data from {RAW_PATH} ...")
    raw = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {raw.shape}")

    cleaned = clean(raw)
    print(f"Cleaned shape: {cleaned.shape}")

    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)

    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Saved cleaned data -> {CLEANED_PATH}")


if __name__ == "__main__":
    main()