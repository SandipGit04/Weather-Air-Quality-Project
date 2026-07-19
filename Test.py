import os, datetime, joblib
from prophet.serialize import model_from_json

path = "aqi_models/Ahmedabad_forecast.json"
print("File last modified on disk:", datetime.datetime.fromtimestamp(os.path.getmtime(path)))

with open(path) as f:
    model = model_from_json(f.read())
print("Model's actual training cutoff:", model.history["ds"].max())