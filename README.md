# 🌍 India Forecasting Hub

**A unified weather and air-quality forecasting platform for 20 Indian cities**, powered by Meta's Prophet time-series models. Built as a single Streamlit application with an automated, self-updating machine learning pipeline running on a schedule via GitHub Actions.

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-FF4B4B?style=for-the-badge)](https://forecasting-platform.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-0467DF?style=for-the-badge)](https://facebook.github.io/prophet/)

---

## 🔗 Live Links

| | |
|---|---|
| 🚀 **Live App** | [forecasting-platform.streamlit.app](https://forecasting-platform.streamlit.app/) |
| 🌐 **ClimaSphere (companion site)** | *Coming soon* |
| 📂 **Repository** | [github.com/SandipGit04/Weather-Air-Quality-Project](https://github.com/SandipGit04/Weather-Air-Quality-Project) |

---

## 📸 Screenshots

<!--
  Add 1-3 screenshots here once available — a landing page shot and at
  least one forecast result screen. GitHub READMEs with visuals get
  noticeably more engagement from recruiters and visitors skimming a
  profile, so this is worth prioritizing once you have clean screenshots.

  Suggested embed format once images are added to an `Insights/` or
  `screenshots/` folder in the repo:

  ![Landing Page](Insights/landing_page.png)
  ![Weather Forecast](Insights/weather_forecast.png)
  ![AQI Forecast](Insights/aqi_forecast.png)
-->

*Screenshots coming soon.*

---

## 📖 Overview

**India Forecasting Hub** brings together two independent forecasting systems — **Weather** and **Air Quality Index (AQI)** — into a single, unified web application. Instead of maintaining two separate deployments, the project uses a session-state-driven landing page and switcher, letting a user move between both systems within one continuous session, on one URL.

Both systems are powered by **per-city Prophet models**, trained on historical weather and pollution data collected through a companion backend, and are kept fresh through a **fully automated retraining pipeline** that runs on a schedule — no manual retraining required.

### What it does

- **🌦️ Weather Forecasting** — a 6-day rolling outlook per city, with hourly temperature curves, rain probability, wind speed, atmospheric pressure, UV index, and real-time sunrise/sunset tracking.
- **🌍 AQI Forecasting** — predicts Air Quality Index up to 365 days ahead per city, with pollution category breakdowns (Good → Severe), confidence intervals, and monthly seasonal AQI pattern charts.
- **🔁 Self-updating models** — a GitHub Actions workflow fetches fresh data, cleans it, and retrains every per-city model on a schedule, committing updated model files back to the repository automatically.

---

## ✨ Features

### Weather Forecasting System
- 20 Indian cities, each with its own independently trained model
- 6-day rolling forecast horizon with hourly granularity
- Temperature, humidity, pressure, and wind speed predictions
- Real-time sunrise/sunset tracking, adjusted to IST regardless of server timezone
- Daily breakdown table with condition icons and min/max temperatures

### AQI Forecasting System
- Same 20-city coverage, forecasting up to 365 days ahead
- AQI category classification (Good, Satisfactory, Moderate, Poor, Severe) with color-coded visual scale
- Confidence bands (upper/lower bound) on every prediction, not just a single point estimate
- Monthly seasonal AQI pattern chart to visualize longer-term trends

### Platform-level features
- **Unified navigation** — a landing page lets users choose a system, with a persistent switcher bar to jump between Weather and AQI without losing their place
- **Fully automated ML pipeline** — see [Architecture](#architecture-and-automated-pipeline) below
- **IST-aware throughout** — all "current time" and "today" logic is explicitly pinned to Indian Standard Time, independent of whatever timezone the hosting server runs in

---

## 🏗️ Architecture and Automated Pipeline

A core design goal of this project was to avoid manual retraining entirely. The system is split into two halves that never block each other:

```
┌─────────────────────────────────────────────────────────────┐
│  SCHEDULED PIPELINE  (GitHub Actions — runs independently   │
│  of site traffic, on a cron schedule)                       │
│                                                               │
│   Fetch_Data.py  ──►  Clean_Data.py  ──►  Training Scripts   │
│   (pulls raw CSV      (cleans nulls,      (retrains every    │
│    from backend)       dupes, applies      per-city Prophet  │
│                         cascading fallback  model)           │
│                         imputation)                          │
│                              │                                │
│                              ▼                                │
│               Commits updated Datasets/Forecasting_Data.csv, │
│               weather_models/*.pkl, aqi_models/*.json        │
│               back to the repository                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LIVE STREAMLIT APP  (India_Forecasting_Hub.py)              │
│                                                                │
│   Only ever READS whatever model/data files currently exist  │
│   on disk. It never trains anything itself, so page loads    │
│   stay fast regardless of how many people visit at once.     │
└─────────────────────────────────────────────────────────────┘
```

**Why this split matters:** training happens on a clock, not on a page visit. This means the live app never blocks a user while a model retrains, and multiple visitors can't trigger overlapping/conflicting training runs — a real risk if training were ever tied to page loads instead.

### Data flow, in detail

1. **`Fetch and Cleaning Scripts/Fetch_Data.py`** pulls the latest raw CSV export from a companion backend service that independently collects live weather and pollution data.
2. **`Fetch and Cleaning Scripts/Clean_Data.py`** applies the cleaning logic developed in the project's analysis notebooks — duplicate removal, time-based interpolation for wind speed, circular-mean handling for wind direction, and cascading median fallback imputation (city → weather condition → city-wide) for cloud coverage and select pollutants.
3. **`Training Model Scripts/Weather_Model_Training.py`** and **`Training Model Scripts/AQI_Model_Training.py`** retrain one Prophet model per city per forecasted variable, saving the results to `weather_models/` and `aqi_models/` respectively.
4. The workflow commits the refreshed dataset and model files back to the repository, and the live Streamlit app picks up the change on its next load.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / App Framework** | Streamlit |
| **Forecasting Models** | Meta Prophet (per-city, per-variable time series) |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Plotly |
| **Automation** | GitHub Actions (scheduled CI/CD for data + model refresh) |
| **Model Serialization** | joblib (`.pkl`), Prophet's native JSON serializer |
| **Language** | Python 3.10+ |

---

## 📂 Repository Structure

```
Weather-Air-Quality-Project/
│
├── India_Forecasting_Hub.py        # Main entrypoint — the unified app
├── Weather_Forecast_App.py         # Standalone Weather app (legacy / reference)
├── AQI_Forecast_App.py             # Standalone AQI app (legacy / reference)
├── Weather_Logic.py                # Shared weather calculation helpers
├── Test.py                         # Testing/scratch script
├── requirements.txt
├── README.md
│
├── .github/workflows/
│   └── pipeline.yml                # Scheduled fetch → clean → retrain automation
├── .streamlit/
│   └── config.toml
├── .devcontainer/
│   └── devcontainer.json
│
├── Notebooks/                       # Original data analysis & model development
│   ├── Weather_Analysis_Project.ipynb
│   ├── Weather_Forecasting_Model.ipynb
│   └── AQI_Forecasting_Model.ipynb
│
├── Fetch and Cleaning Scripts/
│   ├── Fetch_Data.py                # Pulls latest raw data from backend
│   └── Clean_Data.py                # Cleaning/imputation, notebook logic as a script
│
├── Training Model Scripts/
│   ├── Weather_Model_Training.py    # Retrains all per-city weather models
│   └── AQI_Model_Training.py        # Retrains all per-city AQI models
│
├── Datasets/
│   └── Forecasting_Data.csv         # Cleaned dataset (also read live by the app)
│
├── weather_models/                  # Per-city Prophet models, one folder per variable
│   ├── Temperature/*.pkl
│   ├── Humidity/*.pkl
│   ├── Pressure/*.pkl
│   └── WindSpeed/*.pkl
│
├── aqi_models/                      # Per-city Prophet AQI models
│   └── {City}_forecast.json
│
├── Temporary Test Models/
│   └── AQI/aqi_model.pkl            # Early exploratory RandomForest model (see note below)
│
└── Insights/                        # Supporting charts/visuals from EDA
```

> **Note on `Temporary Test Models/AQI/aqi_model.pkl`:** this was an early exploratory approach — a single combined RandomForest regressor predicting AQI from live feature readings (temperature, humidity, pollutants) across all cities at once. It was superseded by the per-city Prophet approach (`aqi_models/`) for architectural consistency with the Weather system, and is kept in the repo as a reference of that exploration rather than as an active part of the app.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the repository
```bash
git clone https://github.com/SandipGit04/Weather-Air-Quality-Project.git
cd Weather-Air-Quality-Project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> Prophet has native (non-Python) build dependencies on some platforms. If installation fails, consult [Prophet's official installation guide](https://facebook.github.io/prophet/docs/installation.html) for platform-specific instructions.

### 4. Run the app locally
```bash
streamlit run India_Forecasting_Hub.py
```

The app will open automatically in your default browser, typically at `http://localhost:8501`.

### 5. (Optional) Set up the automated pipeline yourself

If you want to run the retraining pipeline on your own fork:

1. Fork this repository.
2. Add the following repository secrets under **Settings → Secrets and variables → Actions**:
   - `BACKEND_CSV_URL` — your data source's export endpoint
   - `BACKEND_API_KEY` — if your data source requires authentication
3. Enable **Read and write permissions** under **Settings → Actions → General → Workflow permissions**, so the workflow can commit refreshed data/models back to the repo.
4. The workflow in `.github/workflows/pipeline.yml` will then run on its defined schedule, or can be triggered manually from the **Actions** tab.

---

## 📊 Data Source

The raw historical weather and air-quality data used to train these models is collected and exported by a companion backend service:

```
https://climasphere-vk5q.onrender.com/download/downloadCSV
```

This backend independently gathers live weather and pollution readings for the covered cities over time, which are then fetched, cleaned, and used for model training as described in the [Architecture](#architecture-and-automated-pipeline) section above.

---

## 🎯 Design Decisions & Scope

Every real project involves tradeoffs. Documenting them here rather than leaving them implicit:

**Why Prophet, and not a physics-based weather model?**
Real operational weather forecasting (e.g. by national meteorological services) relies on physics-based numerical weather prediction models like GFS or ECMWF, which require atmospheric input data and significant compute infrastructure. Given a single historical time series per city, Prophet was chosen for its strength in modeling seasonal and trend patterns without that infrastructure requirement. This project is scoped as a **seasonal-trend forecasting tool**, not a competitor to official meteorological forecasts — it hasn't been benchmarked against alternative time-series approaches (e.g. ARIMA, LSTM-based models), which is a natural next step if forecasting accuracy needed to be rigorously validated.

**Why session-state routing instead of Streamlit's multipage (`st.Page`) API?**
This app was originally built and deployed as two *separate* Streamlit Cloud apps with cross-links between them. In practice, Streamlit Cloud runs deployed apps inside its own iframe wrapper, which made reliable same-tab navigation between the two separately-hosted apps unreliable — attempts using `target="_self"`, forced JavaScript navigation, and `target="_top"` each surfaced different failure modes (silent no-ops, redirect loops). The two systems were consolidated into a single app using `st.session_state`-based routing specifically because it removes the cross-origin/iframe boundary entirely — both systems now share one process and one URL, so there's no navigation boundary left to fail.

**Why a static, periodically-refreshed CSV instead of live per-request API calls?**
The forecasting models need substantial historical data to train on, not just a live snapshot. Reusing an existing data-collection backend's periodic CSV export was more practical than querying a live API on every page load, and keeps model training decoupled from user traffic (see [Architecture](#architecture-and-automated-pipeline)).

**Current testing status of the automated pipeline**
The fetch → clean → retrain → commit pipeline has been built and each stage verified individually (schema validation, cleaning logic parity against the original notebooks, successful model file generation). A full, unattended production run on the live schedule is the next verification step, rather than something already confirmed over an extended period.

---

## 🗺️ Roadmap

- [ ] Add ClimaSphere companion site link once deployed
- [ ] Add project documentation / written report
- [ ] Add presentation deck (PPT)
- [ ] Benchmark Prophet against alternative forecasting approaches
- [ ] Expand city coverage beyond the current 20

---

## 👤 Author

**Sandip Kundu**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sandipin04/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SandipGit04)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/SandipX04)
[![Gmail](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kundusandip004@gmail.com)

---

## 📄 License

*No license file currently specified. Add one (e.g. MIT, Apache 2.0) if you intend for others to reuse or contribute to this code.*
