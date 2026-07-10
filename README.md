# 🌍 Weather and Air Pollution Analysis

## 📌 Project Overview

Air pollution has become one of the major environmental challenges affecting public health and quality of life. This project focuses on analyzing weather conditions and air pollution data across multiple Indian cities to identify patterns, trends, and relationships between meteorological factors and air quality indicators.

The project performs comprehensive Exploratory Data Analysis (EDA) to uncover meaningful insights from weather and pollution datasets and provides a foundation for future AQI prediction and forecasting systems.

---

## 🎯 Objectives

* Analyze weather conditions across multiple cities.
* Study the distribution of major air pollutants.
* Examine Air Quality Index (AQI) trends.
* Identify relationships between weather variables and pollution levels.
* Compare pollution levels across different cities.
* Discover key factors influencing air quality.
* Build a foundation for future AQI prediction and forecasting systems.

---

## 📊 Dataset Information

The dataset contains weather and air pollution observations collected from multiple Indian cities.

### Dataset Characteristics

* Total Records: 7,600
* Cities Covered: 20
* Time Period: June 2025 – June 2026
* Daily Observations

### Features Included

#### Weather Features

* Temperature (°C)
* Humidity (%)
* Wind Speed (m/s)
* Pressure (hPa)
* Cloud Coverage (%)
* Weather Condition
* Weather Description

#### Pollution Features

* PM2.5 (µg/m³)
* PM10 (µg/m³)
* NO (µg/m³)
* NO₂ (µg/m³)
* SO₂ (µg/m³)
* CO (µg/m³)
* NH₃ (µg/m³)

#### Air Quality Feature

* AQI
* AQI Category

---

## 📂 Project Ecosystem
**This repository contains multiple applications. You can access the live versions below:**

| Application | Deployment Link | Status |
| :--- | :--- | :--- |
| **AQI Forecasting System** | **[AQI Forecasting](https://example-frontend.com)** | **🟢 Operational** |
| **Weather Forecasting System** | **[Weather Forecasting](https://example-admin.com)** | **🟢 Operational** |

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-Learn
* Prophet
* Streamlit

### Development Tools

* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

### Weather Analysis

* Temperature trends over time
* Humidity analysis
* Wind speed distribution
* Weather condition frequency analysis
* City-wise weather comparison

### Pollution Analysis

* PM2.5 distribution analysis
* PM10 distribution analysis
* Pollutant concentration trends
* City-wise pollution comparison
* Top polluted cities identification

### AQI Analysis

* AQI distribution
* AQI category analysis
* AQI comparison across cities
* Proportion of poor and severe AQI records

### Correlation Analysis

* Pollutant correlation heatmap
* Relationship between AQI and pollutants
* Weather vs pollution analysis

---

## 📈 Key Insights

* PM2.5 and PM10 showed the strongest positive correlation with AQI.
* Certain cities consistently recorded higher pollution levels compared to others.
* AQI variations were strongly influenced by particulate matter concentrations.
* Weather conditions such as temperature, humidity, and wind speed showed varying relationships with pollutant concentrations.
* Significant differences in air quality were observed across cities and seasons.

---

## 📂 Project Structure

```text
Weather-Air-Pollution-Analysis/
│
├── data/
│   └── Weather_Pollution_Data.csv
│
├── notebooks/
│   └── Weather_Analysis_Project.ipynb
│
├── forecast_models/
│   ├── City_Forecast_Models
│
├── insights/
│   └── Visualizations
│
├── Weather_App.py
├── Forecasting_App.py
│
├── aqi_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Future Scope

### AQI Prediction Model

Machine learning algorithms can be implemented to predict AQI using weather and pollution parameters such as temperature, humidity, wind speed, PM2.5, PM10, NO₂, SO₂, and CO.

### AQI Forecasting System

Time-series forecasting techniques can be used to predict future AQI levels for different cities, enabling proactive environmental monitoring and planning.

### Interactive Web Dashboard

A web-based dashboard can be developed to provide real-time visualization, AQI prediction, and forecasting capabilities through an intuitive user interface.

### Weather-Based Pollution Forecasting

Future enhancements may integrate weather forecasting data with historical pollution records to estimate future pollutant concentrations and air quality levels.

---

## 📚 Learning Outcomes

Through this project, the following skills were developed:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis
* Data Visualization
* Statistical Analysis
* Time-Series Analysis
* Machine Learning Fundamentals
* AQI and Environmental Data Analysis
* Git and GitHub Version Control

---

## 👨‍💻 Author

Developed as part of a Data Analysis and Environmental Monitoring Project using Python, Machine Learning, and Data Visualization techniques.
