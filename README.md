# ⚡ Home Energy Dashboard

A multi-page **Streamlit** web app for an AI-based home electricity usage
prediction project. It ships with a deterministic, physically plausible
**two-year hourly dataset** (17,520 rows) and a trained **Random Forest
regressor**, so every page works out of the box.

## Pages

| Page | What it does |
| --- | --- |
| 🏠 **Overview** | Project intro, headline metrics, pipeline explainer |
| 📊 **Dashboard** | Interactive time series, appliance breakdown, hour×weekday heatmap, weekday profile, actual-vs-predicted overlay |
| 🤖 **AI Prediction** | Single-hour forecast with cost/CO₂ + live 24-hour forecast with 95% confidence band |
| 📈 **Model Insights** | MAE / RMSE / R², feature importance, correlation matrix, actual-vs-predicted diagnostics |
| 🗂️ **Data Explorer** | Filter, summarize and download the raw dataset |

## Getting started

Requires Python 3.10+.

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the dashboard
streamlit run app.py
```

The app then opens at `http://localhost:8501`.

## Project structure

```
majorproject/
├── app.py                  # entry point (set_page_config + st.navigation)
├── requirements.txt
├── .streamlit/config.toml  # dark theme
├── core/
│   ├── data.py             # synthetic dataset generation & loader
│   ├── features.py         # feature engineering (shared train/predict)
│   ├── model.py            # Random Forest training, metrics, forecasting
│   ├── plotting.py         # Plotly chart helpers
│   └── runtime.py          # Streamlit-cached dataset/model accessors
└── pages/
    ├── home.py
    ├── dashboard.py
    ├── prediction.py
    ├── insights.py
    └── data_explorer.py
```

## How the model works

- **Data**: hourly usage modelled as the sum of seven appliance categories
  (HVAC, water heating, lighting, kitchen, laundry, electronics, other) driven
  by time of day, weekday, season, temperature and humidity.
- **Features (14)**: `hour`, `day_of_week`, `month`, `day_of_year`,
  `hour_sin/cos`, `day_sin/cos`, `is_weekend`, `is_holiday`, `temperature_c`,
  `humidity_pct`, `rolling_mean_24h`, `lag_1h`.
- **Model**: Random Forest regressor (200 trees), trained on the first 80% of
  hours (time-ordered), evaluated on the most recent 20%. The held-out
  residual spread sizes the forecast confidence band (±1.96 × std ≈ 95%).

## Using real data

Replace `core/data.load_dataset` (or `core/runtime.get_dataset`) with a loader
for your own meter / smart-plug export. The rest of the app expects these
columns:

```
timestamp, temperature_c, humidity_pct, is_weekend, is_holiday,
hvac_kwh, water_heating_kwh, lighting_kwh, kitchen_kwh, laundry_kwh,
electronics_kwh, other_kwh, electricity_usage_kwh
```

> The bundled dataset is synthetic and generated with a fixed seed for
> reproducibility — figures shown are illustrative, not real consumption.

## Costs & emissions assumptions

- Default price: **$0.168 / kWh** (configurable in the app).
- Emission factor: **0.386 kg CO₂ / kWh** (typical grid mix).
