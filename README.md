# ⚡ Home Energy Dashboard

A multi-page **Streamlit** web app for an AI-based home electricity usage
prediction project. It ships with a deterministic, physically plausible
**two-year hourly dataset** (17,520 rows) and a trained **Random Forest
regressor**, so every page works out of the box.

## New Features (V2.1)

**Three new fully-implemented features** have been added to EnergyPulse:

### 1. Family Member Management
- Add household members with individual notification preferences
- Each member can choose to receive bill alerts and/or optimization tips
- Personalized language preference per member
- Immediate removal stops all notifications (fully enforced, not just hidden)
- Validation: reject invalid emails and duplicates with language-aware error messages
- Linked to primary user via household ID (not separate, disconnected records)

### 2. Automated Notifications (Email & SMS)
- **Bill Alerts**: Triggered by predicted cost threshold or schedule, content uses only real computed numbers
- **Optimization Tips**: Triggered by real usage anomalies detected in data, each tip references specific real appliance and measured increase
- Email via SendGrid, SMTP, or test mode
- SMS via Twilio (optional)
- Each recipient in their preferred language
- Full audit trail in notification log (who, when, what real data triggered it, status)
- Graceful failure handling: errors logged, non-blocking status shown, app never crashes
- Test notification button to verify email delivery without waiting for real trigger
- "Run scheduled checks now" button triggers detection immediately from real data

#### Email configuration
Copy `.env.example` to `.env` and pick exactly ONE email provider (no env vars set means the app runs in test mode, where no email is sent but the audit log still records delivery):

| Mode | Required variables |
| --- | --- |
| **SendGrid** | `SENDGRID_API_KEY` |
| **SMTP** (Gmail/Outlook/custom) | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM` |
| **SMS (optional Twilio)** | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` |

Email verification prior to sending (all four languages) makes delivery failures visible in the Notifications tab instead of crashing the app.

### 3. Grounded Q&A Chatbot
- Natural language questions about household energy usage
- **Answers ONLY from real household data** via tool calls:
  - `get_usage_history`: Actual measurements
  - `get_appliance_breakdown`: Real appliance-level breakdown
  - `get_current_prediction`: Computed from recent data
  - `get_week_comparison`: This week vs. previous week (only when a real prior week exists)
  - `get_optimization_tips`: Detected anomalies, not generic suggestions
  - `get_family_notification_log`: Actual sent notifications for the household
- Multi-language support: questions understood and answered in all 4 languages
- **No hallucinations**: If data is unavailable (or there is no usable prior week), the chatbot says so honestly
- All numeric claims validated against tool output using LLMHallucinationGuard (including date/time tokens, which are excluded from number matching)
- Conversation history stored per user, clearable from the chat tab

### 3. Grounded Q&A Chatbot
- Natural language questions about household energy usage
- **Answers ONLY from real household data** via tool calls:
  - `get_usage_history`: Actual measurements
  - `get_appliance_breakdown`: Real appliance-level breakdown
  - `get_current_prediction`: Computed from recent data
  - `get_optimization_tips`: Detected anomalies, not generic suggestions
- Multi-language support: questions understood and answered in all 4 languages
- **No hallucinations**: If data unavailable, chatbot says so honestly
- All numeric claims validated against tool output using LLMHallucinationGuard
- Conversation history stored per user
- Includes "Tell me why" examples for common questions

See [FEATURES_NEW.md](FEATURES_NEW.md) for complete documentation.

## Pages

After signing in (or continuing as guest) the app shows a dashboard with eight tabs:

| Tab | What it does |
| --- | --- |
| 🏠 **Overview** | Project intro, headline metrics, pipeline explainer |
| 📊 **Analysis** | Interactive time series, appliance breakdown, hour×weekday heatmap, weekday profile, actual-vs-predicted overlay |
| 💡 **Save** | Optimization suggestions and cost reduction opportunities |
| 🧾 **Bills** | Cost details, tariff settings |
| 📈 **Trends** | 24-hour forecast with confidence band, recent trends |
| 👨‍👩‍👧 **Family** | Manage household members and their notification preferences |
| 📨 **Notifications** | Backend status, alert thresholds, run checks now, test emails, audit log |
| 💬 **Energy Chat** | Grounded multilingual Q&A about your real usage |

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
├── app.py                  # entry point (auth, dashboard + 8 tabs)
├── chatbot.py              # grounded multilingual chatbot (tool-calling pipeline)
├── features_ui.py          # Family / Notifications / Chat tab renderers
├── notifications.py        # email (SendGrid/SMTP) + SMS (Twilio) dispatch
├── llm_guard.py            # hallucination guard for numeric claims
├── i18n.py                 # English/Hindi/Kannada/Telugu translations
├── db.py                   # SQLite: users, family, notifications, chat
├── cost.py                 # tariff/weekly/peak cost helpers
├── data.py                 # synthetic dataset generation & loader
├── data_processing.py      # remapping of raw data to current dates
├── feature_manager.py      # household settings / feature config
├── optimize.py             # optimization/insight rules
├── model.py                # Random Forest training, metrics, forecasting
├── train_model.py          # CLI training script
├── test_new_features.py    # feature suite (5 test groups, all green)
├── requirements.txt
├── .env.example            # email/SMS credential template
└── .streamlit/config.toml  # dark theme
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

- Default price: **₹8 / kWh** (configurable per household in the app).
- Emission factor: **0.386 kg CO₂ / kWh** (typical grid mix).
