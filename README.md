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

#### Email configuration — Gmail SMTP (recommended, no signup)

The app reads these exact variable names: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` (and optionally `SMTP_FROM`).

Set up a free Gmail app password in ~3 minutes (no third-party accounts):

1. **Turn on 2-Step Verification** — go to your Google Account → Security → *2-Step Verification* and turn it on (required once for app passwords).
2. **Create an app password** — Google Account → Security → *App Passwords* → choose **Mail** (or Other → name it `energypulse`).
3. **Copy the 16-character password** it shows (e.g. `abcd efgh ijkl mnop`).
4. **Edit the `.env` file** in the project root (if missing, copy `.env.example` and fill it in):
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your@gmail.com
   SMTP_PASSWORD=the-16-character-app-password
   ```
   (Leave `SMTP_FROM` blank to send from your own Gmail address.)
5. **Save and restart the app.** The Notifications tab will now show "Connected" — click **Send test** to verify delivery.

> Use the 16-character **app password**, not your normal Gmail password. Normal passwords are rejected by Google for SMTP.

#### Email configuration — SendGrid (alternative)

If you prefer SendGrid instead of Gmail, create a free account at sendgrid.com, then in `.env` set the variable the app reads:

```
SENDGRID_API_KEY=your_sendgrid_api_key_here
```

Use **either** the SMTP block **or** `SENDGRID_API_KEY`, not both — the app prefers SendGrid when both are present. If no email variables are set at all, the app runs honestly in test mode: no email is sent, delivery status is recorded as failed, and the tab shows "Not configured".

#### SMS — optional (future step)

SMS uses Twilio (`TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`). It is intentionally optional — leave the variables blank and SMS simply stays "Not configured" in the app. It is not required for email delivery.

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
