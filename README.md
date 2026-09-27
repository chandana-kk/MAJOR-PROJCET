# ⚡ EnergyPulse — Home Energy Dashboard

A **Streamlit** web app for household electricity forecasting, appliance
breakdown, bill forecasting and anomaly tips, in **English, Hindi, Kannada and
Telugu**.

It ships with the **UCI Individual household electric power consumption**
dataset (34,168 hourly rows, December 2006 – November 2010) and a trained
**XGBoost + LSTM hybrid**, so every tab works out of the box.

> The bundled data is a real meter recording from one household, used as a
> **demo sample**. The dashboard labels it as sample data, shifts its timestamps
> forward so it lines up with the present, and never presents it as a live
> meter feed. To use your own readings, upload a CSV — see
> [Using your own data](#using-your-own-data).

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
5. **Save and restart the app.** The Notifications tab will show "Configured — not yet tested" — click **Send test** to verify delivery. It only reports "Connected" after a message has actually gone out.

> Use the 16-character **app password**, not your normal Gmail password. Normal passwords are rejected by Google for SMTP.

#### Email configuration — SendGrid (alternative)

If you prefer SendGrid instead of Gmail, create a free account at sendgrid.com, then in `.env` set the variable the app reads:

```
SENDGRID_API_KEY=your_sendgrid_api_key_here
```

Use **either** the SMTP block **or** `SENDGRID_API_KEY`, not both — the app prefers SendGrid when both are present. If no email variables are set at all, the app runs honestly in test mode: no email is sent, delivery status is recorded as failed, and the tab shows "Not configured".

A mistyped `SMTP_PORT` (anything that is not a number between 1 and 65535) is
reported as a warning and falls back to port 587. It does not stop the app.

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
- Questions that match nothing in scope (weather, sports, general trivia, an empty box) are refused in the user's own language rather than answered with household figures
- All numeric claims validated against tool output using LLMHallucinationGuard (including date/time tokens, which are excluded from number matching)
- Conversation history stored per user, clearable from the chat tab

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
├── data.py                 # UCI download, hourly aggregation, loaders
├── data_source.py          # upload normalisation, units, capability detection
├── appliances.py           # NILM sub-meter → appliance breakdown
├── optimize.py             # optimization/insight rules
├── model.py                # hybrid training, metrics, forecasting
├── replay.py               # replays the active frame row by row
├── convert_to_docx.py      # exports the conference paper to .docx
├── test_csv_import.py      # 26 tests: upload formats, units, timestamps
├── test_chatbot_routing.py # 38 tests: intent routing + guard number checks
├── test_notifications.py   # 12 tests: backend status, ports, log types
├── test_new_features.py    # feature suite (6 test groups, all green)
├── requirements.txt
├── .env.example            # email/SMS credential template
└── .streamlit/config.toml  # dark theme
```

## How the model works

- **Data**: the UCI household dataset, 1-minute readings aggregated to
  **34,168 hourly rows**. The three `Sub_metering_*` columns are reported by the
  source in **Wh per minute**; they are multiplied by 60 to get hourly Wh, which
  is why the appliance breakdown no longer collapses into a single "Other" bar.
- **Model inputs (10)**: the last 10 hours (`WINDOW_SIZE`) of
  `Global_active_power`, `hour`, `day_of_week`, `is_weekend`,
  `Global_reactive_power`, `Voltage`, `Global_intensity` and the three
  sub-meters.
- **Model**: an **XGBoost regressor blended with an LSTM**, weighted by
  inverse MAE. Trained time-ordered on the first 80% of hours, evaluated on the
  most recent 20%. Current metrics from `models/model_meta.pkl`:
  - XGBoost MAE **0.0120**, LSTM MAE **0.3399**
  - hybrid MAE **0.0163** at weights **0.966 / 0.034** (a flat 50/50 blend
    scores 0.1697, roughly ten times worse)
- **Honesty guards**: prediction is withheld unless every one of the 10 input
  columns is present; appliance breakdown and optimization need all three
  sub-meters. Missing inputs produce a localized, actionable message instead of
  a number.
- The LSTM is optional at inference. If it is unavailable or the dataset shape
  is unsupported, the app falls back to XGBoost and records the reason.

## Using your own data

Upload a CSV from the **Data** panel. The uploader accepts `.csv` and `.tsv`,
sniffs the delimiter, and reads Latin-1 as well as UTF-8.

- **Timestamps** may be a single `Date`+`Time` pair or one combined column.
  ISO dates are read unambiguously; for `dd/mm` vs `mm/dd` the order is
  inferred per column, and an ambiguous month-first reading is flagged in the
  provenance banner rather than silently reinterpreted.
- **Units** are detected per column. Sub-hourly data is resampled to hourly:
  `Wh` values are summed and `W` values averaged, with a warning and a manual
  override when the unit is not certain.
- **Column names** are matched through an alias table, so `total_kwh`,
  `Total Consumption` and `Global_active_power` all map to the same field.
- **Timestamps are preserved.** The dashboard never rewrites an upload's dates;
  the current-time match used for the sample does not apply to uploads, whose
  headline reading is simply the newest row in the file.

Anything the upload cannot supply is withheld with a message naming the missing
columns, rather than filled with a default or estimated number. The panel always
shows the file name, row count, time range, and units actually applied.

## Bill photo OCR (optional)

Uploading a bill **image** runs OCR to prefill units, amount and billing period.
This needs the Tesseract engine, which is a separate program from the Python
packages:

- Windows: `winget install UB-Mannheim.TesseractOCR`
- macOS: `brew install tesseract`
- Debian/Ubuntu: `sudo apt install tesseract-ocr`

Without the engine the app says OCR is unavailable and lets you fill the fields
in manually. It never guesses at the contents of an image it could not read.

## Costs & emissions assumptions

- Default price: **₹8 / kWh** (configurable per household in the app).
- Emission factor: **0.386 kg CO₂ / kWh** (typical grid mix).
