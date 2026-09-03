# EnergyPulse - New Features Documentation

## Overview of New Features

This document describes three new interconnected features added to EnergyPulse:

1. **Family Member Management** - Add household members with individual notification preferences
2. **Automated Notifications** - Email/SMS alerts for bill changes and optimization opportunities
3. **Grounded Q&A Chatbot** - Natural language questions answered using only real household data

All features follow the **no-hallucination** principle: data is never invented or assumed; all claims are grounded in real measurements.

---

## 1. Family Member Management

### Overview
Primary users can now add family members to their household with individual preferences for receiving notifications.

### Features
- **Add family members**: name, relationship, email, phone (optional), language preference
- **Notification preferences**: each member can opt-in/out of bill alerts and optimization tips independently
- **Language support**: each member can select their preferred language (English/हिन्दी/ಕನ್ನಡ/తెలుగు)
- **Edit/Remove**: modify or remove members at any time; removal immediately stops notifications

### Database Storage
Family members are stored in SQLite with:
- Link to primary user via `household_id` (not separate, disconnected records)
- Individual notification preferences
- Audit trail in `notifications_log` table

### Validation
- Email format validation (rejects invalid formats with inline error message in selected language)
- Duplicate prevention (same email address + household combination rejected)
- Language-aware error messages

### Usage
```python
from db import get_db

db = get_db()
success, message = db.add_family_member(
    household_id="hh_12345",
    name="Spouse Name",
    relationship="Spouse",
    email="spouse@email.com",
    phone="+91-9999999999",
    preferred_language="en",
    notify_bills=True,
    notify_tips=True
)
```

---

## 2. Automated Notifications

### Overview
Notifications are triggered by real computed data and sent to all eligible household members via email (SMS optional).

### Notification Types

#### A. Bill Alerts
- **Trigger**: Predicted monthly cost exceeds threshold OR scheduled interval (e.g., weekly summary)
- **Content**: Actual computed predicted cost (never rounded guess)
- **Recipients**: Primary user + all family members who opted in
- **Language**: Each recipient's preferred language

Example trigger data:
```python
{
    "type": "bill_alert",
    "month": "2024-09",
    "predicted_cost": 1250.50,  # REAL computed value
    "threshold": 1200.00,
    "triggered_at": "2024-09-01T10:30:00"
}
```

#### B. Optimization Tips
- **Trigger**: Real usage anomaly detected (e.g., appliance usage spike 2x+ above normal)
- **Content**: Actual appliance name, actual usage increase percentage, specific recommended action
- **Recipients**: Primary user + all family members who opted in
- **Language**: Each recipient's preferred language

Example trigger data:
```python
{
    "type": "optimization_tip",
    "appliance": "Air Conditioner",           # REAL appliance name
    "current_usage_kwh": 3.5,                 # REAL measurement
    "avg_usage_kwh": 2.0,                     # REAL historical average
    "increase_pct": 75.0,                     # REAL calculated percentage
    "triggered_at": "2024-09-01T14:00:00"
}
```

### Email Configuration

Choose ONE of the following:

#### Option 1: SendGrid (Recommended)
```bash
# .env file
SENDGRID_API_KEY=SG.your_key_here
SMTP_FROM=noreply@energypulse.local
```

#### Option 2: SMTP (Gmail, Outlook, or custom server)
```bash
# .env file
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password  # For Gmail: use app password, not account password
SMTP_FROM=your_email@gmail.com
```

#### Option 3: SMS via Twilio (Optional)
```bash
# .env file
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM_NUMBER=+1234567890
```

### Notification Log

All notifications are logged for audit trail:
```python
from db import get_db

db = get_db()
log = db.get_notification_log(household_id="hh_12345", limit=100)

# Each log entry contains:
# - recipient_email: who received it
# - notification_type: "bill_alert" or "optimization_tip"
# - triggered_by: what real data triggered it
# - trigger_data_json: the actual real data (as JSON)
# - sent_at: when it was sent
# - status: "sent" or "failed"
# - error_message: if failed, why
```

### Test Notification

Users can send themselves a test notification to verify email delivery:
```python
from notifications import get_notification_service

service = get_notification_service()
success, message = service.send_test_notification(
    household_id="hh_12345",
    recipient_email="user@example.com"
)
```

### Failure Handling
- Email send failures log the error and display a non-blocking status message
- Never crashes the app
- User can retry or check email configuration
- All attempts logged in notification log for audit

### API Reference
```python
from notifications import get_notification_service

service = get_notification_service()

# Send bill alert
sent_to = service.send_bill_alert(
    household_id="hh_12345",
    primary_email="user@example.com",
    month="2024-09",
    predicted_cost=1250.50,
    threshold=1200.00,
    language="en"
)

# Send optimization tip
sent_to = service.send_optimization_tips(
    household_id="hh_12345",
    primary_email="user@example.com",
    tip_data={
        "appliance_name": "Air Conditioner",
        "current_usage": 3.5,
        "avg_usage": 2.0,
        "increase_pct": 75.0,
        "suggested_action": "Consider reducing AC usage to save..."
    },
    language="en"
)

# Send test notification
success, msg = service.send_test_notification(
    household_id="hh_12345",
    recipient_email="user@example.com"
)
```

---

## 3. Grounded Q&A Chatbot

### Overview
A natural language chatbot that answers questions about household energy usage using ONLY real data from measurements and computations. Never guesses or uses general knowledge.

### Supported Question Types
- "Which appliance uses the most power?"
- "What's my predicted monthly cost?"
- "Why was my bill higher last week?"
- "How can I reduce my AC usage cost?"
- "What's my average daily consumption?"
- "Which hours do I use the most energy?"

### Answer Grounding

The chatbot works with tools that query real data:

1. **get_usage_history**: Actual hourly/daily consumption from measurements
2. **get_appliance_breakdown**: Real usage by appliance category from sub-metering
3. **get_current_prediction**: Predicted cost calculated from recent usage
4. **get_optimization_tips**: Real anomalies detected from data patterns

### Multi-Language Support
- Questions understood in all 4 languages (English/हिन्दी/ಕನ್ನಡ/తెలుగు)
- Answers provided fully in the user's selected language
- No fallback to English mid-conversation

### Validation & Hallucination Prevention

All responses are validated using the LLMHallucinationGuard:
1. Extract numeric claims from chatbot answer
2. Verify each number against actual tool output
3. If claims don't match real data, response is corrected or marked invalid
4. Users see a warning if answer was corrected

### Conversation History

Stored in database per user session:
```python
from chatbot import get_chatbot

chatbot = get_chatbot()
history = chatbot.get_conversation_history(
    household_id="hh_12345",
    email="user@example.com",
    limit=20
)

# Each turn contains:
# - question: user's question
# - answer: chatbot's answer
# - tool_calls: which tools were used
# - grounding_data: real data returned by tools
# - is_valid_grounded: whether answer was validated
```

### Honest Responses

When data can't answer a question, the chatbot says so:
```
User: "What will the weather be tomorrow?"
Bot: "I don't have that information. I can only answer questions about your household's electricity usage based on your real meter data."
```

Not:
```
Bot: "Based on typical weather patterns..." (HALLUCINATION!)
```

### API Reference
```python
from chatbot import get_chatbot

chatbot = get_chatbot()

# Answer a question
answer, is_valid, metadata = chatbot.answer_question(
    household_id="hh_12345",
    email="user@example.com",
    question="Which appliance uses the most power?",
    language="en",
    tariff_rate=8.0  # Rs/kWh for cost predictions
)

# answer: The chatbot's response (string)
# is_valid: Whether response was validated (bool)
# metadata: Tool calls, grounding data, validation issues

# Get conversation history
history = chatbot.get_conversation_history(
    household_id="hh_12345",
    email="user@example.com",
    limit=20
)

# Clear conversation history
success = chatbot.clear_conversation_history(
    household_id="hh_12345",
    email="user@example.com"
)
```

---

## Database Schema

### New Tables

#### `family_members`
```sql
CREATE TABLE family_members (
    id INTEGER PRIMARY KEY,
    household_id TEXT NOT NULL,
    name TEXT NOT NULL,
    relationship TEXT,
    email TEXT NOT NULL,
    phone TEXT,
    preferred_language TEXT DEFAULT 'en',
    notification_bill_alerts BOOLEAN DEFAULT 1,
    notification_optimization_tips BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (household_id) REFERENCES users(household_id),
    UNIQUE(household_id, email)
);
```

#### `notifications_log`
```sql
CREATE TABLE notifications_log (
    id INTEGER PRIMARY KEY,
    household_id TEXT NOT NULL,
    recipient_email TEXT NOT NULL,
    recipient_name TEXT,
    notification_type TEXT NOT NULL,
    triggered_by TEXT,
    trigger_data_json TEXT,
    subject TEXT NOT NULL,
    body_text TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'sent',
    error_message TEXT,
    FOREIGN KEY (household_id) REFERENCES users(household_id)
);
```

#### `chatbot_conversations`
```sql
CREATE TABLE chatbot_conversations (
    id INTEGER PRIMARY KEY,
    household_id TEXT NOT NULL,
    email TEXT NOT NULL,
    message_type TEXT NOT NULL,
    language TEXT DEFAULT 'en',
    question TEXT,
    answer TEXT,
    tool_calls_json TEXT,
    grounding_data_json TEXT,
    is_valid_grounded BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (household_id) REFERENCES users(household_id)
);
```

---

## Testing

Run the comprehensive test suite:
```bash
python test_new_features.py
```

This validates:
- Database operations (create user, add member, log notifications)
- Family member management (add, edit, remove)
- Notification system (email configuration, sending, logging)
- Chatbot (multi-language Q&A, grounding, validation)
- LLM Guard (hallucination detection)

---

## Integration with Existing App

The three features are integrated as follows:

1. **Settings Panel**: Users can manage family members and test notifications
2. **Sidebar**: Family management accessible from main navigation
3. **Chat Tab** (optional): Dedicated chatbot interface or accessible from main menu
4. **Automatic Triggers**: Notifications can be triggered by app logic when anomalies detected

---

## Security & Privacy

- Emails stored in plain text (production: encrypt)
- Passwords hashed with SHA-256 (production: use bcrypt/argon2)
- API keys stored in .env, NOT committed to git
- .gitignore includes .env and .env.* files
- All personal data (emails, names) linked to household via ID

---

## Known Limitations & Future Improvements

1. **Email**: Currently no support for HTML emails (plaintext only)
2. **SMS**: Twilio integration available but requires credentials
3. **Chatbot**: Uses keyword matching for tool selection (future: NLU/intent classification)
4. **Scheduling**: Notifications are event-triggered only (future: scheduled digests)
5. **Preferences**: No timezone support yet (future: per-member timezone)

---

## Support & Troubleshooting

### Email not sending?
1. Check .env file has correct credentials
2. Run test notification: `service.send_test_notification(...)`
3. Check notification log in database for error message
4. For Gmail: use app password, not account password

### Chatbot giving wrong answers?
1. Check that real data exists (run get_usage_history tool)
2. Verify LLM guard didn't correct response (check metadata)
3. Test chatbot with: `python test_new_features.py`

### Family member not receiving notifications?
1. Verify member's notification preference checkbox is enabled
2. Check notification log: was notification sent to that email?
3. Check email spam folder
4. Verify family member was added correctly (no duplicate emails)
