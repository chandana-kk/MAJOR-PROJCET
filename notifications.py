"""
Notification System for EnergyPulse
===================================
Sends automated notifications (email, optional SMS) to the primary user
and currently enrolled family members only.

Notification types (always grounded in computed figures):
  1. Bill alerts — predicted monthly cost from next_month_cost / forecast
  2. Weekly summary — weekly_cost on stored usage
  3. Optimization tips — detect_anomalies on stored usage

Removed family members are never recipients: the recipient list is always
re-queried from family_members at send time.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from db import get_db
from i18n import t_lang


def _round2(value: float) -> float:
    return round(float(value), 2)


class NotificationService:
    """Sends notifications via email and optional SMS. Never raises to callers."""

    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_FROM = os.getenv("SMTP_FROM") or os.getenv("SMTP_USERNAME") or "noreply@energypulse.local"
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

    def __init__(self):
        self.db = get_db()
        self.email_backend = None
        self.sms_backend = None
        self.sg_client = None
        self.twilio_client = None
        self._init_email_backend()
        self._init_sms_backend()

    def _init_email_backend(self):
        key = (self.SENDGRID_API_KEY or "").strip()
        if key and key != "your_sendgrid_api_key_here":
            try:
                from sendgrid import SendGridAPIClient
                self.email_backend = "sendgrid"
                self.sg_client = SendGridAPIClient(key)
                return
            except ImportError:
                self.email_backend = None
        host = (self.SMTP_HOST or "").strip()
        user = (self.SMTP_USERNAME or "").strip()
        password = (self.SMTP_PASSWORD or "").strip()
        if host and user and password and password != "your_app_password_here":
            self.email_backend = "smtp"

    def _init_sms_backend(self):
        sid = (self.TWILIO_ACCOUNT_SID or "").strip()
        token = (self.TWILIO_AUTH_TOKEN or "").strip()
        if sid and token and sid != "your_account_sid_here":
            try:
                from twilio.rest import Client
                self.sms_backend = "twilio"
                self.twilio_client = Client(sid, token)
            except ImportError:
                self.sms_backend = None

    def recipients_for(self, household_id: str, primary_email: str,
                       preference_field: str, default_language: str) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Live recipient list: primary user plus family members who still exist
        and opted into preference_field. Removed members cannot appear here.
        Returns (email, name, language, phone).
        """
        recipients = []
        primary_email = (primary_email or "").strip().lower()
        primary_user = self.db.get_user(primary_email)
        if primary_user:
            lang = primary_user.get("preferred_language") or default_language or "en"
            recipients.append((
                primary_email,
                primary_user.get("name") or "User",
                lang,
                None,
            ))
        for member in self.db.get_family_members(household_id):
            if not member.get(preference_field):
                continue
            email = (member.get("email") or "").strip().lower()
            if not email or email == primary_email:
                continue
            lang = member.get("preferred_language") or default_language or "en"
            recipients.append((
                email,
                member.get("name") or email,
                lang,
                member.get("phone"),
            ))
        return recipients

    def send_bill_alert(self, household_id: str, primary_email: str,
                        month: str, predicted_cost: float, threshold: Optional[float] = None,
                        language: str = "en", predicted_kwh: Optional[float] = None) -> List[str]:
        predicted_cost = _round2(predicted_cost)
        predicted_kwh = _round2(predicted_kwh) if predicted_kwh is not None else None
        threshold = _round2(threshold) if threshold is not None else None
        trigger_data = {
            "type": "bill_alert",
            "month": month,
            "predicted_cost": predicted_cost,
            "predicted_kwh": predicted_kwh,
            "threshold": threshold,
            "triggered_at": datetime.now().isoformat(),
        }
        recipients = self.recipients_for(
            household_id, primary_email, "notification_bill_alerts", language
        )
        sent_to = []
        for email, name, lang, phone in recipients:
            if threshold is not None:
                body_core = t_lang(
                    "email_bill_body", lang, month=month, cost=predicted_cost, threshold=threshold
                )
            else:
                body_core = t_lang("email_bill_body_no_th", lang, month=month, cost=predicted_cost)
            subject = t_lang("email_bill_subject", lang, month=month)
            body = self._compose(name, body_core, lang)
            ok, err = self._deliver(email, subject, body, phone)
            self.db.log_notification(
                household_id=household_id,
                recipient_email=email,
                recipient_name=name,
                notification_type="bill_alert",
                triggered_by="predicted_monthly_cost",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=lang,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                sent_to.append(email)
        return sent_to

    def send_weekly_summary(self, household_id: str, primary_email: str,
                            week_kwh: float, week_cost: float, pct_change: float,
                            language: str = "en") -> List[str]:
        week_kwh = _round2(week_kwh)
        week_cost = _round2(week_cost)
        pct_change = round(float(pct_change), 1)
        trigger_data = {
            "type": "weekly_summary",
            "week_kwh": week_kwh,
            "week_cost": week_cost,
            "pct_change": pct_change,
            "triggered_at": datetime.now().isoformat(),
        }
        recipients = self.recipients_for(
            household_id, primary_email, "notification_bill_alerts", language
        )
        sent_to = []
        for email, name, lang, phone in recipients:
            subject = t_lang("email_weekly_subject", lang)
            body_core = t_lang(
                "email_weekly_body", lang, kwh=week_kwh, cost=week_cost, pct=pct_change
            )
            body = self._compose(name, body_core, lang)
            ok, err = self._deliver(email, subject, body, phone)
            self.db.log_notification(
                household_id=household_id,
                recipient_email=email,
                recipient_name=name,
                notification_type="bill_alert",
                triggered_by="weekly_usage_summary",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=lang,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                sent_to.append(email)
        return sent_to

    def send_optimization_tips(self, household_id: str, primary_email: str,
                               tip_data: Dict[str, Any], language: str = "en") -> List[str]:
        required = ["appliance_name", "current_usage", "avg_usage", "increase_pct"]
        if not all(field in tip_data for field in required):
            return []
        current = _round2(tip_data["current_usage"])
        avg = _round2(tip_data["avg_usage"])
        increase = round(float(tip_data["increase_pct"]), 1)
        when = tip_data.get("when") or datetime.now().isoformat()
        appliance = tip_data["appliance_name"]
        trigger_data = {
            "type": "optimization_tip",
            "appliance": appliance,
            "current_usage_kw": current,
            "avg_usage_kw": avg,
            "increase_pct": increase,
            "when": when,
            "triggered_at": datetime.now().isoformat(),
        }
        recipients = self.recipients_for(
            household_id, primary_email, "notification_optimization_tips", language
        )
        sent_to = []
        for email, name, lang, phone in recipients:
            subject = t_lang("email_tip_subject", lang, appliance=appliance)
            body_core = t_lang(
                "email_tip_body", lang,
                appliance=appliance, current=current, avg=avg,
                increase_pct=increase, when=when,
            )
            body = self._compose(name, body_core, lang)
            ok, err = self._deliver(email, subject, body, phone)
            self.db.log_notification(
                household_id=household_id,
                recipient_email=email,
                recipient_name=name,
                notification_type="optimization_tip",
                triggered_by="usage_anomaly",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=lang,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                sent_to.append(email)
        return sent_to

    def send_test_notification(self, household_id: str, recipient_email: str,
                               language: str = "en",
                               predicted_cost: Optional[float] = None,
                               predicted_kwh: Optional[float] = None,
                               month: Optional[str] = None) -> Tuple[bool, str]:
        """
        Send a test email whose body contains real computed prediction figures.
        If those figures are not provided, they are loaded from stored forecast data.
        """
        try:
            if predicted_cost is None or month is None:
                stats = load_prediction_stats(tariff_rate=8.0)
                predicted_cost = stats["predicted_cost"]
                predicted_kwh = stats["predicted_kwh"]
                month = stats["month"]
            predicted_cost = _round2(predicted_cost)
            predicted_kwh = _round2(predicted_kwh or 0)
            when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subject = t_lang("email_test_subject", language)
            body_core = t_lang(
                "email_test_body", language,
                when=when, month=month, cost=predicted_cost, kwh=predicted_kwh,
            )
            body = self._compose("EnergyPulse user", body_core, language)
            ok, err = self._deliver(recipient_email, subject, body, None)
            trigger_data = {
                "type": "test",
                "predicted_cost": predicted_cost,
                "predicted_kwh": predicted_kwh,
                "month": month,
                "triggered_at": datetime.now().isoformat(),
            }
            self.db.log_notification(
                household_id=household_id,
                recipient_email=recipient_email,
                recipient_name="Test recipient",
                notification_type="test",
                triggered_by="manual_test",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=language,
                status="sent" if ok else "failed",
                error_message=err,
            )
            if ok:
                return (True, "sent")
            return (False, err or "email_failed")
        except Exception as e:
            return (False, str(e))

    def _compose(self, name: str, body_core: str, language: str) -> str:
        hello = t_lang("email_hello", language, name=name)
        footer = t_lang("email_footer", language)
        return f"{hello}\n\n{body_core}\n\n{footer}"

    def _deliver(self, to_email: str, subject: str, body: str,
                 phone: Optional[str]) -> Tuple[bool, Optional[str]]:
        try:
            email_ok = self._send_email(to_email, subject, body)
            if phone and self.sms_backend:
                try:
                    self.send_sms(phone, subject[:150])
                except Exception:
                    pass
            if email_ok:
                return (True, None)
            if not self.email_backend:
                return (False, "No email backend configured (set SENDGRID_API_KEY or SMTP_* in .env)")
            return (False, "Email send failed")
        except Exception as e:
            return (False, str(e))

    def _send_email(self, to_email: str, subject: str, body_text: str) -> bool:
        try:
            if self.email_backend == "sendgrid":
                return self._send_via_sendgrid(to_email, subject, body_text)
            if self.email_backend == "smtp":
                return self._send_via_smtp(to_email, subject, body_text)
            return False
        except Exception as e:
            print(f"[ERROR] Email send failed to {to_email}: {e}")
            return False

    def _send_via_sendgrid(self, to_email: str, subject: str, body_text: str) -> bool:
        try:
            from sendgrid.helpers.mail import Mail, Email, To, Content
            message = Mail(
                from_email=Email(self.SMTP_FROM),
                to_emails=To(to_email),
                subject=subject,
                plain_text_content=Content("text/plain", body_text),
            )
            response = self.sg_client.send(message)
            return 200 <= response.status_code < 300
        except Exception as e:
            print(f"[ERROR] SendGrid send failed: {e}")
            return False

    def _send_via_smtp(self, to_email: str, subject: str, body_text: str) -> bool:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.SMTP_FROM
            msg["To"] = to_email
            msg.attach(MIMEText(body_text, "plain"))
            with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT, timeout=20) as server:
                server.starttls()
                server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)
                server.sendmail(self.SMTP_FROM, [to_email], msg.as_string())
            return True
        except Exception as e:
            print(f"[ERROR] SMTP send failed: {e}")
            return False

    def send_sms(self, to_phone: str, message: str) -> bool:
        if self.sms_backend != "twilio":
            return False
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.TWILIO_FROM_NUMBER,
                to=to_phone,
            )
            return True
        except Exception as e:
            print(f"[ERROR] SMS send failed to {to_phone}: {e}")
            return False


def load_prediction_stats(tariff_rate: float = 8.0) -> Dict[str, Any]:
    """Load predicted monthly cost from the stored forecast CSV (same as the dashboard)."""
    import os
    import pandas as pd
    from cost import next_month_cost
    path = os.path.join("data", "next_month_forecast.csv")
    if os.path.exists(path):
        forecast_df = pd.read_csv(path, parse_dates=["datetime"])
        info = next_month_cost(forecast_df, tariff_rate)
        return {
            "predicted_cost": _round2(info.get("total_cost") or 0),
            "predicted_kwh": _round2(info.get("total_kwh") or 0),
            "month": info.get("month") or "N/A",
        }
    return {"predicted_cost": 0.0, "predicted_kwh": 0.0, "month": "N/A"}


def dispatch_household_alerts(household_id: str, primary_email: str,
                              predicted_cost: float, predicted_kwh: float,
                              month: str, tariff_rate: float,
                              history_df, language: str = "en") -> Dict[str, Any]:
    """
    Trigger bill/weekly/optimization alerts from real dashboard numbers.
    Failures never raise. Returns a status dict for a non-blocking UI message.
    """
    status = {"sent": [], "failed": False, "errors": []}
    try:
        db = get_db()
        service = get_notification_service()
        settings = db.get_household_settings(household_id)
        threshold = float(settings.get("bill_threshold") or 1500.0)
        now = datetime.now()

        if predicted_cost >= threshold:
            last = settings.get("last_bill_alert_at")
            due = True
            if last:
                try:
                    due = datetime.fromisoformat(str(last)) < now - timedelta(hours=20)
                except Exception:
                    due = True
            if due:
                sent = service.send_bill_alert(
                    household_id, primary_email, month,
                    predicted_cost, threshold, language, predicted_kwh,
                )
                status["sent"].append(("bill_alert", sent))
                db.update_household_settings(
                    household_id, {"last_bill_alert_at": now.isoformat()}
                )

        last_week = settings.get("last_weekly_summary_at")
        weekly_due = True
        if last_week:
            try:
                weekly_due = datetime.fromisoformat(str(last_week)) < now - timedelta(days=7)
            except Exception:
                weekly_due = True
        if weekly_due and history_df is not None and not history_df.empty:
            from cost import weekly_cost
            latest = history_df["datetime"].dt.date.iloc[-1]
            week_start = latest - timedelta(days=6)
            week_info = weekly_cost(history_df, str(week_start), tariff_rate)
            sent = service.send_weekly_summary(
                household_id, primary_email,
                week_kwh=week_info.get("total_kwh") or 0,
                week_cost=week_info.get("total_cost") or 0,
                pct_change=week_info.get("pct_change") or 0,
                language=language,
            )
            status["sent"].append(("weekly_summary", sent))
            db.update_household_settings(
                household_id, {"last_weekly_summary_at": now.isoformat()}
            )

        if history_df is not None and not history_df.empty and len(history_df) > 100:
            from optimize import detect_anomalies
            anomalies = detect_anomalies(history_df)
            if anomalies is not None and not anomalies.empty:
                row = anomalies.iloc[-1]
                key = f"{row['datetime']}|{row['top_appliance']}"
                if key != settings.get("last_optimization_key"):
                    usage = float(row["Global_active_power"])
                    mean = float(row["rolling_mean"])
                    pct = ((usage - mean) / mean * 100.0) if mean else 0.0
                    sent = service.send_optimization_tips(
                        household_id, primary_email,
                        {
                            "appliance_name": row["top_appliance"],
                            "current_usage": usage,
                            "avg_usage": mean,
                            "increase_pct": pct,
                            "when": str(row["datetime"]),
                        },
                        language=language,
                    )
                    status["sent"].append(("optimization_tip", sent))
                    db.update_household_settings(
                        household_id, {"last_optimization_key": key}
                    )
    except Exception as e:
        status["failed"] = True
        status["errors"].append(str(e))
    return status


_notification_service = None


def get_notification_service() -> NotificationService:
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
