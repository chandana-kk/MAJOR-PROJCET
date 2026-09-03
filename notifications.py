"""
Notification System for EnergyPulse
===================================
Sends automated notifications (email/SMS) to users and family members.

Notification types:
  1. Bill alerts — triggered by cost threshold or scheduled
  2. Optimization tips — triggered by real usage patterns

All notifications:
  - Use real computed data (never invented numbers)
  - Are logged for audit trail
  - Support per-recipient language preferences
  - Fail gracefully without crashing the app
  - Include a unique trigger reference for grounding

Configuration via .env:
  SENDGRID_API_KEY=<your-key>         # For email via SendGrid
  SMTP_HOST=<host>                    # For email via SMTP
  SMTP_PORT=<port>
  SMTP_USERNAME=<user>
  SMTP_PASSWORD=<password>
  SMTP_FROM=<sender@email.com>
  TWILIO_ACCOUNT_SID=<sid>            # For SMS (optional)
  TWILIO_AUTH_TOKEN=<token>
  TWILIO_FROM_NUMBER=<+1234567890>
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from db import get_db
from i18n import T


class NotificationService:
    """
    Sends notifications via email and SMS.
    Integrates with database for logging.
    """
    
    # Email configuration
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_FROM = os.getenv("SMTP_FROM", "noreply@energypulse.local")
    
    # SMS configuration
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
    
    def __init__(self):
        """Initialize notification service."""
        self.db = get_db()
        self._init_email_backend()
        self._init_sms_backend()
    
    def _init_email_backend(self):
        """Initialize email backend (SendGrid or SMTP)."""
        if self.SENDGRID_API_KEY:
            try:
                from sendgrid import SendGridAPIClient
                self.email_backend = "sendgrid"
                self.sg_client = SendGridAPIClient(self.SENDGRID_API_KEY)
            except ImportError:
                print("[WARN] SendGrid API key set but sendgrid package not installed.")
                self.email_backend = None
        elif self.SMTP_HOST:
            self.email_backend = "smtp"
        else:
            self.email_backend = None
    
    def _init_sms_backend(self):
        """Initialize SMS backend (Twilio)."""
        if self.TWILIO_ACCOUNT_SID and self.TWILIO_AUTH_TOKEN:
            try:
                from twilio.rest import Client
                self.sms_backend = "twilio"
                self.twilio_client = Client(self.TWILIO_ACCOUNT_SID, self.TWILIO_AUTH_TOKEN)
            except ImportError:
                print("[WARN] Twilio credentials set but twilio package not installed.")
                self.sms_backend = None
        else:
            self.sms_backend = None
    
    # ────────────── BILL ALERT NOTIFICATIONS ──────────────────────────────
    
    def send_bill_alert(self, household_id: str, primary_email: str,
                       month: str, predicted_cost: float, threshold: Optional[float] = None,
                       language: str = "en") -> List[str]:
        """
        Send bill alert to primary user and eligible family members.
        
        Parameters
        ----------
        household_id : str
            Household identifier
        primary_email : str
            Primary user's email
        month : str
            Month (e.g., "2024-09")
        predicted_cost : float
            Predicted monthly cost in rupees (REAL COMPUTED DATA)
        threshold : float, optional
            Threshold that triggered the alert (if any)
        language : str
            Default language for notification
        
        Returns
        -------
        List[str]
            List of email addresses successfully notified
        """
        sent_to = []
        
        # Get primary user
        primary_user = self.db.get_user(primary_email)
        if not primary_user:
            return []
        
        # Build notification content
        subject_key = "notif_bill_subject"
        body_key = "notif_bill_body"
        
        trigger_data = {
            "type": "bill_alert",
            "month": month,
            "predicted_cost": float(predicted_cost),
            "threshold": float(threshold) if threshold else None,
            "triggered_at": datetime.now().isoformat(),
        }
        
        # Send to primary user
        user_lang = primary_user.get("preferred_language", language)
        recipient_list = [(primary_email, primary_user.get("name", "User"), user_lang)]
        
        # Get and filter family members (only those who opted in)
        family_members = self.db.get_family_members(household_id)
        for member in family_members:
            if member.get("notification_bill_alerts"):
                recipient_list.append((
                    member["email"],
                    member["name"],
                    member.get("preferred_language", language)
                ))
        
        # Send to each recipient in their language
        for recipient_email, recipient_name, recipient_lang in recipient_list:
            success, error_msg = self._send_bill_alert_to_recipient(
                household_id, recipient_email, recipient_name,
                month, predicted_cost, threshold,
                trigger_data, recipient_lang
            )
            
            # Log notification
            if success:
                sent_to.append(recipient_email)
                status = "sent"
                error = None
            else:
                status = "failed"
                error = error_msg
            
            # Use i18n.T for message template (fallback to English keys)
            try:
                subject = T(subject_key, month=month) if "T" in dir() else f"Bill Alert - {month}"
            except:
                subject = f"Bill Alert - {month}"
            
            body = f"Predicted cost: Rs. {predicted_cost:.2f}"
            
            self.db.log_notification(
                household_id=household_id,
                recipient_email=recipient_email,
                recipient_name=recipient_name,
                notification_type="bill_alert",
                triggered_by="bill_threshold",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=recipient_lang,
                status=status,
                error_message=error
            )
        
        return sent_to
    
    def _send_bill_alert_to_recipient(self, household_id: str, recipient_email: str,
                                     recipient_name: str, month: str,
                                     predicted_cost: float, threshold: Optional[float],
                                     trigger_data: Dict[str, Any],
                                     language: str = "en") -> Tuple[bool, Optional[str]]:
        """Send bill alert to a single recipient."""
        try:
            # Compose message
            subject = f"📊 Bill Alert - {month}"
            
            # Build body with REAL data only
            body_lines = [
                f"Hello {recipient_name},",
                f"",
                f"Your predicted electricity cost for {month} is Rs. {predicted_cost:.2f}",
            ]
            
            if threshold:
                body_lines.append(f"This exceeds your set threshold of Rs. {threshold:.2f}")
            
            body_lines.extend([
                f"",
                f"Review your usage patterns and consider our optimization tips.",
                f"",
                f"—",
                f"EnergyPulse Dashboard"
            ])
            
            body = "\n".join(body_lines)
            
            # Send via available backend
            if not self._send_email(recipient_email, subject, body):
                return (False, "Email send failed")
            
            return (True, None)
        
        except Exception as e:
            return (False, str(e))
    
    # ────────────── OPTIMIZATION TIP NOTIFICATIONS ──────────────────────────────
    
    def send_optimization_tips(self, household_id: str, primary_email: str,
                              tip_data: Dict[str, Any], language: str = "en") -> List[str]:
        """
        Send optimization tips based on REAL detected anomalies.
        
        Parameters
        ----------
        household_id : str
            Household identifier
        primary_email : str
            Primary user's email
        tip_data : dict
            Tip data with keys:
            - appliance_name: str (REAL appliance from data)
            - current_usage: float (REAL measurement)
            - avg_usage: float (REAL average)
            - increase_pct: float (REAL percentage)
            - suggested_action: str (Grounded in real data)
        language : str
            Default language
        
        Returns
        -------
        List[str]
            List of emails successfully notified
        """
        sent_to = []
        
        # Validate that tip_data comes from real measurements
        required_fields = ['appliance_name', 'current_usage', 'avg_usage', 'increase_pct', 'suggested_action']
        if not all(field in tip_data for field in required_fields):
            return []
        
        # Get primary user
        primary_user = self.db.get_user(primary_email)
        if not primary_user:
            return []
        
        trigger_data = {
            "type": "optimization_tip",
            "appliance": tip_data["appliance_name"],
            "current_usage_kwh": float(tip_data["current_usage"]),
            "avg_usage_kwh": float(tip_data["avg_usage"]),
            "increase_pct": float(tip_data["increase_pct"]),
            "triggered_at": datetime.now().isoformat(),
        }
        
        # Send to primary user
        user_lang = primary_user.get("preferred_language", language)
        recipient_list = [(primary_email, primary_user.get("name", "User"), user_lang)]
        
        # Get and filter family members (only those who opted in)
        family_members = self.db.get_family_members(household_id)
        for member in family_members:
            if member.get("notification_optimization_tips"):
                recipient_list.append((
                    member["email"],
                    member["name"],
                    member.get("preferred_language", language)
                ))
        
        # Send to each recipient
        for recipient_email, recipient_name, recipient_lang in recipient_list:
            success, error_msg = self._send_optimization_tip_to_recipient(
                household_id, recipient_email, recipient_name,
                tip_data, trigger_data, recipient_lang
            )
            
            if success:
                sent_to.append(recipient_email)
                status = "sent"
                error = None
            else:
                status = "failed"
                error = error_msg
            
            subject = f"💡 Optimization Tip - {tip_data['appliance_name']}"
            body = tip_data["suggested_action"]
            
            self.db.log_notification(
                household_id=household_id,
                recipient_email=recipient_email,
                recipient_name=recipient_name,
                notification_type="optimization_tip",
                triggered_by="usage_anomaly",
                trigger_data=trigger_data,
                subject=subject,
                body_text=body,
                language=recipient_lang,
                status=status,
                error_message=error
            )
        
        return sent_to
    
    def _send_optimization_tip_to_recipient(self, household_id: str,
                                           recipient_email: str, recipient_name: str,
                                           tip_data: Dict[str, Any], trigger_data: Dict[str, Any],
                                           language: str = "en") -> Tuple[bool, Optional[str]]:
        """Send optimization tip to a single recipient."""
        try:
            subject = f"💡 Optimization Tip - {tip_data['appliance_name']}"
            
            # Build body using REAL data
            body_lines = [
                f"Hello {recipient_name},",
                f"",
                f"We detected unusual usage of {tip_data['appliance_name']}:",
                f"",
                f"• Current usage: {tip_data['current_usage']:.2f} kWh",
                f"• Your average: {tip_data['avg_usage']:.2f} kWh",
                f"• Increase: {tip_data['increase_pct']:.1f}%",
                f"",
                f"Recommendation:",
                f"{tip_data['suggested_action']}",
                f"",
                f"—",
                f"EnergyPulse Dashboard"
            ]
            
            body = "\n".join(body_lines)
            
            if not self._send_email(recipient_email, subject, body):
                return (False, "Email send failed")
            
            return (True, None)
        
        except Exception as e:
            return (False, str(e))
    
    # ────────────── EMAIL DELIVERY ──────────────────────────────────────────────────
    
    def _send_email(self, to_email: str, subject: str, body_text: str) -> bool:
        """
        Send email via configured backend (SendGrid or SMTP).
        Returns True on success, False on failure.
        Never raises exceptions — errors are logged to database.
        """
        try:
            if self.email_backend == "sendgrid":
                return self._send_via_sendgrid(to_email, subject, body_text)
            elif self.email_backend == "smtp":
                return self._send_via_smtp(to_email, subject, body_text)
            else:
                # No email backend configured
                return False
        except Exception as e:
            print(f"[ERROR] Email send failed to {to_email}: {e}")
            return False
    
    def _send_via_sendgrid(self, to_email: str, subject: str, body_text: str) -> bool:
        """Send via SendGrid API."""
        try:
            from sendgrid.helpers.mail import Mail, Email, To, Content
            
            message = Mail(
                from_email=Email(self.SMTP_FROM),
                to_emails=To(to_email),
                subject=subject,
                plain_text_content=Content("text/plain", body_text)
            )
            
            response = self.sg_client.send(message)
            return 200 <= response.status_code < 300
        
        except Exception as e:
            print(f"[ERROR] SendGrid send failed: {e}")
            return False
    
    def _send_via_smtp(self, to_email: str, subject: str, body_text: str) -> bool:
        """Send via SMTP (requires SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD)."""
        try:
            if not all([self.SMTP_HOST, self.SMTP_USERNAME, self.SMTP_PASSWORD]):
                return False
            
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.SMTP_FROM
            msg["To"] = to_email
            
            # Add body
            part = MIMEText(body_text, "plain")
            msg.attach(part)
            
            # Send
            with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT) as server:
                server.starttls()
                server.login(self.SMTP_USERNAME, self.SMTP_PASSWORD)
                server.sendmail(self.SMTP_FROM, [to_email], msg.as_string())
            
            return True
        
        except Exception as e:
            print(f"[ERROR] SMTP send failed: {e}")
            return False
    
    # ────────────── SMS DELIVERY (OPTIONAL) ──────────────────────────────────────────────────
    
    def send_sms(self, to_phone: str, message: str) -> bool:
        """
        Send SMS via Twilio (optional feature).
        Returns True on success.
        """
        if not self.sms_backend == "twilio":
            return False
        
        try:
            self.twilio_client.messages.create(
                body=message,
                from_=self.TWILIO_FROM_NUMBER,
                to=to_phone
            )
            return True
        except Exception as e:
            print(f"[ERROR] SMS send failed to {to_phone}: {e}")
            return False
    
    # ────────────── TEST NOTIFICATION ──────────────────────────────────────────────────
    
    def send_test_notification(self, household_id: str, recipient_email: str,
                              language: str = "en") -> Tuple[bool, str]:
        """
        Send a test notification to verify email delivery is working.
        Used in settings/configuration.
        
        Returns (success: bool, message: str)
        """
        try:
            subject = "🧪 EnergyPulse Test Notification"
            body_lines = [
                "Hello,",
                "",
                "This is a test notification from EnergyPulse.",
                "If you receive this, your email configuration is working correctly!",
                "",
                f"Test sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "—",
                "EnergyPulse Dashboard"
            ]
            body = "\n".join(body_lines)
            
            success = self._send_email(recipient_email, subject, body)
            
            if success:
                # Log the test notification
                self.db.log_notification(
                    household_id=household_id,
                    recipient_email=recipient_email,
                    recipient_name="Test Recipient",
                    notification_type="test",
                    triggered_by="manual_test",
                    trigger_data={},
                    subject=subject,
                    body_text=body,
                    language=language,
                    status="sent",
                    error_message=None
                )
                return (True, "Test notification sent successfully!")
            else:
                return (False, "Failed to send test notification. Check email configuration.")
        
        except Exception as e:
            return (False, f"Error: {str(e)}")


# Singleton instance
_notification_service = None

def get_notification_service() -> NotificationService:
    """Get or create the notification service singleton."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service
