"""
Regression tests for notification backend detection, status honesty and log types.
"""
import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import notifications
from notifications import NotificationService


def build(env):
    """Construct a service with a controlled environment, no sockets opened."""
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        return NotificationService()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestSmtpPortParsing(unittest.TestCase):
    def test_bad_port_does_not_raise(self):
        """int() on a bad port used to crash every tab, not just notifications."""
        svc = build({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
            "SMTP_PORT": "smtp.gmail.com",
        })
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertIsNotNone(svc.SMTP_PORT_ERROR)
        self.assertIn("587", svc.SMTP_PORT_ERROR)

    def test_out_of_range_port_falls_back(self):
        svc = build({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
            "SMTP_PORT": "99999",
        })
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertIsNotNone(svc.SMTP_PORT_ERROR)

    def test_valid_port_is_kept(self):
        svc = build({
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
            "SMTP_PORT": "465",
        })
        self.assertEqual(svc.SMTP_PORT, 465)
        self.assertIsNone(svc.SMTP_PORT_ERROR)

    def test_blank_port_uses_default_without_warning(self):
        svc = build({"SMTP_PORT": ""})
        self.assertEqual(svc.SMTP_PORT, 587)
        self.assertIsNone(svc.SMTP_PORT_ERROR)


class TestStatusHonesty(unittest.TestCase):
    def test_no_credentials_is_not_configured(self):
        svc = build({
            "SENDGRID_API_KEY": None, "SMTP_HOST": None,
            "SMTP_USERNAME": None, "SMTP_PASSWORD": None,
        })
        self.assertEqual(svc.email_status(), "not_configured")

    def test_credentials_alone_are_not_reported_as_connected(self):
        """Reporting "Connected" from env vars alone was a claim we never tested."""
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        self.assertEqual(svc.email_status(), "configured")
        self.assertFalse(svc.email_verified)

    def test_successful_send_marks_verified(self):
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        svc._send_email = lambda *a, **k: True
        ok, err = svc._deliver("to@example.com", "s", "b", None)
        self.assertTrue(ok)
        self.assertEqual(svc.email_status(), "verified")

    def test_failed_send_is_reported(self):
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "user@gmail.com",
            "SMTP_PASSWORD": "real-password",
        })
        svc._send_email = lambda *a, **k: False
        ok, err = svc._deliver("to@example.com", "s", "b", None)
        self.assertFalse(ok)
        self.assertEqual(svc.email_status(), "failed")
        self.assertIsNotNone(svc.last_email_error)

    def test_placeholder_credentials_are_not_configured(self):
        svc = build({
            "SENDGRID_API_KEY": None,
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_USERNAME": "your_email@gmail.com",
            "SMTP_PASSWORD": "your_app_password",
        })
        self.assertEqual(svc.email_status(), "not_configured")


class TestLogTypes(unittest.TestCase):
    """Every stored row's type must match the type in its own payload."""

    def test_weekly_summary_is_not_logged_as_bill_alert(self):
        import inspect
        src = inspect.getsource(NotificationService.send_weekly_summary)
        self.assertIn('notification_type="weekly_summary"', src)
        self.assertNotIn('notification_type="bill_alert"', src)

    def test_payload_and_row_types_agree(self):
        import inspect
        for method, expected in [
            ("send_bill_alert", "bill_alert"),
            ("send_weekly_summary", "weekly_summary"),
            ("send_optimization_tips", "optimization_tip"),
            ("send_test_notification", "test"),
        ]:
            src = inspect.getsource(getattr(NotificationService, method))
            self.assertIn(f'notification_type="{expected}"', src, method)
            self.assertIn(f'"type": "{expected}"', src, method)

    def test_every_logged_type_has_a_localized_label(self):
        from i18n import TRANSLATIONS
        import chatbot
        mapping = chatbot.NOTIFICATION_TYPE_KEY
        for type_name in ("bill_alert", "weekly_summary", "optimization_tip", "test"):
            key = mapping[type_name]
            for lang, catalog in TRANSLATIONS.items():
                self.assertIn(key, catalog, f"{type_name} missing in {lang}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

