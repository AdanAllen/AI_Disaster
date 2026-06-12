import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app as stayready
import submission_repository
from security_utils import reset_rate_limits


BASE_DIR = Path(__file__).resolve().parents[1]


def csrf_token(html, field_name):
    match = re.search(
        rf'name="{re.escape(field_name)}"[^>]*value="([^"]+)"',
        html,
    )
    if not match:
        match = re.search(
            rf'value="([^"]+)"[^>]*name="{re.escape(field_name)}"',
            html,
        )
    if not match:
        raise AssertionError(f"CSRF token field {field_name} not found")
    return match.group(1)


class FeedbackRouteTests(unittest.TestCase):
    def setUp(self):
        stayready.app.config.update(TESTING=True, WTF_CSRF_ENABLED=True)
        self.client = stayready.app.test_client()
        reset_rate_limits()

    def feedback_token(self, query=""):
        response = self.client.get(f"/feedback{query}")
        return csrf_token(response.get_data(as_text=True), "feedback-csrf_token")

    def organization_token(self):
        response = self.client.get("/feedback")
        return csrf_token(response.get_data(as_text=True), "organization-csrf_token")

    def update_token(self):
        response = self.client.get("/")
        return csrf_token(response.get_data(as_text=True), "updates-csrf_token")

    def valid_feedback(self, token, **overrides):
        data = {
            "feedback-csrf_token": token,
            "feedback-form_kind": "feedback",
            "feedback-page_context": "risk_summary",
            "feedback-category": "general_feedback",
            "feedback-name": "Test Resident",
            "feedback-email": "",
            "feedback-message": "The summary language was useful and clear.",
            "feedback-website": "",
        }
        data.update(overrides)
        return data

    def test_feedback_page_and_contextual_category_render(self):
        response = self.client.get(
            "/feedback?category=incorrect_source&page_context=map"
        )
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Do not use this form for emergencies", html)
        self.assertIn("For immediate danger, call 911", html)
        self.assertIn('value="map"', html)
        self.assertIn(
            '<option selected value="incorrect_source">',
            html,
        )

    def test_contact_redirects_and_terms_render(self):
        response = self.client.get("/contact")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/feedback"))

        terms = self.client.get("/terms").get_data(as_text=True)
        self.assertIn("Not an emergency service", terms)
        self.assertIn("call 911", terms)

    @patch("app.save_feedback_submission", return_value=True)
    def test_feedback_submission_is_private_and_normalized(self, save):
        token = self.feedback_token()
        response = self.client.post(
            "/feedback",
            data=self.valid_feedback(
                token,
                **{
                    "feedback-email": " PERSON@Example.COM ",
                    "feedback-category": "incorrect_source",
                },
            ),
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("status=success", response.headers["Location"])
        payload = save.call_args.args[0]
        self.assertEqual(payload["email"], "person@example.com")
        self.assertEqual(payload["admin_tag"], "bug_source_report")
        self.assertEqual(payload["page_context"], "risk_summary")
        self.assertNotIn("ip", payload)
        self.assertNotIn("user_agent", payload)

    @patch("app.save_feedback_submission")
    def test_honeypot_returns_success_without_storage(self, save):
        token = self.feedback_token()
        response = self.client.post(
            "/feedback",
            data=self.valid_feedback(
                token,
                **{"feedback-website": "https://spam.example"},
            ),
        )
        self.assertIn("status=success", response.headers["Location"])
        save.assert_not_called()

    @patch("app.save_feedback_submission", return_value=True)
    def test_question_requires_email(self, save):
        token = self.feedback_token()
        response = self.client.post(
            "/feedback",
            data=self.valid_feedback(
                token,
                **{"feedback-category": "other_question"},
            ),
        )
        self.assertIn("status=invalid", response.headers["Location"])
        save.assert_not_called()

    @patch("app.save_feedback_submission", return_value=True)
    def test_missing_csrf_is_rejected(self, save):
        response = self.client.post(
            "/feedback",
            data=self.valid_feedback(""),
        )
        self.assertIn("status=invalid", response.headers["Location"])
        save.assert_not_called()

    @patch("app.save_feedback_submission", return_value=True)
    def test_unknown_category_control_characters_and_external_context_are_rejected(self, save):
        token = self.feedback_token()
        response = self.client.post(
            "/feedback",
            data=self.valid_feedback(
                token,
                **{
                    "feedback-category": "emergency_help",
                    "feedback-page_context": "https://evil.example",
                    "feedback-message": "Unsafe control character:\x01",
                },
            ),
        )
        self.assertIn("status=invalid", response.headers["Location"])
        self.assertIn("page_context=feedback", response.headers["Location"])
        save.assert_not_called()

    @patch("app.save_feedback_submission", return_value=False)
    def test_provider_failure_never_claims_success(self, _save):
        token = self.feedback_token()
        response = self.client.post(
            "/feedback",
            data=self.valid_feedback(token),
            follow_redirects=True,
        )
        html = response.get_data(as_text=True)
        self.assertIn("temporarily unavailable", html)
        self.assertIn("was not saved", html)
        self.assertNotIn("Your message was received for review", html)

    @patch("app.save_feedback_submission", return_value=True)
    def test_organization_submission_uses_lead_tag(self, save):
        token = self.organization_token()
        response = self.client.post(
            "/feedback",
            data={
                "organization-csrf_token": token,
                "organization-form_kind": "organization",
                "organization-page_context": "home",
                "organization-name": "Alex Rivera",
                "organization-organization": "Alameda Community Center",
                "organization-role": "Program coordinator",
                "organization-email": "alex@example.org",
                "organization-interest_type": "pilot",
                "organization-message": "We want to explore a preparedness workshop pilot.",
                "organization-website": "",
            },
        )
        self.assertIn("status=organization_success", response.headers["Location"])
        payload = save.call_args.args[0]
        self.assertEqual(payload["submission_category"], "community_interest")
        self.assertEqual(payload["admin_tag"], "organization_lead")
        self.assertEqual(payload["interest_type"], "pilot")

    @patch("app.save_email_interest", return_value=True)
    def test_update_signup_records_explicit_consent(self, save):
        token = self.update_token()
        response = self.client.post(
            "/updates/subscribe",
            data={
                "updates-csrf_token": token,
                "updates-form_kind": "updates",
                "updates-email": " PERSON@Example.COM ",
                "updates-location": "Oakland",
                "updates-user_type": "resident",
                "updates-consent": "y",
                "updates-website": "",
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("updates=success", response.headers["Location"])
        payload = save.call_args.args[0]
        self.assertEqual(payload["email"], "person@example.com")
        self.assertEqual(payload["subscription_status"], "subscribed")
        self.assertEqual(payload["consent_version"], stayready.UPDATE_CONSENT_VERSION)

    @patch("app.save_email_interest", return_value=True)
    def test_update_signup_rejects_malformed_email(self, save):
        token = self.update_token()
        response = self.client.post(
            "/updates/subscribe",
            data={
                "updates-csrf_token": token,
                "updates-form_kind": "updates",
                "updates-email": "not-an-email",
                "updates-location": "",
                "updates-user_type": "",
                "updates-consent": "y",
                "updates-website": "",
            },
        )
        self.assertIn("updates=invalid", response.headers["Location"])
        save.assert_not_called()

    @patch("app.save_email_interest", return_value=True)
    def test_update_signup_rejects_street_address_location(self, save):
        token = self.update_token()
        response = self.client.post(
            "/updates/subscribe",
            data={
                "updates-csrf_token": token,
                "updates-form_kind": "updates",
                "updates-email": "person@example.com",
                "updates-location": "123 Main Street",
                "updates-user_type": "resident",
                "updates-consent": "y",
                "updates-website": "",
            },
        )
        self.assertIn("updates=invalid", response.headers["Location"])
        save.assert_not_called()

    @patch("app.save_feedback_submission", return_value=True)
    def test_feedback_rate_limit_uses_safe_page_state(self, _save):
        token = self.feedback_token()
        for _ in range(5):
            response = self.client.post(
                "/feedback",
                data=self.valid_feedback(token),
            )
            self.assertEqual(response.status_code, 302)
        limited = self.client.post(
            "/feedback",
            data=self.valid_feedback(token),
        )
        self.assertEqual(limited.status_code, 302)
        self.assertIn("status=rate_limited", limited.headers["Location"])
        self.assertIn("Retry-After", limited.headers)

    def test_contextual_ctas_and_privacy_copy(self):
        home = self.client.get("/").get_data(as_text=True)
        sources = self.client.get("/sources").get_data(as_text=True)
        privacy = self.client.get("/privacy").get_data(as_text=True)
        self.assertIn("Get StayReady updates", home)
        self.assertIn("Interested in using StayReady with your community", home)
        self.assertIn("Report a source issue", sources)
        self.assertIn("private Supabase database", privacy)
        self.assertIn("does not store your IP address", privacy)


class SubmissionRepositoryTests(unittest.TestCase):
    def setUp(self):
        submission_repository.get_submission_client.cache_clear()

    def tearDown(self):
        submission_repository.get_submission_client.cache_clear()

    @patch("submission_repository.get_submission_client")
    def test_email_interest_uses_idempotent_upsert(self, get_client):
        client = MagicMock()
        client.table.return_value.upsert.return_value.execute.return_value.data = [{}]
        get_client.return_value = client
        payload = {"email": "person@example.com"}
        self.assertTrue(submission_repository.save_email_interest(payload))
        client.table.assert_called_once_with("email_interests")
        client.table.return_value.upsert.assert_called_once_with(
            payload,
            on_conflict="email",
        )

    @patch("submission_repository.get_submission_client")
    def test_feedback_insert_failure_is_false(self, get_client):
        client = MagicMock()
        client.table.return_value.insert.side_effect = RuntimeError("offline")
        get_client.return_value = client
        self.assertFalse(
            submission_repository.save_feedback_submission({"message": "test"})
        )

    def test_migration_keeps_public_roles_out(self):
        sql = (
            BASE_DIR
            / "supabase"
            / "migrations"
            / "003_create_feedback_and_interest_tables.sql"
        ).read_text(encoding="utf-8").lower()
        self.assertIn("enable row level security", sql)
        self.assertIn("revoke all on table public.email_interests from anon, authenticated", sql)
        self.assertIn("revoke all on table public.feedback_submissions from anon, authenticated", sql)
        self.assertNotIn("create policy", sql)
        self.assertIn("grant select, insert, update, delete", sql)

    @patch.dict(
        "os.environ",
        {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SECRET_KEY": "test-secret",
        },
        clear=False,
    )
    def test_submission_config_status_reports_only_booleans(self):
        status = submission_repository.get_submission_config_status()
        self.assertEqual(status, {
            "configured": True,
            "project_url_configured": True,
            "secret_key_configured": True,
        })
        self.assertNotIn("example.supabase.co", repr(status))
        self.assertNotIn("test-secret", repr(status))


if __name__ == "__main__":
    unittest.main()
