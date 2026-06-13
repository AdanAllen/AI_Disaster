import unittest
from pathlib import Path

from app import app


BASE_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "base.html"


class PostHogAnalyticsTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_public_pages_include_privacy_safe_posthog_config(self):
        html = self.client.get("/").get_data(as_text=True)

        for expected in (
            "phc_qiW2Y8hqapKpVbj3P8wuqB7Cs8vyRMZt9au3HpUqwnwS",
            "api_host: 'https://us.i.posthog.com'",
            "defaults: '2026-05-30'",
            "person_profiles: 'identified_only'",
            "capture_pageview: true",
            "autocapture: false",
            "disable_session_recording: true",
            "capture_pageleave: false",
            "capture_dead_clicks: false",
        ):
            self.assertIn(expected, html)

    def test_custom_event_helper_is_allowlisted_and_sanitized(self):
        source = BASE_TEMPLATE.read_text(encoding="utf-8")

        for event_name in (
            "address_search_started",
            "address_search_completed",
            "risk_summary_loaded",
            "risk_summary_failed",
            "map_opened",
            "map_layer_toggled",
            "source_link_clicked",
            "action_clicked",
            "data_unavailable_shown",
        ):
            self.assertIn(f"'{event_name}'", source)

        for unsafe_key in (
            "address",
            "lat",
            "lng",
            "latitude",
            "longitude",
            "coordinates",
            "household",
            "name",
            "email",
            "phone",
            "notes",
        ):
            self.assertIn(f"'{unsafe_key}'", source)

        self.assertIn("window.trackStayReadyEvent", source)
        self.assertIn("sanitizeProperties(event.properties)", source)
        self.assertIn("event.event !== '$pageview'", source)
        self.assertIn("Private/admin templates must override this block", source)


if __name__ == "__main__":
    unittest.main()
