import re
import unittest
from pathlib import Path

from app import app


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
BASE_TEMPLATE = TEMPLATES_DIR / "base.html"


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
            "capture_pageleave: true",
            "autocapture: true",
            "capture_performance: true",
            "enable_heatmaps: true",
            "disable_session_recording: false",
            "capture_exceptions:",
            "capture_dead_clicks: false",
            "maskAllInputs: true",
            "maskTextSelector: '*'",
            "blockClass: 'ph-no-capture'",
            "request.name.split('?')[0].split('#')[0]",
        ):
            self.assertIn(expected, html)

    def test_safe_helper_allowlists_events_and_properties(self):
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

        for safe_key in (
            "city",
            "jurisdiction",
            "hazard",
            "layer",
            "status",
            "source",
            "page",
            "device_type",
            "result_type",
            "error_type",
        ):
            self.assertIn(f"'{safe_key}'", source)

        for unsafe_key in (
            "address",
            "street",
            "lat",
            "lng",
            "latitude",
            "longitude",
            "coordinates",
            "coords",
            "household",
            "name",
            "email",
            "phone",
            "notes",
            "emergency_notes",
            "user_input",
            "query",
        ):
            self.assertIn(f"'{unsafe_key}'", source)

        self.assertIn("window.trackStayReadyEvent", source)
        self.assertIn("sanitizeCustomProperties(properties)", source)
        self.assertIn("sanitizeCustomProperties(event.properties)", source)
        self.assertIn("sanitizeInternalProperties(event.properties)", source)
        self.assertIn("sanitizeException(event)", source)
        self.assertIn("Private/admin templates must override this block", source)

    def test_sensitive_surfaces_are_blocked_from_replay_and_autocapture(self):
        protected_templates = {
            "_home_reference.html": (
                'id="emergency-form" class="sr-home-search sr-form ph-no-capture"',
                "sr-interest-form ph-no-capture",
            ),
            "_risk_summary_reference.html": ("sr-summary-reference ph-no-capture",),
            "_map_reference.html": ("sr-map-shell ph-no-capture",),
            "_hazard_profile_form.html": ("hazard-profile-form ph-no-capture",),
            "hazards_dashboard.html": ("sr-hazard-location-strip ph-no-capture",),
            "hazard_detail.html": ("sr-hazard-current-location ph-no-capture",),
            "feedback.html": ('class="ph-no-capture"',),
            "sources.html": ("sr-sources-search ph-no-capture",),
        }
        for template_name, expected_values in protected_templates.items():
            source = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
            for expected in expected_values:
                self.assertIn(expected, source, template_name)

    def test_privacy_page_discloses_posthog_features_and_masking(self):
        html = self.client.get("/privacy").get_data(as_text=True)
        for expected in (
            "PostHog analytics",
            "pageviews",
            "pageleaves",
            "performance and web-vitals metrics",
            "heatmaps",
            "session recordings",
            "mask all page text and input fields",
            "address input",
            "household details",
            "Query strings are stripped",
            "not an official emergency service",
        ):
            self.assertIn(expected, html)

    def test_custom_event_calls_do_not_use_sensitive_property_names(self):
        sensitive_keys = {
            "address",
            "street",
            "lat",
            "lng",
            "latitude",
            "longitude",
            "coordinates",
            "coords",
            "household",
            "name",
            "email",
            "phone",
            "notes",
            "emergency_notes",
            "user_input",
            "query",
        }
        for template_path in TEMPLATES_DIR.glob("*.html"):
            source = template_path.read_text(encoding="utf-8")
            for call in re.findall(
                r"trackStayReadyEvent\([^;]+?\);",
                source,
                flags=re.DOTALL,
            ):
                if "function(eventName, properties)" in call:
                    continue
                for key in sensitive_keys:
                    self.assertNotRegex(
                        call,
                        rf"(?<![A-Za-z0-9_$]){re.escape(key)}\s*:",
                        f"{template_path.name} uses unsafe analytics property {key}",
                    )


if __name__ == "__main__":
    unittest.main()
