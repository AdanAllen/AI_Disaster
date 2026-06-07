import unittest
from pathlib import Path
from unittest.mock import patch

import app as stayready


class SessionPrivacyTests(unittest.TestCase):
    def setUp(self):
        stayready.app.config.update(TESTING=True)
        self.client = stayready.app.test_client()

    def submit_address(self):
        return self.client.post(
            "/form",
            data={
                "address": "1020 Sierra St, Berkeley, CA",
                "household": "2",
                "preparedness": "basic",
                "special_needs": "Private medical and access details",
                "household_tags": ["medical", "pets", "no_car"],
            },
            follow_redirects=False,
        )

    @patch(
        "app.geocode_address",
        return_value=(
            37.89,
            -122.27,
            "94707",
            "1020 Sierra Street, Berkeley, Alameda County, California, 94707, United States",
        ),
    )
    def test_sensitive_values_are_not_stored_in_flask_cookie_session(self, _geocode):
        response = self.submit_address()
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers["Location"].endswith("/risk_summary"))

        with self.client.session_transaction() as saved:
            cookie_state = dict(saved)

        self.assertEqual(set(cookie_state), {stayready.RESULT_TOKEN_SESSION_KEY})
        serialized = repr(cookie_state).lower()
        for private_value in (
            "1020 sierra",
            "37.89",
            "-122.27",
            "private medical",
            "medical",
            "pets",
            "no_car",
        ):
            self.assertNotIn(private_value, serialized)

        resident_state = stayready.RESIDENT_STATE_CACHE.get(
            cookie_state[stayready.RESULT_TOKEN_SESSION_KEY]
        )
        self.assertEqual(resident_state["zip_code"], "94707")
        self.assertEqual(resident_state["location_mode"], "address")
        self.assertNotIn("special_needs", resident_state)

    @patch(
        "app.geocode_address",
        return_value=(
            37.89,
            -122.27,
            "94707",
            "1020 Sierra Street, Berkeley, Alameda County, California, 94707, United States",
        ),
    )
    @patch("app.get_all_hazards", return_value=[])
    def test_summary_and_map_work_with_opaque_result_token(self, _hazards, _geocode):
        self.submit_address()
        summary = self.client.get("/risk_summary")
        map_page = self.client.get("/map")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(map_page.status_code, 200)
        self.assertIn("1020 Sierra Street", summary.get_data(as_text=True))
        self.assertIn("1020 Sierra Street", map_page.get_data(as_text=True))

    def test_home_removes_sensitive_free_text_and_marks_household_fields_optional(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn('name="special_needs"', html)
        self.assertIn("Optional household planning details", html)
        self.assertIn("Do not enter medical diagnoses", html)

    def test_privacy_page_discloses_geocoding_and_temporary_handling(self):
        response = self.client.get("/privacy")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        for expected in (
            "not an official government site",
            "not an emergency alert system",
            "Nominatim",
            "does not save raw addresses",
            "random result token",
            "Call 911",
            "AC Alert",
        ):
            self.assertIn(expected, html)

    def test_health_endpoints_do_not_expose_resident_values(self):
        private_value = "1020 Sierra Street, Berkeley"
        combined = (
            self.client.get("/api/health").get_data(as_text=True)
            + self.client.get("/api/supabase-health").get_data(as_text=True)
        )
        self.assertNotIn(private_value, combined)
        self.assertNotIn(stayready.RESULT_TOKEN_SESSION_KEY, combined)

    def test_source_does_not_log_raw_coordinates_or_form_bodies(self):
        source = Path(stayready.__file__).read_text(encoding="utf-8")
        self.assertNotIn("lookup failed for coordinates (%s, %s)", source)
        self.assertNotIn("outside coverage bounds: %s, %s", source)
        self.assertNotIn('session["form_data"]', source)
        self.assertNotIn("request.form.to_dict()", source)

    def test_map_layer_requests_do_not_put_exact_coordinates_in_urls(self):
        source = (Path(stayready.BASE_DIR) / "templates" / "map.html").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('lat: "{{ user_lat }}"', source)
        self.assertNotIn('lon: "{{ user_lon }}"', source)
        self.assertIn("fetch(`/api/cgs-map-layer/${layerKey}`)", source)


if __name__ == "__main__":
    unittest.main()
