import unittest

from app import app
from testing_utils import set_test_resident_state


class UIFeedbackTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_home_renders_honest_structured_summary_loading_state(self):
        html = self.client.get("/").get_data(as_text=True)

        self.assertIn('id="sr-summary-loading"', html)
        self.assertIn("Finding address…", html)
        self.assertIn("Checking mapped hazard zones…", html)
        self.assertIn("Checking FEMA flood data…", html)
        self.assertIn("Checking CAL FIRE wildfire data…", html)
        self.assertIn("Loading local plan facts…", html)
        self.assertIn(
            "show findings only after the available source checks return",
            html,
        )

    def test_summary_has_scan_first_sections_and_plain_language_help(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "location_mode": "zip",
        })
        html = self.client.get("/risk_summary").get_data(as_text=True)

        self.assertIn("Key findings for this location", html)
        self.assertIn("What this means", html)
        self.assertIn("What to do next", html)
        self.assertIn("Help with map and evidence terms", html)
        self.assertIn("FEMA Zone X", html)
        self.assertIn("Liquefaction zone", html)
        self.assertIn("Mapped match found", html)
        self.assertIn("No mapped match found", html)
        self.assertIn("Not checked", html)

    def test_empty_summary_does_not_imply_checks_have_run(self):
        html = self.client.get("/risk_summary").get_data(as_text=True)
        self.assertIn(
            "No source-backed findings are shown until StayReady has a location to check.",
            html,
        )

    def test_map_defers_flood_and_reuses_loaded_layers(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "location_mode": "zip",
        })
        html = self.client.get("/map").get_data(as_text=True)
        init_map = html.split("function initMap()", 1)[1].split(
            "// Load ZIP code boundary",
            1,
        )[0]

        self.assertNotIn("loadFloodZones()", init_map)
        self.assertIn("layers[layerType].addTo(map);", html)
        self.assertNotIn("layers[layerType] = null;", html)
        self.assertIn("if (state !== 'hidden')", html)
        self.assertIn("no data shown for this address or map area", html)
        self.assertIn("Other map layers are still available", html)
        self.assertIn("layerRequestState[layerType] === 'loaded'", html)


if __name__ == "__main__":
    unittest.main()
