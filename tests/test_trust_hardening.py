import json
import unittest
from pathlib import Path

from app import app


BASE_DIR = Path(__file__).resolve().parents[1]


class PublicAdviceSurfaceTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_homepage_examples_are_action_library_records(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-action-id="portable_go_bag"', html)
        self.assertIn('data-action-id="protect_critical_documents"', html)
        self.assertIn("Source note:", html)
        self.assertNotIn(
            "Build a go-bag, alert plan, and utility safety plan.",
            html,
        )

    def test_resources_page_is_a_neutral_official_directory(self):
        response = self.client.get("/resources")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "This directory does not create or personalize preparedness recommendations.",
            html,
        )
        self.assertNotIn("Start with one bag", html)
        self.assertNotIn("Match supplies to your household", html)
        self.assertNotIn("every household should keep ready", html)

    def test_legacy_hazard_urls_redirect_to_structured_pages(self):
        for path in ("/wildfire", "/flood", "/earthquake"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302)
            self.assertIn("/hazards/", response.headers["Location"])


class StaticTrustContractTests(unittest.TestCase):
    def test_no_freeform_ai_advice_generator_remains(self):
        app_source = (BASE_DIR / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("OpenAI", app_source)
        self.assertNotIn("chat.completions", app_source)
        self.assertNotIn("def hazard_page(", app_source)
        self.assertNotIn("initial_response", app_source)
        self.assertNotIn(
            "openai",
            (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").lower(),
        )

        template_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (BASE_DIR / "templates").glob("*.html")
        )
        self.assertNotIn("initial_response", template_source)
        self.assertNotIn("generateAIResponse", template_source)

    def test_chunk_recommendations_are_not_copied_to_compatibility_fields(self):
        engine_source = (BASE_DIR / "hazard_engine.py").read_text(encoding="utf-8")
        resident_source = (BASE_DIR / "resident_guidance_engine.py").read_text(encoding="utf-8")
        self.assertNotIn("action_step_text", engine_source)
        self.assertNotIn("item.recommended_action", engine_source)
        self.assertNotIn("recommended_action\") or", resident_source)

    def test_current_geospatial_datasets_remain_provisional(self):
        datasets = json.loads(
            (BASE_DIR / "data" / "geospatial" / "datasets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {item["dataset_id"] for item in datasets},
            {"fema_nfhl_local", "calfire_fhsz_local", "usgs_cgs_faults_local"},
        )
        self.assertTrue(all(item["status"] == "provisional" for item in datasets))
        self.assertTrue(all(not item.get("human_reviewer") for item in datasets))

    def test_legacy_generator_is_explicitly_retired(self):
        source = (BASE_DIR / "generate_risk_csv.py").read_text(encoding="utf-8")
        self.assertIn("Retired legacy ZIP-score generator", source)
        self.assertNotIn("chatbot_prompt", source)
        self.assertNotIn("to_csv", source)


if __name__ == "__main__":
    unittest.main()
