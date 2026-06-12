import csv
import json
import unittest
from pathlib import Path

from app import app
from testing_utils import set_test_resident_state


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

    def test_legacy_zip_api_is_scoped_as_fallback_ranking(self):
        response = self.client.get("/api/risk-assessment/94619")
        payload = response.get_json()
        text = response.get_data(as_text=True).lower()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["result_type"], "legacy_zip_fallback")
        self.assertIn("do not determine address exposure or safety", payload["limitation"])
        self.assertIn("ranking_score", payload["risks"]["flood"])
        self.assertIn("priority_band", payload["risks"]["flood"])
        self.assertNotIn("overall_risk", payload)
        for phrase in (
            "low risk",
            "minimal risk",
            "very low wildfire risk",
            "risk is low",
            "not exposed",
        ):
            self.assertNotIn(phrase, text)

    def test_public_hazard_outputs_do_not_use_unsafe_reassurance(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "location_mode": "zip",
        })
        for path in ("/api/hazards", "/api/top-risks", "/hazards", "/map"):
            response = self.client.get(path)
            text = response.get_data(as_text=True).lower()
            self.assertEqual(response.status_code, 200)
            for phrase in (
                "low risk",
                "minimal risk",
                "very low wildfire risk",
                "risk is low",
                "not exposed",
            ):
                self.assertNotIn(phrase, text, path)


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
            {
                "fema_nfhl_local",
                "calfire_fhsz_local",
                "usgs_cgs_faults_local",
                "cgs_alquist_priolo_remote",
                "cgs_liquefaction_remote",
                "cgs_earthquake_landslide_remote",
                "cgs_tsunami_hazard_area_remote",
            },
        )
        self.assertTrue(all(item["status"] == "provisional" for item in datasets))
        self.assertTrue(all(not item.get("human_reviewer") for item in datasets))

    def test_legacy_generator_is_explicitly_retired(self):
        source = (BASE_DIR / "generate_risk_csv.py").read_text(encoding="utf-8")
        self.assertIn("Retired legacy ZIP-score generator", source)
        self.assertNotIn("chatbot_prompt", source)
        self.assertNotIn("to_csv", source)

    def test_legacy_zip_descriptions_are_evidence_scoped(self):
        with (BASE_DIR / "static" / "zip_risk_scores.csv").open(
            newline="",
            encoding="utf-8",
        ) as source:
            rows = list(csv.DictReader(source))
        self.assertTrue(rows)
        for row in rows:
            descriptions = " ".join([
                row["Earthquake_Risk_Explanation"],
                row["Flood_Risk_Explanation"],
                row["Wildfire_Risk_Explanation"],
            ]).lower()
            self.assertIn("legacy zip fallback ranking signal", descriptions)
            self.assertIn("safety determination", descriptions)
            for phrase in (
                "low risk",
                "minimal risk",
                "very low wildfire risk",
                "risk is low",
                "minimal wildfire threat",
            ):
                self.assertNotIn(phrase, descriptions)

    def test_templates_do_not_label_fallbacks_as_low_risk(self):
        templates = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (BASE_DIR / "templates").glob("*.html")
        )
        self.assertNotIn("Risk Level:", templates)
        self.assertNotIn("}} risk", templates)
        self.assertNotIn("Exposure not determined", templates)
        guide_source = (BASE_DIR / "hazard_guide.py").read_text(encoding="utf-8")
        self.assertIn("Add an address to check this layer", guide_source)


if __name__ == "__main__":
    unittest.main()
