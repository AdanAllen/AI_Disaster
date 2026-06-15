import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from source_registry import local_plan_groups_payload, source_page_payload


BASE_DIR = Path(__file__).resolve().parents[1]


class SourceRegistryPresentationTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    @patch("source_registry._supabase_source_records_payload", return_value=[])
    def test_sources_page_uses_plain_language_statuses_and_filters(self, _supabase):
        response = self.client.get("/sources")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Data sources and methods", html)
        self.assertIn("Official source; integration under review", html)
        self.assertIn("Reviewed official guidance", html)
        self.assertIn('id="source-search"', html)
        self.assertIn('data-source-filter="mapped-data"', html)
        self.assertIn('data-source-filter="guidance"', html)
        self.assertNotIn("draft_reviewed", html)
        self.assertNotIn("mixed_support", html)

    @patch("source_registry._supabase_source_records_payload", return_value=[])
    def test_sources_page_explains_dates_and_links_to_official_viewers(self, _supabase):
        html = self.client.get("/sources").get_data(as_text=True)

        self.assertIn("it is not the dataset's", html)
        self.assertIn("Source link reviewed", html)
        self.assertIn("June 12, 2026", html)
        self.assertIn("https://msc.fema.gov/nfhl", html)
        self.assertIn(
            "community-wildfire-preparedness-and-mitigation/fire-hazard-severity-zones",
            html,
        )
        self.assertIn("https://www.conservation.ca.gov/cgs/geohazards/eq-zapp", html)
        self.assertIn("Report a source issue", html)

    @patch("source_registry._supabase_source_records_payload", return_value=[])
    def test_source_page_payload_separates_registry_categories(self, _supabase):
        payload = source_page_payload()

        self.assertEqual(payload["counts"]["registered"], 26)
        self.assertEqual(payload["counts"]["mapped"], 8)
        self.assertEqual(payload["counts"]["guidance"], 7)
        self.assertEqual(payload["counts"]["plan_records"], 11)
        self.assertEqual(payload["counts"]["jurisdictions"], 15)
        self.assertEqual(len(payload["sources"]), 15)
        self.assertEqual(len(payload["local_plan_groups"]), 11)

    def test_shared_local_plans_are_grouped_once(self):
        groups = {group["plan_group"]: group for group in local_plan_groups_payload()}

        self.assertEqual(
            groups["Tri-City LHMP"]["jurisdictions"],
            ["Fremont", "Newark", "Union City"],
        )
        self.assertEqual(
            groups["Tri-Valley LHMP"]["jurisdictions"],
            ["Dublin", "Livermore", "Pleasanton"],
        )

    def test_corrected_registry_records_are_stable(self):
        sources = {
            item["source_id"]: item
            for item in json.loads((BASE_DIR / "sources.json").read_text(encoding="utf-8"))
        }

        self.assertEqual(
            sources["calfire_fhsz"]["url"],
            "https://osfm.fire.ca.gov/what-we-do/community-wildfire-preparedness-and-mitigation/fire-hazard-severity-zones",
        )
        self.assertEqual(sources["fema_nfhl"]["last_verified"], "2026-06-12")
        self.assertEqual(sources["cgs_tsunami_hazard_area"]["last_verified"], "2026-06-12")
        self.assertNotIn("Dublin", sources["alameda_cahmp"]["notes"])
        self.assertNotIn("Livermore", sources["alameda_cahmp"]["notes"])


if __name__ == "__main__":
    unittest.main()
