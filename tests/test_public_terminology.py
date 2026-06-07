import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from hazard_engine import display_data_status
from testing_utils import set_test_resident_state


BASE_DIR = Path(__file__).resolve().parents[1]


class PublicTerminologyTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_structured_status_labels_are_plain_language(self):
        self.assertEqual(display_data_status("checked"), "Mapped match found")
        self.assertEqual(display_data_status("not_in_layer"), "No mapped match found")
        self.assertEqual(display_data_status("not_checked"), "Not checked")
        self.assertEqual(display_data_status("fallback_used"), "General area priority")

    def test_risk_summary_separates_evidence_from_priority(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "location_mode": "zip",
        })
        html = self.client.get("/risk_summary").get_data(as_text=True)
        self.assertIn("How To Read This", html)
        self.assertIn("Mapped evidence and priority are different.", html)
        self.assertIn("General area priority", html)

    def test_cgs_card_has_plain_english_sections(self):
        fixture = [{
            "dataset_id": "cgs_liquefaction_remote",
            "map_layer_key": "liquefaction",
            "dataset_name": "CGS Liquefaction Zones",
            "checked": True,
            "data_available": True,
            "matched": True,
            "exposure": "mapped_match",
            "result_label": "Inside a CGS mapped liquefaction zone.",
            "status_label": "Mapped match found",
            "plain_definition": "Liquefaction can happen when strong earthquake shaking makes loose, wet soil temporarily lose strength.",
            "what_this_means": "The location appears inside a CGS mapped area where liquefaction may occur during a strong earthquake.",
            "why_it_matters": "CGS maps regulatory seismic hazard zones.",
            "what_this_does_not_mean": "This does not guarantee damage and is not a site-specific engineering report.",
            "priority_band": "Mapped evidence",
            "ranking_score": None,
            "evidence_tier": "official_mapped_data",
            "claim_type": "regulatory_zone",
            "precision": "address_point",
            "source_summary": "CGS maps regulatory seismic hazard zones.",
            "source_id": "cgs_liquefaction",
            "source_agency": "California Geological Survey",
            "source_url": "https://www.conservation.ca.gov/cgs/shma",
            "effective_date": "2025",
            "checked_at": "2026-06-06T12:00:00Z",
            "public_claim_status": "official_provisional",
            "matched_features": [],
            "limitations": ["Not a property-specific determination."],
        }]
        with patch("hazard_engine.check_cgs_layers", return_value=fixture):
            set_test_resident_state(self.client, {
                "zip_code": "94619",
                "location_mode": "address",
                "address": "Test address, Oakland, California",
                "lat": 37.79,
                "lon": -122.19,
            })
            html = self.client.get("/hazards/earthquake").get_data(as_text=True)
        self.assertIn("Plain meaning:", html)
        self.assertIn("What this means:", html)
        self.assertIn("What this does not mean:", html)
        self.assertIn("loose, wet soil temporarily lose strength", html)

    def test_unknown_exposure_is_not_a_public_label(self):
        templates = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (BASE_DIR / "templates").glob("*.html")
        ).lower()
        self.assertNotIn("unknown exposure", templates)

    def test_public_pages_avoid_banned_reassurance_phrases(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "location_mode": "zip",
        })
        text = " ".join(
            self.client.get(path).get_data(as_text=True).lower()
            for path in ("/risk_summary", "/hazards", "/map")
        )
        for phrase in (
            "low risk",
            "no risk",
            "minimal risk",
            "not exposed",
            "no hazard",
            "hazard-free",
        ):
            self.assertNotIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
