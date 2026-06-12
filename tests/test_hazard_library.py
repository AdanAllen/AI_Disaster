import unittest
from unittest.mock import patch

from app import app
from hazard_guide import build_hazard_detail_guide, build_hazard_library
from testing_utils import set_test_resident_state


BANNED_PUBLIC_TEXT = (
    "Highest-priority hazards",
    "Full ranked list",
    "Exposure evidence",
    "View API",
    "All hazards JSON",
    "Unknown mapped-zone exposure",
    "Not determined from checked map data",
    "Legacy scoring",
    "0/10",
    "View JSON",
    "Top hazards for this profile",
)


def source(source_id, name, agency="Official agency", reviewed=False):
    return {
        "source_id": source_id,
        "name": name,
        "agency": agency,
        "url": f"https://example.gov/{source_id}",
        "use_in_app": f"Supports {name}.",
        "review_status": "reviewed" if reviewed else "draft_reviewed",
        "confidence": "source_backed" if reviewed else "mixed_support",
    }


def hazard(slug):
    names = {"earthquake": "Earthquake", "flood": "Flood", "wildfire": "Wildfire"}
    sources = [
        source("alameda_county_lhmp", "Alameda County Local Hazard Mitigation Plan"),
        source("ready_recovery", "Recovering from Disaster", "Ready.gov", True),
    ]
    if slug == "earthquake":
        sources += [
            source("usgs_faults", "Fault and earthquake information", "USGS"),
            source("cgs_alquist_priolo", "Alquist-Priolo zones", "CGS"),
            source("cgs_liquefaction", "Liquefaction zones", "CGS"),
            source("cgs_earthquake_landslide", "Earthquake-induced landslide zones", "CGS"),
            source("ready_earthquakes", "Earthquakes", "Ready.gov", True),
        ]
    elif slug == "flood":
        sources += [
            source("fema_nfhl", "National Flood Hazard Layer", "FEMA"),
            source("ready_floods", "Floods", "Ready.gov", True),
        ]
    else:
        sources += [
            source("calfire_fhsz", "Fire Hazard Severity Zones", "CAL FIRE"),
            source("berkeley_fire_evacuation", "Fire Weather and Evacuation", "City of Berkeley", True),
        ]
    return {
        "slug": slug,
        "name": names[slug],
        "sources": sources,
        "matched_layers": [],
        "additional_geospatial_evidence": [],
        "normalized_mapped_evidence": [],
        "action_steps": [],
        "recovery_questions": [],
        "specialized_guidance": {},
        "limitations": [],
        "match_type": "none",
    }


class HazardLibraryRouteTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_index_is_a_library_not_a_ranked_dashboard(self):
        html = self.client.get("/hazards").get_data(as_text=True)

        self.assertIn("Alameda County hazards", html)
        self.assertIn("Hazards StayReady checks", html)
        self.assertIn("Core hazard guides", html)
        self.assertIn("Other regional hazards under review", html)
        self.assertLess(html.index(">Earthquake<"), html.index(">Flood<"))
        self.assertLess(html.index(">Flood<"), html.index(">Wildfire<"))
        for text in BANNED_PUBLIC_TEXT:
            self.assertNotIn(text, html)

    def test_index_shows_existing_secondary_hazards_without_full_guide_claims(self):
        html = self.client.get("/hazards").get_data(as_text=True)

        for label in (
            "Landslide",
            "Tsunami",
            "Dam failure",
            "Drought",
            "Extreme heat",
            "Smoke and poor air quality",
            "Power and utility disruption",
            "Severe storms",
        ):
            self.assertIn(label, html)
        self.assertIn("still reviewing local source coverage", html)

    def test_active_location_uses_compact_context_and_summary_cta(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "location_mode": "zip",
        })
        html = self.client.get("/hazards").get_data(as_text=True)

        self.assertIn("ZIP 94619", html)
        self.assertIn("View your Risk Summary for ranked preparedness priorities", html)
        self.assertIn("View summary", html)
        self.assertNotIn("hazard-profile-form", html)

    def test_local_plan_secondary_hazards_are_cited(self):
        set_test_resident_state(self.client, {
            "zip_code": "94501",
            "location_mode": "address",
            "address": "Shore Line Drive, Alameda, CA",
            "lat": 37.754029,
            "lon": -122.24918,
            "city": "Alameda",
        })
        html = self.client.get("/hazards").get_data(as_text=True)

        self.assertIn("Climate Adaptation and Hazard Mitigation Plan", html)
        self.assertIn("does not yet provide a full address-level guide", html)

    def test_detail_pages_share_public_guide_structure(self):
        for slug in ("earthquake", "flood", "wildfire"):
            html = self.client.get(f"/hazards/{slug}").get_data(as_text=True)
            self.assertIn("Why this matters locally", html)
            self.assertIn("What StayReady checks", html)
            self.assertIn("What to do", html)
            self.assertIn("Recovery readiness", html)
            self.assertIn("Sources and methodology", html)
            self.assertIn("Add an address to check this layer", html)
            self.assertNotIn("hazard-profile-form", html)
            for text in BANNED_PUBLIC_TEXT:
                self.assertNotIn(text, html)

    def test_earthquake_has_five_independent_checks(self):
        html = self.client.get("/hazards/earthquake").get_data(as_text=True)
        checks = (
            "Regional earthquake shaking context",
            "Nearby fault context",
            "CGS Alquist-Priolo fault rupture zones",
            "CGS liquefaction zones",
            "CGS earthquake-induced landslide zones",
        )
        for check in checks:
            self.assertEqual(html.count(check), 1)

    def test_existing_hazard_apis_keep_their_contract(self):
        for path in ("/api/hazards", "/api/hazards/earthquake", "/api/top-risks"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.is_json)


class HazardGuidePresentationTests(unittest.TestCase):
    def test_mixed_earthquake_evidence_stays_separate(self):
        quake = hazard("earthquake")
        quake["match_type"] = "near_fault"
        quake["matched_layers"] = [{"name": "Hayward Fault"}]
        quake["additional_geospatial_evidence"] = [
            {
                "dataset_id": "cgs_alquist_priolo_remote",
                "checked": True,
                "data_available": True,
                "matched": False,
            },
            {
                "dataset_id": "cgs_liquefaction_remote",
                "checked": True,
                "data_available": True,
                "matched": True,
            },
            {
                "dataset_id": "cgs_earthquake_landslide_remote",
                "checked": False,
                "data_available": False,
                "matched": None,
            },
        ]
        guide = build_hazard_detail_guide(quake, {
            "address": "Test address",
            "has_precise_location": True,
            "display_name": "Test address",
        })
        statuses = {check["key"]: check["status"] for check in guide["checks"]}

        self.assertEqual(statuses["fault_proximity"], "Nearby hazard context found")
        self.assertEqual(
            statuses["cgs_alquist_priolo_remote"],
            "Checked: no address-level match found",
        )
        self.assertEqual(statuses["cgs_liquefaction_remote"], "Mapped match found")
        self.assertEqual(statuses["cgs_earthquake_landslide_remote"], "Data unavailable")

    def test_library_order_does_not_follow_input_ranking(self):
        payload = build_hazard_library(
            [hazard("wildfire"), hazard("flood"), hazard("earthquake")],
            {},
        )
        self.assertEqual(
            [item["slug"] for item in payload["hazards"]],
            ["earthquake", "flood", "wildfire"],
        )

    @patch("hazard_guide._public_sources")
    def test_source_statuses_are_public_labels(self, public_sources):
        public_sources.return_value = [{
            "name": "Official source",
            "agency": "Agency",
            "url": "https://example.gov",
            "supports": "Supports this check.",
            "status_label": "Official source; StayReady integration under review",
        }]
        guide = build_hazard_detail_guide(hazard("flood"), {})
        self.assertEqual(
            guide["sources"][0]["status_label"],
            "Official source; StayReady integration under review",
        )


if __name__ == "__main__":
    unittest.main()
