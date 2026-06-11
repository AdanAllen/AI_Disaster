import unittest
from unittest.mock import patch

from hazard_engine import build_hazard_results, clear_location_check_cache
from pydantic_models import LocationResult


def cgs_evidence(dataset_id, layer_key, matched):
    return {
        "dataset_id": dataset_id,
        "map_layer_key": layer_key,
        "dataset_name": f"Test {layer_key}",
        "checked": True,
        "data_available": True,
        "matched": matched,
        "exposure": "mapped_match" if matched else "no_mapped_match",
        "result_label": f"Inside test {layer_key} zone." if matched else "No mapped match found.",
        "status_label": "Mapped match found" if matched else "No mapped match found",
        "plain_definition": "Test definition.",
        "what_this_means": "Test mapped evidence.",
        "why_it_matters": "Test official evidence.",
        "what_this_does_not_mean": "Not a damage prediction.",
        "priority_band": "Mapped evidence" if matched else "Not an address-level priority ranking",
        "ranking_score": None,
        "evidence_tier": "official_mapped_data",
        "claim_type": "hazard_zone" if layer_key == "tsunami" else "regulatory_zone",
        "precision": "address_point",
        "source_summary": "Test official evidence.",
        "source_id": "cgs_tsunami_hazard_area" if layer_key == "tsunami" else "cgs_liquefaction",
        "source_agency": "California Geological Survey",
        "source_url": "https://www.conservation.ca.gov/cgs",
        "effective_date": "2025",
        "checked_at": "2026-06-11T12:00:00Z",
        "public_claim_status": "official_provisional",
        "matched_features": [{"name": "Test zone"}] if matched else [],
        "limitations": ["Not a property-specific determination."],
    }


class PreparednessPriorityIntegrationTests(unittest.TestCase):
    def setUp(self):
        clear_location_check_cache()
        self.location = LocationResult(
            input_address="Test address",
            formatted_address="Test address, Alameda, California",
            lat=37.754029,
            lon=-122.24918,
            city="Alameda",
            county="Alameda County",
            zip_code="94501",
        )
        self.hazards = [
            {"name": "Flood", "slug": "flood", "priority_score": 0},
            {"name": "Wildfire", "slug": "wildfire", "priority_score": 0},
            {"name": "Earthquake", "slug": "earthquake", "priority_score": 0},
        ]

    @patch("hazard_engine._collect_location_checks")
    def test_cgs_match_outranks_legacy_only_wildfire(self, collect):
        collect.return_value = {
            "flood": {"checked": True, "data_status": "not_in_layer", "inside": False, "layers": []},
            "wildfire": {"checked": True, "data_status": "not_in_layer", "inside": False, "layers": []},
            "fault": {
                "checked": True,
                "data_status": "checked",
                "near": False,
                "layers": [],
                "nearest_layer": {},
            },
            "earthquake_cgs": [
                cgs_evidence("cgs_liquefaction_remote", "liquefaction", True)
            ],
            "tsunami_cgs": [],
        }
        results = build_hazard_results(
            self.hazards,
            self.location,
            {
                "wildfire": {"score": 10},
                "earthquake": {"score": 1},
                "flood": {"score": 2},
            },
        )
        self.assertEqual(results[0].hazard_id, "earthquake")
        self.assertEqual(results[0].hazard_exposure, "mapped_match")
        self.assertEqual(results[0].action_priority, "start_here")

    @patch("hazard_engine._collect_location_checks")
    def test_fault_proximity_and_cgs_zone_are_preserved_separately(self, collect):
        collect.return_value = {
            "flood": {"checked": True, "data_status": "not_in_layer", "inside": False, "layers": []},
            "wildfire": {"checked": True, "data_status": "not_in_layer", "inside": False, "layers": []},
            "fault": {
                "checked": True,
                "data_status": "checked",
                "near": True,
                "layers": [{"name": "Test Fault", "distance_km": 0.4}],
                "nearest_layer": {"name": "Test Fault", "distance_km": 0.4},
            },
            "earthquake_cgs": [
                cgs_evidence("cgs_earthquake_landslide_remote", "earthquake-landslide", True)
            ],
            "tsunami_cgs": [],
        }
        earthquake = build_hazard_results(self.hazards, self.location, {})[0]
        self.assertEqual(earthquake.match_type, "near_fault")
        dataset_ids = {
            item["dataset_id"] for item in earthquake.normalized_mapped_evidence
        }
        self.assertIn("usgs_cgs_faults_local", dataset_ids)
        self.assertIn("cgs_earthquake_landslide_remote", dataset_ids)

    @patch("hazard_engine._collect_location_checks")
    def test_tsunami_match_materializes_summary_hazard(self, collect):
        collect.return_value = {
            "flood": {"checked": True, "data_status": "not_in_layer", "inside": False, "layers": []},
            "wildfire": {"checked": True, "data_status": "not_in_layer", "inside": False, "layers": []},
            "fault": {
                "checked": True,
                "data_status": "checked",
                "near": False,
                "layers": [],
                "nearest_layer": {},
            },
            "earthquake_cgs": [],
            "tsunami_cgs": [
                cgs_evidence("cgs_tsunami_hazard_area_remote", "tsunami", True)
            ],
        }
        results = build_hazard_results(self.hazards, self.location, {})
        tsunami = next(item for item in results if item.hazard_id == "tsunami-seiche")
        self.assertEqual(tsunami.hazard_exposure, "mapped_match")
        self.assertEqual(tsunami.action_priority, "start_here")

    @patch("hazard_engine._collect_location_checks")
    def test_unavailable_data_does_not_remove_regional_earthquake_priority(self, collect):
        collect.return_value = {
            "flood": {"checked": False, "data_status": "data_unavailable", "layers": []},
            "wildfire": {"checked": False, "data_status": "data_unavailable", "layers": []},
            "fault": {"checked": False, "data_status": "data_unavailable", "layers": []},
            "earthquake_cgs": [],
            "tsunami_cgs": [],
        }
        results = build_hazard_results(self.hazards, self.location, {})
        earthquake = next(item for item in results if item.hazard_id == "earthquake")
        self.assertEqual(earthquake.hazard_exposure, "not_checked")
        self.assertEqual(earthquake.hazard_importance, "major_regional")
        self.assertEqual(earthquake.action_priority, "important")
        self.assertIn("missing data", " ".join(earthquake.priority_reasons).lower())

    def test_household_context_changes_action_priority_not_exposure(self):
        location = LocationResult(
            input_address="94501",
            formatted_address="",
            zip_code="94501",
        )
        base = build_hazard_results(
            [{"name": "Flood", "slug": "flood", "priority_score": 0}],
            location,
            {"flood": {"score": 2, "explanation": "Fallback context."}},
        )[0]
        personalized = build_hazard_results(
            [{"name": "Flood", "slug": "flood", "priority_score": 0}],
            location,
            {"flood": {"score": 2, "explanation": "Fallback context."}},
            user_context={"household_tags": ["medical"]},
        )[0]
        self.assertEqual(base.hazard_exposure, personalized.hazard_exposure)
        self.assertEqual(personalized.action_priority, "important")
        self.assertIn("not to change mapped exposure", " ".join(personalized.priority_reasons))


if __name__ == "__main__":
    unittest.main()
