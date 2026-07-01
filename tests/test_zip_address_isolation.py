import unittest
from unittest.mock import patch

import app as stayready
from hazard_engine import build_hazard_results, clear_location_check_cache, merge_structured_result
from hazard_priority import rank_hazards_for_risk_summary
from pydantic_models import LocationResult
from testing_utils import set_test_resident_state


def cgs_evidence(dataset_id, layer_key, matched=False):
    return {
        "dataset_id": dataset_id,
        "map_layer_key": layer_key,
        "dataset_name": f"CGS {layer_key}",
        "checked": True,
        "data_available": True,
        "matched": matched,
        "exposure": "mapped_match" if matched else "no_mapped_match",
        "result_label": "Mapped match found." if matched else "No mapped match found.",
        "status_label": "Mapped match found" if matched else "No mapped match found",
        "claim_type": "regulatory_zone",
        "source_agency": "California Geological Survey",
        "source_url": "https://www.conservation.ca.gov/cgs",
        "matched_features": [{"name": "Test zone"}] if matched else [],
        "limitations": ["Not a property-specific determination."],
    }


def location(city="Berkeley", address=True):
    return LocationResult(
        input_address="Civic test location" if address else "94704",
        formatted_address=f"{city} City Hall, {city}, CA" if address else "",
        lat=37.8696 if address else None,
        lon=-122.2727 if address else None,
        city=city if address else "",
        county="Alameda County" if address else "",
        zip_code="94704",
    )


def checks(*, unavailable=False, earthquake_match=False):
    if unavailable:
        base = {"checked": False, "data_status": "data_unavailable", "layers": []}
        return {
            "flood": dict(base),
            "wildfire": dict(base),
            "fault": dict(base),
            "earthquake_cgs": [],
            "tsunami_cgs": [],
            "dam_inundation": {},
        }
    return {
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
            cgs_evidence("cgs_liquefaction_remote", "liquefaction", earthquake_match)
        ],
        "tsunami_cgs": [],
        "dam_inundation": {},
    }


class ZipAddressIsolationTests(unittest.TestCase):
    def setUp(self):
        clear_location_check_cache()
        self.hazards = [
            {"name": "Flood", "slug": "flood", "priority_score": 0},
            {"name": "Wildfire", "slug": "wildfire", "priority_score": 0},
            {"name": "Earthquake", "slug": "earthquake", "priority_score": 0},
        ]
        self.low_zip = {
            "earthquake": {"score": 1, "explanation": "Low legacy value."},
            "flood": {"score": 2, "explanation": "Low legacy value."},
            "wildfire": {"score": 3, "explanation": "Low legacy value."},
        }
        self.high_zip = {
            "earthquake": {"score": 10, "explanation": "High legacy value."},
            "flood": {"score": 9, "explanation": "High legacy value."},
            "wildfire": {"score": 8, "explanation": "High legacy value."},
        }

    @patch("hazard_engine._collect_location_checks")
    def test_address_order_is_invariant_to_zip_scores(self, collect):
        collect.return_value = checks()
        low = build_hazard_results(self.hazards, location(), self.low_zip)
        high = build_hazard_results(self.hazards, location(), self.high_zip)
        self.assertEqual([item.hazard_id for item in low], [item.hazard_id for item in high])
        self.assertTrue(all(item.legacy_score is None for item in low + high))

    @patch("hazard_engine._collect_location_checks")
    def test_address_display_labels_are_invariant_to_zip_scores(self, collect):
        collect.return_value = checks()
        low = [merge_structured_result({}, item) for item in build_hazard_results(self.hazards, location(), self.low_zip)]
        high = [merge_structured_result({}, item) for item in build_hazard_results(self.hazards, location(), self.high_zip)]
        self.assertEqual(
            [
                (item["structured_result"]["hazard_id"], item["risk_level"], item["scope"])
                for item in low
            ],
            [
                (item["structured_result"]["hazard_id"], item["risk_level"], item["scope"])
                for item in high
            ],
        )

    @patch("hazard_engine._collect_location_checks")
    def test_unavailable_address_data_does_not_use_zip_score(self, collect):
        collect.return_value = checks(unavailable=True)
        results = build_hazard_results(self.hazards, location(), self.high_zip)
        core_results = [item for item in results if item.hazard_id in {"earthquake", "flood", "wildfire"}]
        self.assertTrue(all(item.data_status == "data_unavailable" for item in core_results))
        self.assertTrue(all(item.exposure_level == "unknown" for item in core_results))
        self.assertTrue(all(item.legacy_score is None for item in results))

    @patch("hazard_engine._collect_location_checks")
    def test_address_and_zip_only_results_remain_separate(self, collect):
        collect.return_value = checks()
        address_result = build_hazard_results(
            [{"name": "Flood", "slug": "flood", "priority_score": 0}],
            location(),
            self.high_zip,
        )[0]
        zip_result = build_hazard_results(
            [{"name": "Flood", "slug": "flood", "priority_score": 0}],
            location(address=False),
            self.high_zip,
        )[0]
        self.assertEqual(address_result.scope, "address_level")
        self.assertEqual(zip_result.scope, "zip_estimate")
        self.assertEqual(zip_result.basis, "zip_csv_heuristic")
        self.assertEqual(zip_result.legacy_score, 9)

    def test_zip_only_route_metadata_does_not_hide_zip_fallback_scope(self):
        zip_location = location(address=False)
        zip_location.county = "Alameda County"
        result = build_hazard_results(
            [{"name": "Earthquake", "slug": "earthquake", "priority_score": 0}],
            zip_location,
            self.high_zip,
        )[0]
        self.assertEqual(result.scope, "zip_estimate")
        self.assertEqual(result.basis, "zip_csv_heuristic")
        self.assertEqual(result.location_precision, "zip")
        self.assertIsNone(result.is_in_hazard_zone)

    def test_zip_snapshot_is_explicitly_scoped_as_fallback(self):
        with patch.dict(stayready.zip_risk_data, {
            "94704": {
                "Earthquake_Risk_Score": 7,
                "Earthquake_Risk_Explanation": "Broad ZIP context.",
                "Flood_Risk_Score": 4,
                "Flood_Risk_Explanation": "Broad ZIP context.",
                "Wildfire_Risk_Score": 2,
                "Wildfire_Risk_Explanation": "Broad ZIP context.",
            }
        }, clear=True):
            snapshot = stayready.get_zip_risk_snapshot_for_context({
                "location_mode": "zip",
                "zip_code": "94704",
            })
        self.assertTrue(snapshot)
        self.assertTrue(all(item["evidence_scope"] == "zip_fallback" for item in snapshot.values()))

    def test_address_context_rejects_even_preloaded_zip_snapshot(self):
        hazard = {"name": "Earthquake", "slug": "earthquake", "priority_score": 5}
        result = stayready.personalize_hazard(hazard, {}, {
            "location_mode": "address",
            "zip_risk_snapshot": self.high_zip,
        })
        self.assertEqual(result["priority_score"], 5)
        self.assertNotIn("local_risk_score", result)
        self.assertEqual(result["personalization_notes"], [])

    def test_zip_only_information_is_not_address_mapped_evidence(self):
        result = build_hazard_results(
            [{"name": "Earthquake", "slug": "earthquake", "priority_score": 0}],
            location(address=False),
            self.high_zip,
        )[0]
        self.assertEqual(result.scope, "zip_estimate")
        self.assertEqual(result.location_precision, "zip")
        self.assertIsNone(result.is_in_hazard_zone)
        self.assertEqual(result.hazard_exposure, "regional_context")
        self.assertIn("ZIP-level fallback context", result.why_shown)
        ranked = rank_hazards_for_risk_summary("", [result.model_dump()])
        earthquake = next(item for item in ranked if item["slug"] == "earthquake")
        self.assertEqual(earthquake["mapped_finding_status"], "not_supported")

    def test_zip_only_summary_is_visibly_labeled_as_broad_context(self):
        client = stayready.app.test_client()
        set_test_resident_state(client, {
            "zip_code": "94704",
            "lat": 37.87,
            "lon": -122.27,
            "address": "",
            "input_address": "94704",
            "county": "Alameda County",
            "city": "",
            "location_mode": "zip",
            "household": "",
            "preparedness": "",
            "household_tags": [],
        })
        html = client.get("/risk_summary").get_data(as_text=True)
        self.assertIn("broad ZIP-level fallback context", html)
        self.assertIn("not an address or property finding", html)
        self.assertIn("do not show whether an address intersects", html)

    @patch("hazard_engine._collect_location_checks")
    def test_legitimate_address_gis_match_is_unchanged(self, collect):
        collect.return_value = checks(earthquake_match=True)
        low = build_hazard_results(self.hazards, location(), self.low_zip)
        high = build_hazard_results(self.hazards, location(), self.high_zip)
        for results in (low, high):
            earthquake = next(item for item in results if item.hazard_id == "earthquake")
            self.assertEqual(earthquake.hazard_exposure, "mapped_match")
            self.assertTrue(earthquake.is_in_hazard_zone)
            self.assertIsNone(earthquake.legacy_score)

    def test_berkeley_and_oakland_civic_nonmatches_remain_nonmatches(self):
        for city in ("Berkeley", "Oakland"):
            evidence = cgs_evidence("cgs_alquist_priolo_remote", "alquist-priolo", False)
            ranked = rank_hazards_for_risk_summary(city, [{
                "slug": "earthquake",
                "data_status": "not_in_layer",
                "is_in_hazard_zone": False,
                "additional_geospatial_evidence": [evidence],
                "matched_layers": [],
            }])
            earthquake = next(item for item in ranked if item["slug"] == "earthquake")
            self.assertNotEqual(earthquake["mapped_finding_status"], "significant_official_finding")
            self.assertEqual(earthquake["local_exposure"]["polygon_intersections"], [])


if __name__ == "__main__":
    unittest.main()
