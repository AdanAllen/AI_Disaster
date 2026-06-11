import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask, render_template, session
from pydantic import ValidationError

import hazard_engine
from hazard_engine import build_hazard_results, check_fault_layer, check_flood_layer
from location_service import location_from_session
from pydantic_models import LHMPLocationFact, LocationResult
from resident_guidance_engine import _coordinate_rule_matches, _match_facts, _named_area_match


BASE_DIR = Path(__file__).resolve().parents[1]


def valid_fact(**overrides):
    fact = {
        "id": "test_fact",
        "jurisdiction": "Oakland",
        "applies_to_jurisdictions": [],
        "hazard": "wildfire",
        "evidence_tier": "area_based",
        "geography_type": "hillside",
        "named_areas": ["Oakland Hills"],
        "location_aliases": ["oakland hills"],
        "coordinate_rule": {},
        "location_cue": "Oakland Hills",
        "resident_meaning": "Reviewed area context.",
        "before_actions": ["Prepare."],
        "during_actions": ["Follow official instructions."],
        "after_actions": ["Wait for re-entry guidance."],
        "recovery_steps": ["Keep records available."],
        "resident_impacts": ["Evacuation"],
        "household_factors": [],
        "infrastructure_dependencies": ["Roads"],
        "requires_gis_confirmation": True,
        "precision_limitations": ["This is not parcel-level evidence."],
        "source_document": "official-plan.pdf",
        "source_page": 12,
        "source_excerpt_summary": "Reviewed summary.",
        "source_name": "Official plan",
        "source_url": "https://example.gov/plan",
        "review_status": "reviewed",
    }
    fact.update(overrides)
    return fact


class GISAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        Path(self.temp_dir.name, "static").mkdir()
        self.base_patch = patch.object(hazard_engine, "BASE_DIR", self.temp_dir.name)
        self.base_patch.start()
        self.cgs_patch = patch.object(hazard_engine, "check_cgs_layers", return_value=[])
        self.cgs_patch.start()
        hazard_engine.load_geojson.cache_clear()

    def tearDown(self):
        hazard_engine.load_geojson.cache_clear()
        self.cgs_patch.stop()
        self.base_patch.stop()
        self.temp_dir.cleanup()

    def write_layer(self, filename, payload):
        Path(self.temp_dir.name, "static", filename).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        hazard_engine.load_geojson.cache_clear()

    def test_missing_layer_is_not_checked(self):
        result = check_flood_layer(37.8, -122.2)
        self.assertFalse(result["checked"])
        self.assertEqual(result["data_status"], "data_unavailable")
        self.assertIsNone(result["inside"])

    def test_empty_layer_is_not_checked(self):
        self.write_layer("FldHaz.geojson", {"type": "FeatureCollection", "features": []})
        result = check_flood_layer(37.8, -122.2)
        self.assertFalse(result["checked"])
        self.assertEqual(result["data_status"], "data_unavailable")

    def test_unreadable_json_is_not_checked(self):
        Path(self.temp_dir.name, "static", "FldHaz.geojson").write_text("{broken", encoding="utf-8")
        hazard_engine.load_geojson.cache_clear()
        result = check_flood_layer(37.8, -122.2)
        self.assertFalse(result["checked"])
        self.assertEqual(result["data_status"], "data_unavailable")

    def test_layer_with_no_valid_geometry_is_not_checked(self):
        self.write_layer(
            "FldHaz.geojson",
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": None}],
            },
        )
        result = check_flood_layer(37.8, -122.2)
        self.assertFalse(result["checked"])
        self.assertEqual(result["data_status"], "data_unavailable")

    def test_unavailable_layer_does_not_create_low_exposure(self):
        location = LocationResult(
            input_address="1 Test Street",
            formatted_address="1 Test Street, Oakland, California",
            lat=37.8,
            lon=-122.2,
            city="Oakland",
            county="Alameda County",
            zip_code="94601",
        )
        hazards = [{"name": "Flood", "slug": "flood", "risk_level": "low", "priority_score": 1}]
        result = build_hazard_results(hazards, location, {})[0]
        self.assertEqual(result.data_status, "data_unavailable")
        self.assertEqual(result.exposure_level, "unknown")
        self.assertIsNone(result.is_in_hazard_zone)

    def test_provisional_flood_non_match_does_not_create_low_exposure(self):
        location = LocationResult(
            input_address="1 Test Street",
            formatted_address="1 Test Street, Oakland, California",
            lat=37.95,
            lon=-122.45,
            city="Oakland",
            county="Alameda County",
            zip_code="94601",
        )
        hazards = [{"name": "Flood", "slug": "flood", "risk_level": "low", "priority_score": 1}]
        with patch("hazard_engine.check_flood_layer") as flood_check:
            flood_check.return_value = {
                "checked": True,
                "data_status": "not_in_layer",
                "message": "",
                "inside": False,
                "layers": [],
                "geospatial_evidence": {
                    "claim_type": "regulatory_zone",
                    "checked_at": "2026-06-06T12:00:00Z",
                    "effective_date": None,
                    "public_claim_status": "official_provisional",
                    "source_agency": "Federal Emergency Management Agency",
                    "source_url": "https://www.fema.gov/flood-maps/national-flood-hazard-layer",
                    "limitations": ["A non-match does not mean the property is safe from flooding."],
                },
            }
            result = build_hazard_results(hazards, location, {})[0]
        self.assertEqual(result.data_status, "not_in_layer")
        self.assertEqual(result.exposure_level, "unknown")
        self.assertFalse(result.is_in_hazard_zone)

    def test_provisional_wildfire_non_match_does_not_create_low_exposure(self):
        location = LocationResult(
            input_address="1 Test Street",
            formatted_address="1 Test Street, Oakland, California",
            lat=37.80,
            lon=-122.20,
            city="Oakland",
            county="Alameda County",
            zip_code="94601",
        )
        hazards = [{"name": "Wildfire", "slug": "wildfire", "risk_level": "low", "priority_score": 1}]
        with patch("hazard_engine.check_wildfire_layer") as wildfire_check:
            wildfire_check.return_value = {
                "checked": True,
                "data_status": "not_in_layer",
                "message": "",
                "inside": False,
                "layers": [],
                "geospatial_evidence": {
                    "claim_type": "hazard_zone",
                    "checked_at": "2026-06-06T12:00:00Z",
                    "effective_date": None,
                    "public_claim_status": "official_provisional",
                    "source_agency": "California Department of Forestry and Fire Protection",
                    "source_url": "https://www.fire.ca.gov/osfm/what-we-do/community-wildfire-preparedness-and-mitigation/fire-hazard-severity-zones",
                    "limitations": ["A non-match is not a safety determination."],
                },
            }
            result = build_hazard_results(hazards, location, {})[0]
        self.assertEqual(result.data_status, "not_in_layer")
        self.assertEqual(result.exposure_level, "unknown")
        self.assertFalse(result.is_in_hazard_zone)
        self.assertIn("not a safety determination", result.why_shown.lower())
        self.assertIn("smoke", " ".join(result.limitations).lower())

    def test_fault_proximity_is_not_zone_membership(self):
        self.write_layer(
            "Fault_lines.Geojson",
            {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"fault_name": "Test Fault"},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-122.2, 37.79], [-122.2, 37.81]],
                    },
                }],
            },
        )
        check = check_fault_layer(37.8, -122.2)
        self.assertTrue(check["checked"])
        self.assertTrue(check["near"])

        location = LocationResult(
            input_address="1 Test Street",
            formatted_address="1 Test Street, Oakland, California",
            lat=37.8,
            lon=-122.2,
            city="Oakland",
            county="Alameda County",
            zip_code="94601",
        )
        result = hazard_engine._earthquake_address_result(
            {"name": "Earthquake", "slug": "earthquake"},
            location,
            check,
            {},
        )
        self.assertIsNone(result.is_in_hazard_zone)
        self.assertEqual(result.match_type, "near_fault")
        self.assertEqual(result.exposure_level, "unknown")


class LocationFactSafetyTests(unittest.TestCase):
    def test_all_25_facts_pass_strict_validation(self):
        facts = json.loads((BASE_DIR / "lhmp_location_facts.json").read_text(encoding="utf-8"))
        validated = [LHMPLocationFact(**fact) for fact in facts]
        self.assertEqual(len(validated), 25)

    def test_required_list_cannot_be_empty(self):
        with self.assertRaises(ValidationError):
            LHMPLocationFact(**valid_fact(precision_limitations=[]))

    def test_source_page_is_strict(self):
        with self.assertRaises(ValidationError):
            LHMPLocationFact(**valid_fact(source_page="page twelve"))
        with self.assertRaises(ValidationError):
            LHMPLocationFact(**valid_fact(source_page=True))

    def test_one_sided_coordinate_rule_is_rejected(self):
        with self.assertRaises(ValidationError):
            LHMPLocationFact(**valid_fact(coordinate_rule={"min_lon": -122.2}))

    def test_complete_coordinate_boundaries_are_inclusive(self):
        rule = {
            "min_lat": 37.7,
            "max_lat": 37.8,
            "min_lon": -122.3,
            "max_lon": -122.2,
        }
        context = {"location_result": {"lat": 37.8, "lon": -122.2}}
        self.assertTrue(_coordinate_rule_matches(rule, context))
        context["location_result"]["lon"] = -122.199
        self.assertFalse(_coordinate_rule_matches(rule, context))
        self.assertFalse(_coordinate_rule_matches({"min_lon": -122.3}, context))

    def test_generic_word_inside_street_name_does_not_match(self):
        context = {
            "address": "123 Shoreline Drive, Alameda, California",
            "display_name": "123 Shoreline Drive, Alameda, California",
            "location_result": {
                "formatted_address": "123 Shoreline Drive, Alameda, California",
                "neighborhood": "",
            },
        }
        self.assertIsNone(_named_area_match("shoreline", context))

    def test_reviewed_multiword_place_name_matches(self):
        context = {
            "address": "123 Example Street, Bay Farm Island, Alameda, California",
            "display_name": "123 Example Street, Bay Farm Island, Alameda, California",
            "location_result": {
                "formatted_address": "123 Example Street, Bay Farm Island, Alameda, California",
                "neighborhood": "Bay Farm Island",
            },
        }
        match = _named_area_match("bay farm island", context)
        self.assertEqual(match["method"], "structured_named_area")

    @patch("resident_guidance_engine.load_hazard_facts")
    def test_equal_confidence_multiple_matches_are_suppressed(self, load_facts):
        first = valid_fact(id="first", location_aliases=["oakland hills"])
        second = valid_fact(id="second", location_aliases=["oakland hills"])
        load_facts.return_value = [first, second]
        context = {
            "address": "Example, Oakland Hills, Oakland, California",
            "display_name": "Example, Oakland Hills, Oakland, California",
            "location_result": {
                "formatted_address": "Example, Oakland Hills, Oakland, California",
                "neighborhood": "Oakland Hills",
            },
        }
        matches = _match_facts("Oakland", "wildfire", context)
        self.assertEqual(matches["area"], [])
        self.assertEqual(len(matches["ambiguous_area_matches"]), 2)

    def test_unincorporated_communities_are_detected_without_alameda_city_confusion(self):
        for community in ("Ashland", "Cherryland", "Fairview", "Sunol", "Hayward Acres"):
            result = location_from_session({
                "address": f"123 Example Street, {community}, Alameda County, California",
                "zip_code": "94546",
                "lat": 37.7,
                "lon": -122.1,
                "location_mode": "address",
            })
            self.assertEqual(result.city, community)

    def test_legacy_berkeley_coordinate_context_is_disabled(self):
        location = LocationResult(
            input_address="1020 Sierra Street",
            formatted_address="1020 Sierra Street, Berkeley, California",
            lat=37.89,
            lon=-122.255,
            city="Berkeley",
            county="Alameda County",
            zip_code="94707",
        )
        self.assertEqual(hazard_engine._location_specific_context(location, "wildfire"), [])


class RenderedLimitationsTests(unittest.TestCase):
    def test_risk_summary_renders_limitations_in_labeled_details(self):
        app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
        app.secret_key = "test"
        for endpoint in (
            "home",
            "risk_summary",
            "map",
            "hazards_dashboard",
            "resources",
            "sources",
            "about",
            "privacy_basics",
            "live_earthquake_map",
        ):
            app.add_url_rule(
                f"/{endpoint}",
                endpoint,
                lambda: "",
            )

        resident_plan = {
            "address_summary": {
                "summary": "Summary",
                "display_name": "Test address",
                "city": "Oakland",
                "county": "Alameda County",
                "zip_code": "94601",
                "precision_label": "Address point",
                "gis_status": "Official layer unavailable — not checked",
            },
            "hazards": [{
                "priority": 1,
                "hazard": "Flood",
                "exposure_level": "Unknown",
                "evidence_status_label": "Not checked",
                "evidence_tier_label": "Citywide source",
                "priority_label": "General area priority",
                "evidence_badges": ["Jurisdiction-level", "Official layer unavailable — not checked"],
                "why_it_matters": "Jurisdiction context only.",
                "limitations": ["Official layer unavailable — not checked."],
                "location_matches": [],
                "official_mapped_evidence": [],
                "before_actions": [],
                "during_actions": [],
                "after_actions": [],
                "recovery_steps": [],
                "checked": [],
                "not_checked": ["Official layer unavailable — not checked."],
                "sources": [],
            }],
            "household_context": {"has_context": False},
            "household_priorities": [],
            "what_to_do_now": [],
            "recovery_plan": [],
            "checks": {"checked": [], "not_checked": []},
            "additional_local_hazards": [],
            "sources": [],
            "limits": [],
        }
        with app.test_request_context("/risk_summary"):
            session["zip_code"] = "94601"
            html = render_template(
                "risk_summary.html",
                resident_plan=resident_plan,
                warning_message=None,
                empty_state=False,
            )
        self.assertIn("Limits and source details", html)
        self.assertIn("Important limits", html)
        self.assertIn("Official layer unavailable", html)


if __name__ == "__main__":
    unittest.main()
