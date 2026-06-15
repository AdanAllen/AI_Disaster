import unittest
from unittest.mock import Mock, patch

from app import app
from dam_inundation import (
    BASE_LIMITATIONS,
    SOURCE_AGENCY,
    SOURCE_URL,
    check_dam_inundation,
)
from geospatial.adapters.arcgis_feature_service import _remote_record_count
from geospatial.models import DatasetProvenance
from geospatial.registry import DatasetRegistry
from geospatial.service import GeospatialEvidenceService
from hazard_engine import build_hazard_results, clear_location_check_cache
from pydantic_models import LocationResult
from testing_utils import set_test_resident_state


def dataset():
    return DatasetProvenance(
        dataset_id="dwr_dsod_dam_inundation_remote",
        dataset_version="test-v1",
        hazard_type="dam_failure",
        agency=SOURCE_AGENCY,
        authoritative_landing_url="https://water.ca.gov/dam-inundation",
        exact_service_or_download_url=SOURCE_URL,
        dataset_name="Test DSOD dam inundation polygons",
        claim_type="scenario",
        source_type="remote_service",
        license_terms_notes="Official test fixture.",
        coverage_area="Alameda County test coverage.",
        intended_claim="Point intersection with hypothetical inundation polygons.",
        prohibited_claims=list(BASE_LIMITATIONS),
        effective_date="2025-10-01",
        status="provisional",
    )


class Registry:
    def get(self, dataset_id):
        return dataset() if dataset_id == "dwr_dsod_dam_inundation_remote" else None


def checker():
    return GeospatialEvidenceService(project_root=".", registry=Registry())


def response(payload):
    mocked = Mock()
    mocked.raise_for_status.return_value = None
    mocked.json.return_value = payload
    return mocked


def feature(dam_name, scenario, *, structure="MainDam", loading="Sunny Day"):
    return {
        "attributes": {
            "DamName": dam_name,
            "FailedStr": structure,
            "Scenario": scenario,
            "LoadingScn": loading,
            "HazardCl": "Extremely High",
            "PubDate": 1759276800000,
            "NID": "CA00001",
            "StateID": "1.001",
        }
    }


class DamInundationCheckerTests(unittest.TestCase):
    def setUp(self):
        _remote_record_count.cache_clear()

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_outside_boundary_returns_false(self, get):
        get.side_effect = [response({"count": 1202}), response({"features": []})]
        result = check_dam_inundation(37.80, -122.20, service=checker())
        self.assertEqual(result["data_status"], "checked")
        self.assertFalse(result["inside_inundation_boundary"])
        self.assertEqual(result["matched_dam_scenarios"], [])

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_inside_fixture_polygon_returns_matched_dam_scenario(self, get):
        get.side_effect = [
            response({"count": 1202}),
            response({"features": [feature("Chabot", "Scenario1")]}),
        ]
        result = check_dam_inundation(37.73, -122.12, service=checker())
        self.assertTrue(result["inside_inundation_boundary"])
        self.assertEqual(result["matched_dam_scenarios"][0]["dam_name"], "Chabot")
        self.assertEqual(result["matched_dam_scenarios"][0]["scenario"], "Scenario1")
        self.assertIn("Chabot", result["matched_dam_scenario_names"][0])

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_missing_or_failed_service_returns_data_unavailable(self, get):
        get.side_effect = ValueError("invalid service response")
        result = check_dam_inundation(37.80, -122.20, service=checker())
        self.assertEqual(result["data_status"], "data_unavailable")
        self.assertIsNone(result["inside_inundation_boundary"])
        self.assertNotIn("low risk", " ".join(result["limitations"]).lower())

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_multiple_overlapping_scenarios_return_all_matches(self, get):
        get.side_effect = [
            response({"count": 1202}),
            response({
                "features": [
                    feature("New Upper San Leandro", "Scenario1"),
                    feature("Chabot", "Scenario1"),
                    feature("Chabot", "Scenario2", structure="Spillway1"),
                ]
            }),
        ]
        result = check_dam_inundation(37.74, -122.16, service=checker())
        self.assertTrue(result["inside_inundation_boundary"])
        self.assertEqual(len(result["matched_dam_scenarios"]), 3)
        self.assertEqual(
            {item["dam_name"] for item in result["matched_dam_scenarios"]},
            {"Chabot", "New Upper San Leandro"},
        )

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_source_and_limitations_are_always_included(self, get):
        get.side_effect = [response({"count": 1202}), response({"features": []})]
        checked = check_dam_inundation(37.80, -122.20, service=checker())
        get.side_effect = ValueError("service failed")
        unavailable = check_dam_inundation(37.80, -122.20, service=checker())
        for result in (checked, unavailable):
            self.assertEqual(result["source_agency"], SOURCE_AGENCY)
            self.assertEqual(result["source_url"], SOURCE_URL)
            self.assertTrue(result["limitations"])
            self.assertIn("local emergency officials", " ".join(result["limitations"]))


class DamInundationMapTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        set_test_resident_state(self.client, {
            "zip_code": "94501",
            "lat": 37.754029,
            "lon": -122.24918,
            "address": "Test address",
            "input_address": "Test address",
            "county": "Alameda County",
            "city": "Alameda",
            "location_mode": "address",
        })

    @patch("app.GeospatialEvidenceService.map_geojson")
    def test_map_layer_returns_source_and_limitations(self, map_geojson):
        map_geojson.return_value = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"DamName": "Chabot", "Scenario": "Scenario1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.30, 37.70],
                        [-122.10, 37.70],
                        [-122.10, 37.90],
                        [-122.30, 37.90],
                        [-122.30, 37.70],
                    ]],
                },
            }],
            "available_feature_count": 1,
            "partial": False,
        }
        payload = self.client.get("/api/dam-inundation-layer").get_json()
        self.assertEqual(payload["data_status"], "checked")
        self.assertEqual(payload["source"], SOURCE_AGENCY)
        self.assertTrue(payload["source_url"])
        self.assertTrue(payload["limitations"])
        map_geojson.assert_called_once_with(
            "dwr_dsod_dam_inundation_remote",
            lat=37.754029,
            lon=-122.24918,
            radius_degrees=0.04,
        )

    def test_map_template_has_plain_language_toggle_and_warning(self):
        html = self.client.get("/map").get_data(as_text=True)
        self.assertIn("Dam Failure Inundation", html)
        self.assertIn("Flood &amp; inundation", html)
        self.assertIn("Dam failure inundation area", html)
        self.assertIn("where water could travel if a mapped dam", html)
        self.assertIn("not predictions or live evacuation maps", html)


class DamInundationHazardIntegrationTests(unittest.TestCase):
    def setUp(self):
        clear_location_check_cache()

    @patch("hazard_engine._collect_location_checks")
    def test_dam_failure_hazard_uses_address_level_mapped_evidence(self, collect):
        collect.return_value = {
            "flood": {"checked": False, "data_status": "data_unavailable", "layers": []},
            "wildfire": {"checked": False, "data_status": "data_unavailable", "layers": []},
            "fault": {"checked": False, "data_status": "data_unavailable", "layers": []},
            "earthquake_cgs": [],
            "tsunami_cgs": [],
            "dam_inundation": {
                "dataset_id": "dwr_dsod_dam_inundation_remote",
                "map_layer_key": "dam-inundation",
                "dataset_name": "DSOD Approved Dam Inundation Boundaries",
                "checked": True,
                "data_available": True,
                "matched": True,
                "exposure": "mapped_match",
                "result_label": "Inside a DSOD-approved hypothetical boundary.",
                "status_label": "Mapped match found",
                "plain_definition": "Hypothetical planning boundary.",
                "what_this_means": "The selected point intersects a published boundary.",
                "why_it_matters": "Review evacuation readiness.",
                "what_this_does_not_mean": "Not a prediction or evacuation order.",
                "priority_band": "Mapped evidence",
                "ranking_score": None,
                "evidence_tier": "official_mapped_data",
                "claim_type": "scenario",
                "precision": "address_point",
                "source_summary": "Official DWR/DSOD planning data.",
                "source_id": "dwr_dsod_dam_inundation",
                "source_agency": SOURCE_AGENCY,
                "source_url": "https://water.ca.gov/dam-inundation",
                "effective_date": "2025-10-01",
                "checked_at": "2026-06-15T12:00:00Z",
                "public_claim_status": "official_provisional",
                "matched_features": [{
                    "dam_name": "Chabot",
                    "scenario": "Scenario1",
                    "display_name": "Chabot · MainDam · Scenario1 · Sunny Day",
                }],
                "limitations": list(BASE_LIMITATIONS),
            },
        }
        location = LocationResult(
            input_address="Test address",
            formatted_address="Test address, Alameda County, California",
            lat=37.73,
            lon=-122.12,
            city="San Leandro",
            county="Alameda County",
            zip_code="94577",
        )
        result = build_hazard_results(
            [{"name": "Dam Failure", "slug": "dam-failure", "priority_score": 0}],
            location,
            {},
        )[0]
        self.assertEqual(result.hazard_id, "dam-failure")
        self.assertEqual(result.scope, "address_level")
        self.assertEqual(result.data_status, "checked")
        self.assertTrue(result.is_in_hazard_zone)
        self.assertEqual(result.match_type, "inside")
        self.assertEqual(result.matched_layers[0]["name"], "Chabot · MainDam · Scenario1 · Sunny Day")
        self.assertEqual(result.hazard_exposure, "mapped_match")


class DamInundationRegistryTests(unittest.TestCase):
    def test_official_dataset_is_registered_with_required_metadata(self):
        registered = DatasetRegistry().get("dwr_dsod_dam_inundation_remote")
        self.assertIsNotNone(registered)
        self.assertEqual(registered.claim_type, "scenario")
        self.assertEqual(registered.source_crs, "EPSG:3310")
        self.assertEqual(registered.record_count, 1202)
        self.assertIn("/FeatureServer/100", registered.exact_service_or_download_url)


if __name__ == "__main__":
    unittest.main()
