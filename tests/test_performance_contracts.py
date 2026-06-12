import unittest
import json
from pathlib import Path
from unittest.mock import patch

from geospatial.adapters.local_geojson import LocalGeoJSONAdapter, _load_validated_layer
from geospatial.registry import get_default_registry
from geospatial.models import GeoPoint
from hazard_engine import (
    _collect_location_checks,
    build_hazard_results,
    clear_location_check_cache,
)
from app import app, load_geojson_file
from pydantic_models import LocationResult
from testing_utils import set_test_resident_state


BASE_DIR = Path(__file__).resolve().parents[1]


class StaticLayerCacheTests(unittest.TestCase):
    def setUp(self):
        _load_validated_layer.cache_clear()

    def test_cached_static_layer_returns_identical_evidence(self):
        dataset = get_default_registry().get("calfire_fhsz_local")
        adapter = LocalGeoJSONAdapter(BASE_DIR)
        point = GeoPoint(lat=37.79, lon=-122.19)
        first = adapter.check_point(dataset, point).model_dump(mode="json")
        second = adapter.check_point(dataset, point).model_dump(mode="json")
        first.pop("checked_at")
        second.pop("checked_at")
        first["validation"].pop("checked_at")
        second["validation"].pop("checked_at")
        self.assertEqual(first, second)
        self.assertGreaterEqual(_load_validated_layer.cache_info().hits, 1)

    def test_public_geojson_loader_reuses_static_parse(self):
        load_geojson_file.cache_clear()
        first = load_geojson_file("countbound.geojson")
        second = load_geojson_file("countbound.geojson")
        self.assertEqual(first, second)
        self.assertGreaterEqual(load_geojson_file.cache_info().hits, 1)


class LocationCheckCacheTests(unittest.TestCase):
    def setUp(self):
        clear_location_check_cache()

    @patch("hazard_engine.check_cgs_layers")
    @patch("hazard_engine.check_fault_layer")
    @patch("hazard_engine.check_wildfire_layer")
    @patch("hazard_engine.check_flood_layer")
    def test_duplicate_location_checks_are_reused(
        self,
        flood,
        wildfire,
        fault,
        cgs,
    ):
        flood.return_value = {
            "checked": True,
            "data_status": "not_in_layer",
            "inside": False,
            "layers": [],
        }
        wildfire.return_value = {
            "checked": True,
            "data_status": "not_in_layer",
            "inside": False,
            "layers": [],
        }
        fault.return_value = {
            "checked": True,
            "data_status": "checked",
            "near": False,
            "layers": [],
            "nearest_layer": {},
        }
        cgs.return_value = []

        first = _collect_location_checks(37.790001, -122.190001)
        second = _collect_location_checks(37.790002, -122.190002)
        self.assertEqual(first, second)
        self.assertEqual(flood.call_count, 1)
        self.assertEqual(wildfire.call_count, 1)
        self.assertEqual(fault.call_count, 1)
        self.assertEqual(cgs.call_count, 2)

    @patch("hazard_engine.check_cgs_layers", return_value=[])
    @patch("hazard_engine.check_fault_layer")
    @patch("hazard_engine.check_wildfire_layer")
    @patch("hazard_engine.check_flood_layer")
    def test_cached_checks_do_not_change_hazard_output(
        self,
        flood,
        wildfire,
        fault,
        _cgs,
    ):
        flood.return_value = {
            "checked": True,
            "data_status": "not_in_layer",
            "inside": False,
            "layers": [],
        }
        wildfire.return_value = {
            "checked": True,
            "data_status": "not_in_layer",
            "inside": False,
            "layers": [],
        }
        fault.return_value = {
            "checked": True,
            "data_status": "checked",
            "near": False,
            "layers": [],
            "nearest_layer": {},
        }
        location = LocationResult(
            input_address="Test address",
            formatted_address="Test address, Oakland, California",
            lat=37.79,
            lon=-122.19,
            city="Oakland",
            county="Alameda County",
            zip_code="94619",
        )
        hazards = [
            {"name": "Flood", "slug": "flood", "priority_score": 5},
            {"name": "Wildfire", "slug": "wildfire", "priority_score": 5},
            {"name": "Earthquake", "slug": "earthquake", "priority_score": 5},
        ]
        first = [
            result.model_dump(mode="json")
            for result in build_hazard_results(hazards, location, {})
        ]
        second = [
            result.model_dump(mode="json")
            for result in build_hazard_results(hazards, location, {})
        ]
        self.assertEqual(first, second)


class MapPerformanceTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    @patch("app.get_all_hazards")
    def test_map_keeps_layer_controls_without_running_hazard_checks(self, hazards):
        html = self.client.get("/map").get_data(as_text=True)
        for label in (
            "Wildfire Zones",
            "Flood Zones",
            "Fault Lines",
            "CGS Alquist-Priolo Fault Zones",
            "CGS Liquefaction Zones",
            "CGS Earthquake-Induced Landslide Zones",
            "CGS Tsunami Hazard Areas",
        ):
            self.assertIn(label, html)
        self.assertIn("toggleRiskFocus(highestRisk, false)", html)
        self.assertIn("Checking official mapped data", html)
        hazards.assert_not_called()

    def test_wildfire_map_response_is_filtered(self):
        set_test_resident_state(self.client, {
            "zip_code": "94619",
            "lat": 37.79,
            "lon": -122.19,
        })
        response = self.client.get(
            "/api/wildfire-zones?zip=94619&lat=37.79&lon=-122.19"
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["filtered"])
        self.assertEqual(payload["feature_count"], len(payload["features"]))
        self.assertLess(len(json.dumps(payload)), 9_000_000)

    @patch("app.load_geojson_file")
    def test_flood_map_response_uses_processed_zip_slice_only(self, raw_loader):
        set_test_resident_state(self.client, {
            "zip_code": "94501",
            "lat": 37.754029,
            "lon": -122.24918,
            "location_mode": "address",
        })
        response = self.client.get(
            "/api/flood-zones?zip=94501&lat=37.754029&lon=-122.24918"
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["filtered"])
        self.assertLessEqual(payload["feature_count"], 120)
        self.assertEqual(payload["address_match_status"], "not_checked")
        self.assertLess(len(response.data), 3_000_000)
        raw_loader.assert_not_called()

    def test_missing_flood_zip_slice_returns_safe_empty_collection(self):
        response = self.client.get("/api/flood-zones?zip=99999")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["features"], [])
        self.assertEqual(payload["data_status"], "data_unavailable")
        self.assertIn("map is still usable", payload["message"])


if __name__ == "__main__":
    unittest.main()
