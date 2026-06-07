import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app import app
from geospatial.models import DatasetProvenance
from geospatial.adapters.arcgis_feature_service import ArcGISFeatureServiceAdapter
from geospatial.adapters.arcgis_feature_service import _remote_record_count
from geospatial.models import GeoPoint
from hazard_engine import _cgs_public_evidence, build_hazard_results, merge_structured_result
from pydantic_models import LocationResult


BASE_DIR = Path(__file__).resolve().parents[1]


def remote_dataset(dataset_id="cgs_liquefaction_remote"):
    return DatasetProvenance(
        dataset_id=dataset_id,
        dataset_version="test-v1",
        hazard_type="tsunami" if "tsunami" in dataset_id else "earthquake",
        agency="California Geological Survey",
        authoritative_landing_url="https://www.conservation.ca.gov/cgs",
        exact_service_or_download_url="https://services2.arcgis.com/example/FeatureServer/0",
        dataset_name="Test CGS layer",
        claim_type="hazard_zone" if "tsunami" in dataset_id else "regulatory_zone",
        source_type="remote_service",
        license_terms_notes="Official test fixture.",
        coverage_area="Test coverage.",
        intended_claim="Official mapped evidence test.",
        prohibited_claims=[
            "A non-match does not establish safety.",
            "This is not a site-specific determination.",
        ],
        status="provisional",
        result_label="Inside a CGS mapped test zone.",
        source_summary="Official CGS mapped-zone evidence.",
    )


def response(features):
    mocked = Mock()
    mocked.raise_for_status.return_value = None
    mocked.json.return_value = {"features": features}
    return mocked


def count_response(count=1):
    mocked = Mock()
    mocked.raise_for_status.return_value = None
    mocked.json.return_value = {"count": count}
    return mocked


class CGSEvidenceTests(unittest.TestCase):
    def setUp(self):
        _remote_record_count.cache_clear()

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_mapped_match_returns_official_mapped_data(self, get):
        get.side_effect = [
            count_response(),
            response([{"attributes": {"ZONE": "Yes"}}]),
        ]
        evidence = ArcGISFeatureServiceAdapter().check_point(
            remote_dataset(), GeoPoint(lat=37.8, lon=-122.2)
        )
        public = _cgs_public_evidence("cgs_liquefaction_remote", evidence)
        self.assertTrue(public["checked"])
        self.assertTrue(public["matched"])
        self.assertEqual(public["exposure"], "mapped_match")
        self.assertEqual(public["evidence_tier"], "official_mapped_data")
        self.assertTrue(public["source_summary"])
        self.assertTrue(public["limitations"])

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_checked_non_match_is_unknown_not_low_risk(self, get):
        get.side_effect = [count_response(), response([])]
        evidence = ArcGISFeatureServiceAdapter().check_point(
            remote_dataset(), GeoPoint(lat=37.8, lon=-122.2)
        )
        public = _cgs_public_evidence("cgs_liquefaction_remote", evidence)
        self.assertFalse(public["matched"])
        self.assertEqual(public["exposure"], "no_mapped_match")
        self.assertEqual(public["priority_band"], "Not an address-level priority ranking")
        self.assertIsNone(public["ranking_score"])
        self.assertEqual(public["status_label"], "No mapped match found")
        self.assertIn("no mapped match found", public["result_label"].lower())
        self.assertNotIn("low risk", public["result_label"].lower())
        self.assertNotIn("low exposure", public["result_label"].lower())

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_missing_or_corrupt_remote_response_is_unavailable(self, get):
        get.side_effect = ValueError("invalid JSON")
        evidence = ArcGISFeatureServiceAdapter().check_point(
            remote_dataset(), GeoPoint(lat=37.8, lon=-122.2)
        )
        self.assertEqual(evidence.evidence_status, "data_unavailable")
        self.assertIsNone(evidence.matched)

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_empty_remote_layer_is_unavailable(self, get):
        get.return_value = count_response(0)
        evidence = ArcGISFeatureServiceAdapter().check_point(
            remote_dataset(), GeoPoint(lat=37.8, lon=-122.2)
        )
        self.assertEqual(evidence.evidence_status, "data_unavailable")
        self.assertIsNone(evidence.matched)

    @patch("geospatial.adapters.arcgis_feature_service.requests.get")
    def test_tsunami_outside_polygon_is_not_a_match(self, get):
        dataset = remote_dataset("cgs_tsunami_hazard_area_remote")
        dataset.match_field = "Evacuate"
        dataset.match_values = ["Yes, Tsunami Hazard Area"]
        get.side_effect = [
            count_response(),
            response([{"attributes": {"Evacuate": "No, Outside Hazard Area"}}]),
        ]
        evidence = ArcGISFeatureServiceAdapter().check_point(
            dataset, GeoPoint(lat=37.8, lon=-122.2)
        )
        self.assertFalse(evidence.matched)

    def test_registered_cgs_sources_remain_provisional(self):
        datasets = json.loads(
            (BASE_DIR / "data" / "geospatial" / "datasets.json").read_text(encoding="utf-8")
        )
        cgs = [item for item in datasets if item["dataset_id"].startswith("cgs_")]
        self.assertEqual(len(cgs), 4)
        self.assertTrue(all(item["status"] == "provisional" for item in cgs))

    def test_layer_language_is_scoped(self):
        registry = {
            item["dataset_id"]: item
            for item in json.loads(
                (BASE_DIR / "data" / "geospatial" / "datasets.json").read_text(encoding="utf-8")
            )
        }
        self.assertIn("surface fault rupture", registry["cgs_alquist_priolo_remote"]["intended_claim"])
        self.assertIn("liquefaction", registry["cgs_liquefaction_remote"]["intended_claim"])
        self.assertIn(
            "earthquake-induced landslide",
            registry["cgs_earthquake_landslide_remote"]["intended_claim"],
        )
        self.assertIn(
            "evacuation and response planning",
            registry["cgs_tsunami_hazard_area_remote"]["intended_claim"],
        )

    @patch("hazard_engine.check_cgs_layers")
    def test_tsunami_seiche_receives_tsunami_evidence(self, check_cgs):
        def checks(_lat, _lon, dataset_ids=None):
            if dataset_ids == ["cgs_tsunami_hazard_area_remote"]:
                return [{
                    "dataset_id": "cgs_tsunami_hazard_area_remote",
                    "map_layer_key": "tsunami",
                    "dataset_name": "California Tsunami Hazard Areas",
                    "checked": True,
                    "data_available": True,
                    "matched": True,
                    "exposure": "mapped_match",
                    "result_label": "Inside a CGS mapped tsunami hazard area.",
                    "priority_band": "Mapped evidence",
                    "ranking_score": None,
                    "evidence_tier": "official_mapped_data",
                    "claim_type": "hazard_zone",
                    "precision": "address_point",
                    "source_summary": "CGS tsunami hazard areas support evacuation and response planning.",
                    "source_id": "cgs_tsunami_hazard_area",
                    "source_agency": "California Geological Survey",
                    "source_url": "https://www.conservation.ca.gov/cgs/tsunami/maps",
                    "effective_date": None,
                    "checked_at": "2026-06-06T12:00:00Z",
                    "public_claim_status": "official_provisional",
                    "matched_features": [{"name": "Inside a CGS mapped tsunami hazard area."}],
                    "limitations": [
                        "This result is for evacuation and response planning.",
                        "This is not a site-specific legal or property determination.",
                    ],
                }]
            return []

        check_cgs.side_effect = checks
        location = LocationResult(
            input_address="2301 Shore Line Drive",
            formatted_address="2301 Shore Line Drive, Alameda, California",
            lat=37.754029,
            lon=-122.24918,
            city="Alameda",
            county="Alameda County",
            zip_code="94501",
        )
        result = build_hazard_results(
            [{"name": "Tsunami/Seiche", "slug": "tsunami-seiche"}],
            location,
            {},
        )[0]
        self.assertEqual(result.hazard_id, "tsunami-seiche")
        self.assertEqual(len(result.additional_geospatial_evidence), 1)
        self.assertEqual(
            result.additional_geospatial_evidence[0]["result_label"],
            "Inside a CGS mapped tsunami hazard area.",
        )


class CGSPublicRenderingTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        with self.client.session_transaction() as session:
            session.update(
                zip_code="94501",
                lat=37.754029,
                lon=-122.24918,
                address="2301 Shore Line Drive, Alameda, Alameda County, California, 94501, United States",
                input_address="2301 Shore Line Drive, Alameda, CA 94501",
                county="Alameda County",
                city="Alameda",
                location_mode="address",
                household="adults",
                preparedness="starting",
                special_needs="",
                household_tags=[],
            )

    def test_summary_and_hazard_pages_render_cgs_evidence(self):
        summary = self.client.get("/risk_summary").get_data(as_text=True)
        earthquake = self.client.get("/hazards/earthquake").get_data(as_text=True)
        tsunami = self.client.get("/hazards/tsunami-seiche").get_data(as_text=True)

        self.assertIn("Inside a CGS mapped liquefaction zone.", summary)
        self.assertIn("Inside a CGS mapped tsunami hazard area.", summary)
        for expected in (
            "CGS Alquist-Priolo Earthquake Fault Zones",
            "CGS Liquefaction Zones",
            "CGS Earthquake-Induced Landslide Zones",
        ):
            self.assertIn(expected, earthquake)
        self.assertIn("Inside a CGS mapped tsunami hazard area.", tsunami)
        for page in (summary, earthquake, tsunami):
            self.assertIn("California Geological Survey", page)
            self.assertIn("Official Provisional", page)
            self.assertIn("This official remote dataset is provisional", page)

    def test_map_has_all_cgs_controls_and_provenance_legends(self):
        page = self.client.get("/map").get_data(as_text=True)
        for layer_key in (
            "alquist-priolo",
            "liquefaction",
            "earthquake-landslide",
            "tsunami",
        ):
            self.assertIn(f'id="cgs-{layer_key}-toggle"', page)
            self.assertIn(f'data-cgs-legend="{layer_key}"', page)
        self.assertIn("California Geological Survey · Official Provisional", page)
        self.assertIn("Open official CGS source", page)
        self.assertIn("California Geological Survey. It is not officially endorsed", page)
        self.assertIn("source_summary", page)
        self.assertIn("Official CGS map layer unavailable — not displayed.", page)

    @patch("app.GeospatialEvidenceService.map_geojson")
    def test_cgs_map_api_returns_polygon_metadata(self, map_geojson):
        map_geojson.return_value = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"ZONE": "test"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.26, 37.74],
                        [-122.24, 37.74],
                        [-122.24, 37.76],
                        [-122.26, 37.76],
                        [-122.26, 37.74],
                    ]],
                },
            }],
        }
        response = self.client.get(
            "/api/cgs-map-layer/liquefaction?lat=37.754029&lon=-122.24918"
        )
        payload = response.get_json()
        self.assertEqual(payload["data_status"], "checked")
        self.assertEqual(payload["feature_count"], 1)
        self.assertEqual(payload["source"], "California Geological Survey")
        self.assertEqual(payload["public_claim_status"], "official_provisional")
        self.assertTrue(payload["source_summary"])
        self.assertTrue(payload["limitations"])

    @patch("app.GeospatialEvidenceService.map_geojson")
    def test_cgs_map_api_unavailable_state_is_non_reassuring(self, map_geojson):
        map_geojson.side_effect = ValueError("service unavailable")
        payload = self.client.get(
            "/api/cgs-map-layer/liquefaction?lat=37.754029&lon=-122.24918"
        ).get_json()
        self.assertEqual(payload["data_status"], "data_unavailable")
        self.assertEqual(
            payload["message"],
            "Official CGS map layer unavailable — not displayed.",
        )
        self.assertNotIn("low risk", payload["message"].lower())
        self.assertNotIn("no risk", payload["message"].lower())


if __name__ == "__main__":
    unittest.main()
