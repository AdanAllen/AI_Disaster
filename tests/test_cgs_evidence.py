import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from geospatial.models import DatasetProvenance
from geospatial.adapters.arcgis_feature_service import ArcGISFeatureServiceAdapter
from geospatial.adapters.arcgis_feature_service import _remote_record_count
from geospatial.models import GeoPoint
from hazard_engine import _cgs_public_evidence


BASE_DIR = Path(__file__).resolve().parents[1]


def remote_dataset(dataset_id="cgs_liquefaction_remote"):
    return DatasetProvenance(
        dataset_id=dataset_id,
        dataset_version="test-v1",
        hazard_type="tsunami" if "tsunami" in dataset_id else "earthquake",
        agency="California Geological Survey",
        authoritative_landing_url="https://www.conservation.ca.gov/cgs",
        exact_service_or_download_url="https://example.gov/FeatureServer/0",
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
        self.assertEqual(public["priority_band"], "Unknown")
        self.assertIsNone(public["ranking_score"])
        self.assertIn("does not mean the location is safe", public["result_label"].lower())
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


if __name__ == "__main__":
    unittest.main()
