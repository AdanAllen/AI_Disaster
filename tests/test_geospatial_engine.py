import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from geospatial.models import DatasetProvenance
from geospatial.registry import DatasetRegistry, DatasetRegistryError
from geospatial.service import GeospatialEvidenceService


def feature_collection():
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "FLD_AR_ID": "test-zone",
                    "FLD_ZONE": "AE",
                    "SFHA_TF": "T",
                },
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
            }
        ],
    }


class GeospatialEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "static").mkdir()
        (self.root / "data").mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_registry(self, *, payload=None, status="provisional", coverage_bbox=None):
        raw = json.dumps(payload or feature_collection()).encode()
        layer_path = self.root / "static" / "test.geojson"
        layer_path.write_bytes(raw)
        dataset = {
            "dataset_id": "test_flood",
            "dataset_version": "test-v1",
            "hazard_type": "flood",
            "agency": "Test Government Agency",
            "authoritative_landing_url": "https://example.gov/flood",
            "exact_service_or_download_url": "",
            "dataset_name": "Test flood polygons",
            "claim_type": "regulatory_zone",
            "source_type": "local_snapshot",
            "license_terms_notes": "Test fixture only.",
            "coverage_area": "Test fixture bounds.",
            "coverage_bbox": coverage_bbox,
            "intended_claim": "Point-in-polygon test.",
            "prohibited_claims": ["A non-match is not a safety determination."],
            "local_path": "static/test.geojson",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "record_count": 1,
            "source_crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
            "converted_crs": "urn:ogc:def:crs:OGC:1.3:CRS84",
            "status": status,
        }
        registry_path = self.root / "data" / "datasets.json"
        registry_path.write_text(json.dumps([dataset]), encoding="utf-8")
        return registry_path, layer_path

    def service(self, registry_path):
        return GeospatialEvidenceService(
            project_root=self.root,
            registry=DatasetRegistry(registry_path),
        )

    def test_successful_point_in_polygon_match_is_provisional(self):
        registry_path, _ = self.write_registry()
        evidence = self.service(registry_path).check_point(
            "test_flood", 37.80, -122.20
        )
        self.assertEqual(evidence.evidence_status, "checked")
        self.assertTrue(evidence.matched)
        self.assertEqual(evidence.public_claim_status, "official_provisional")
        self.assertTrue(evidence.validation.checksum_matches)

    def test_successful_non_match_is_not_safe_claim(self):
        registry_path, _ = self.write_registry()
        evidence = self.service(registry_path).check_point(
            "test_flood", 38.20, -122.20
        )
        self.assertEqual(evidence.evidence_status, "checked")
        self.assertFalse(evidence.matched)
        self.assertIn("not a safety determination", " ".join(evidence.limitations))

    def test_corrupt_file_after_registration_is_unavailable(self):
        registry_path, layer_path = self.write_registry()
        layer_path.write_text("{broken", encoding="utf-8")
        evidence = self.service(registry_path).check_point(
            "test_flood", 37.80, -122.20
        )
        self.assertEqual(evidence.evidence_status, "data_unavailable")
        self.assertIsNone(evidence.matched)
        self.assertEqual(evidence.validation.status, "invalid")

    def test_missing_provenance_is_rejected(self):
        with self.assertRaises(ValidationError):
            DatasetProvenance(
                dataset_id="missing",
                dataset_version="v1",
                hazard_type="flood",
                agency="Agency",
                authoritative_landing_url="https://example.gov",
                dataset_name="Missing checksum",
                claim_type="regulatory_zone",
                source_type="local_snapshot",
                license_terms_notes="Unknown.",
                coverage_area="Unknown.",
                intended_claim="Test.",
                prohibited_claims=["No safety claim."],
                local_path="static/missing.geojson",
                record_count=1,
            )

    def test_point_outside_reviewed_coverage_is_not_covered(self):
        bounds = {
            "min_lat": 37.70,
            "max_lat": 37.90,
            "min_lon": -122.30,
            "max_lon": -122.10,
        }
        registry_path, _ = self.write_registry(coverage_bbox=bounds)
        evidence = self.service(registry_path).check_point(
            "test_flood", 38.20, -122.20
        )
        self.assertEqual(evidence.evidence_status, "not_covered")
        self.assertIsNone(evidence.matched)
        self.assertEqual(evidence.public_claim_status, "not_evaluated")

    def test_verified_status_requires_human_approval(self):
        registry_path, _ = self.write_registry(status="verified")
        with self.assertRaises(DatasetRegistryError):
            DatasetRegistry(registry_path)


if __name__ == "__main__":
    unittest.main()
