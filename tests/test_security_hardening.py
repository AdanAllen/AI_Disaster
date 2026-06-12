import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from geospatial.adapters.arcgis_feature_service import ArcGISFeatureServiceAdapter
from geospatial.models import DatasetProvenance, GeoPoint
from security_utils import allowed_remote_url, reset_rate_limits
from testing_utils import set_test_resident_state


BASE_DIR = Path(__file__).resolve().parents[1]


class SecurityRouteTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()
        reset_rate_limits()

    def test_security_headers_are_present(self):
        response = self.client.get("/")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(
            response.headers["Referrer-Policy"],
            "strict-origin-when-cross-origin",
        )

    def test_invalid_hazard_slug_is_safe_404(self):
        response = self.client.get("/api/hazards/../../etc/passwd")
        self.assertIn(response.status_code, {404, 308})
        self.assertNotIn("Traceback", response.get_data(as_text=True))

        response = self.client.get("/api/hazards/not-a-real-hazard")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("Traceback", response.get_data(as_text=True))

    def test_oversized_address_is_rejected(self):
        response = self.client.post(
            "/search-address",
            data={"address": "1" * 500},
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_map_parameters_are_rejected(self):
        response = self.client.get("/api/flood-zones?zip=../../etc/passwd")
        self.assertEqual(response.status_code, 400)
        response = self.client.get("/api/cgs-map-layer/not-real?lat=37.8&lon=-122.2")
        self.assertEqual(response.status_code, 404)

    def test_expensive_endpoint_rate_limit(self):
        with patch("app.requests.get") as get:
            get.side_effect = RuntimeError("offline")
            for _ in range(20):
                response = self.client.get("/api/live-earthquakes?scope=bay_area&days=7")
                self.assertEqual(response.status_code, 200)
            response = self.client.get("/api/live-earthquakes?scope=bay_area&days=7")
        self.assertEqual(response.status_code, 429)
        self.assertIn("Retry-After", response.headers)
        self.assertNotIn("Traceback", response.get_data(as_text=True))

    def test_external_next_url_cannot_create_open_redirect(self):
        response = self.client.post(
            "/hazards/profile",
            data={"next_url": "https://evil.example/steal"},
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/hazards")

    def test_health_endpoints_do_not_expose_secrets(self):
        combined = (
            self.client.get("/api/health").get_data(as_text=True)
            + self.client.get("/api/supabase-health").get_data(as_text=True)
        )
        for secret_name in (
            "SUPABASE_ANON_KEY",
            "SUPABASE_SECRET_KEY",
            "SUPABASE_SERVICE_ROLE",
            "FLASK_SECRET_KEY",
        ):
            self.assertNotIn(secret_name, combined)
        self.assertNotIn(".supabase.co", combined)

    def test_templates_escape_user_controlled_address(self):
        set_test_resident_state(self.client, {
            "zip_code": "94601",
            "address": "<script>alert(1)</script>",
            "location_mode": "address",
            "lat": 37.8,
            "lon": -122.2,
        })
        response = self.client.get("/map")
        html = response.get_data(as_text=True)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("\\u003cscript\\u003ealert(1)\\u003c/script\\u003e", html)

    def test_no_service_role_or_real_env_is_tracked(self):
        env_example = (BASE_DIR / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("service_role", env_example.lower())
        self.assertNotIn("SUPABASE_SERVICE_ROLE", env_example)
        self.assertIn("SUPABASE_SECRET_KEY", env_example)
        self.assertFalse((BASE_DIR / ".env").is_file() and os.getenv("CI") == "true")


class SSRFTests(unittest.TestCase):
    def test_remote_host_allowlist(self):
        self.assertTrue(
            allowed_remote_url(
                "https://services2.arcgis.com/example/FeatureServer/0"
            )
        )
        self.assertFalse(allowed_remote_url("http://services2.arcgis.com/example"))
        self.assertFalse(allowed_remote_url("https://127.0.0.1/internal"))
        self.assertFalse(allowed_remote_url("https://evil.example/FeatureServer/0"))

    def test_adapter_does_not_query_user_controlled_host(self):
        dataset = DatasetProvenance(
            dataset_id="blocked",
            dataset_version="test",
            hazard_type="flood",
            agency="Test agency",
            authoritative_landing_url="https://example.gov",
            exact_service_or_download_url="https://evil.example/FeatureServer/0",
            dataset_name="Blocked service",
            claim_type="regulatory_zone",
            source_type="remote_service",
            license_terms_notes="Test only.",
            coverage_area="Test only.",
            intended_claim="Test only.",
            prohibited_claims=["No safety claim."],
            status="provisional",
        )
        with patch("geospatial.adapters.arcgis_feature_service.requests.get") as get:
            result = ArcGISFeatureServiceAdapter().check_point(
                dataset,
                GeoPoint(lat=37.8, lon=-122.2),
            )
        get.assert_not_called()
        self.assertEqual(result.evidence_status, "data_unavailable")


if __name__ == "__main__":
    unittest.main()
