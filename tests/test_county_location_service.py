import unittest
from unittest.mock import patch

import app as stayready
from county_location_service import (
    LocationCandidate,
    UnsupportedLocationInput,
    resolve_jurisdiction,
    suggest_locations,
)
from pydantic_models import ResolvedLocation
from location_service import location_from_session


def county_candidate():
    return LocationCandidate(
        candidate_id="opaque-candidate-123456",
        display_label="1221 Oak St, Oakland, CA, 94612",
        latitude=37.7993,
        longitude=-122.2634,
        zip_code="94612",
        provider="alameda_county_address_points",
        match_type="address_point",
        match_score=100,
        source_object_id="53131",
    )


def county_resolution():
    return ResolvedLocation(
        match_id="opaque-candidate-123456",
        normalized_display_address="1221 Oak St, Oakland, CA, 94612",
        latitude=37.7993,
        longitude=-122.2634,
        zip_code="94612",
        incorporated_jurisdiction="Oakland",
        match_type="address_point",
        match_score=100,
        precision="address_point",
        provider="alameda_county_address_points",
        boundary_version="city:2023-07-03;unincorporated:2024-05-01",
    )


class CountyLocationContractTests(unittest.TestCase):
    def test_candidate_public_payload_never_contains_coordinates(self):
        payload = county_candidate().public_payload()
        self.assertNotIn("latitude", payload)
        self.assertNotIn("longitude", payload)
        self.assertNotIn("coordinates", payload)
        self.assertEqual(payload["candidate_id"], "opaque-candidate-123456")

    def test_resolved_public_payload_never_contains_coordinates(self):
        payload = county_resolution().public_payload()
        self.assertNotIn("latitude", payload)
        self.assertNotIn("longitude", payload)
        self.assertEqual(payload["jurisdiction"], "Oakland")

    def test_po_box_requires_zip_or_community_fallback(self):
        with self.assertRaisesRegex(UnsupportedLocationInput, "PO boxes"):
            suggest_locations("PO Box 123, Oakland, CA")

    def test_map_pin_is_a_usable_point_with_an_explicit_limitation(self):
        result = location_from_session(
            {
                "address": "Resident-confirmed map pin",
                "zip_code": "94541",
                "lat": 37.67,
                "lon": -122.11,
                "location_mode": "map_pin",
                "city": "Hayward Acres",
            }
        )
        self.assertEqual(result.geocode_confidence, "mixed_support")
        self.assertIn("not a verified street-address match", result.limitations[0])

    @patch(
        "county_location_service._layer_version",
        side_effect=["2023-07-03", "2024-05-01"],
    )
    @patch(
        "county_location_service._point_boundary",
        side_effect=[[{"NAME": "Alameda County"}], [{"Area_name": "Castro Valley"}]],
    )
    def test_unincorporated_community_comes_from_official_polygon(
        self,
        _point_boundary,
        _layer_version,
    ):
        city, community, is_unincorporated, version = resolve_jurisdiction(37.69, -122.08)
        self.assertEqual(city, "")
        self.assertEqual(community, "Castro Valley")
        self.assertTrue(is_unincorporated)
        self.assertIn("city:2023-07-03", version)


class CountyLocationApiTests(unittest.TestCase):
    def setUp(self):
        stayready.app.config.update(TESTING=True)
        self.client = stayready.app.test_client()

    @patch("app.suggest_locations", return_value=[county_candidate()])
    def test_suggestions_are_coordinate_free(self, _suggest):
        response = self.client.get(
            "/api/location/suggestions?q=1221%20Oak%20St%2C%20Oakland%2C%20CA"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["candidates"]), 1)
        serialized = response.get_data(as_text=True)
        self.assertNotIn("37.7993", serialized)
        self.assertNotIn("-122.2634", serialized)

    @patch("app.resolve_candidate", return_value=county_resolution())
    def test_resolve_stores_private_state_and_returns_opaque_result(self, _resolve):
        stayready.save_location_candidate(county_candidate())
        response = self.client.post(
            "/api/location/resolve",
            json={"candidate_id": "opaque-candidate-123456"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["jurisdiction"], "Oakland")
        self.assertIn("result_token", payload)
        self.assertNotIn("latitude", payload)
        self.assertNotIn("longitude", payload)
        stored = stayready.RESIDENT_STATE_CACHE.get(payload["result_token"])
        self.assertEqual(stored["city"], "Oakland")
        self.assertEqual(stored["lat"], 37.7993)

    def test_expired_candidate_requires_new_confirmation(self):
        response = self.client.post(
            "/api/location/resolve",
            json={"candidate_id": "expired-candidate-123456"},
        )
        self.assertEqual(response.status_code, 410)
        self.assertIn("expired", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
