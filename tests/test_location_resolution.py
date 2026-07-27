import unittest
from unittest.mock import patch

import app
from county_location_service import LocationCandidate
from pydantic_models import ResolvedLocation


class FakeLocation:
    latitude = 37.7889314
    longitude = -122.1791837
    address = (
        "4183;4191, Observatory Avenue, Oakland Hills, Oakland, "
        "Alameda County, California, 94613, United States"
    )


class FakeGeocoder:
    def geocode(self, *args, **kwargs):
        return FakeLocation()


def resolved_observatory():
    return ResolvedLocation(
        match_id="county-match-observatory",
        normalized_display_address=(
            "4183 Observatory Avenue, Oakland Hills, Oakland, "
            "Alameda County, California, 94619, United States"
        ),
        latitude=37.7889314,
        longitude=-122.1791837,
        zip_code="94619",
        incorporated_jurisdiction="Oakland",
        match_type="address_point",
        match_score=100,
        precision="address_point",
        provider="alameda_county_address_points",
        boundary_version="city:test;unincorporated:test",
    )


class LocationResolutionTests(unittest.TestCase):
    def test_observatory_coordinates_resolve_to_local_boundary_zip(self):
        self.assertEqual(
            app.get_zip_from_coordinates(37.7889314, -122.1791837),
            "94619",
        )

    def test_canonical_address_replaces_range_and_geocoder_zip(self):
        display = app.canonicalize_geocoded_address(
            "4183, Observatory Avenue, Oakland, CA",
            FakeLocation.address,
            "94619",
        )
        self.assertEqual(
            display,
            (
                "4183 Observatory Avenue, Oakland Hills, Oakland, "
                "Alameda County, California, 94619, United States"
            ),
        )
        self.assertNotIn("4183;4191", display)
        self.assertNotIn("94613", display)

    @patch("app.resolve_candidate", return_value=resolved_observatory())
    @patch(
        "app.suggest_locations",
        return_value=[
            LocationCandidate(
                candidate_id="county-match-observatory",
                display_label="4183 Observatory Avenue, Oakland, CA, 94619",
                latitude=37.7889314,
                longitude=-122.1791837,
                zip_code="94619",
                provider="alameda_county_address_points",
                match_type="address_point",
                match_score=100,
            )
        ],
    )
    def test_geocode_address_returns_verified_display_address(self, _suggest, _resolve):
        result = app.geocode_address("4183, Observatory Avenue, Oakland, CA")
        self.assertEqual(result[2], "94619")
        self.assertEqual(
            result[3],
            (
                "4183 Observatory Avenue, Oakland Hills, Oakland, "
                "Alameda County, California, 94619, United States"
            ),
        )

    @patch(
        "app.resolve_address_input",
        return_value=resolved_observatory(),
    )
    def test_form_summary_uses_verified_zip_and_canonical_address(self, _resolve):
        app.app.config.update(TESTING=True)
        with app.app.test_client() as client:
            response = client.post(
                "/form",
                data={
                    "address": "4183, Observatory Avenue, Oakland, CA",
                    "household": "2",
                    "preparedness": "Basic supplies",
                },
                follow_redirects=True,
            )
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            (
                "4183 Observatory Avenue, Oakland Hills, Oakland, "
                "Alameda County, California, 94619, United States"
            ),
            html,
        )
        self.assertNotIn("4183;4191", html)
        self.assertNotIn("94613", html)


if __name__ == "__main__":
    unittest.main()
