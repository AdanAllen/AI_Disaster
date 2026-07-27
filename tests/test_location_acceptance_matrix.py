import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LocationAcceptanceMatrixTests(unittest.TestCase):
    def test_matrix_covers_all_cities_and_named_unincorporated_communities(self):
        fixtures = json.loads(
            (ROOT / "data" / "location_acceptance_fixtures.json").read_text()
        )
        self.assertEqual(len(fixtures), 21)
        self.assertEqual(
            {item["jurisdiction"] for item in fixtures if not item["community"]},
            {
                "Alameda",
                "Albany",
                "Berkeley",
                "Dublin",
                "Emeryville",
                "Fremont",
                "Hayward",
                "Livermore",
                "Newark",
                "Oakland",
                "Piedmont",
                "Pleasanton",
                "San Leandro",
                "Union City",
            },
        )
        self.assertEqual(
            {item["community"] for item in fixtures if item["community"]},
            {
                "Ashland",
                "Castro Valley",
                "Cherryland",
                "Fairview",
                "Hayward Acres",
                "San Lorenzo",
                "Sunol",
            },
        )
        self.assertTrue(all(item["fixture_type"] for item in fixtures))


if __name__ == "__main__":
    unittest.main()
