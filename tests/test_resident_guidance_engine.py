import unittest

from location_service import location_from_session
from resident_guidance_engine import build_resident_plan, get_household_context, load_local_hazard_facts


def location_context(city, address, lat, lon, zip_code):
    return {
        "city": city,
        "county": "Alameda County",
        "address": address,
        "display_name": address,
        "zip_code": zip_code,
        "precision_label": "Address point",
        "gis_status": "Address point ready for GIS checks",
        "location_result": {
            "formatted_address": address,
            "neighborhood": "",
            "lat": lat,
            "lon": lon,
        },
    }


def hazard(slug, why="Official address check completed."):
    return {
        "slug": slug,
        "name": slug.title(),
        "scope": "address_level",
        "scope_label": "Address-level",
        "data_status": "checked",
        "data_status_label": "Checked",
        "exposure_level": "High",
        "why_shown": why,
        "recommended_actions": [],
        "limitations": [],
        "sources": [],
        "specialized_guidance": {
            "resident_guidance": {},
            "city_context": [],
            "location_specific_context": [],
            "guidance_source_status": "local_reviewed",
            "recovery_needs": [],
        },
    }


class ResidentGuidanceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_local_hazard_facts.cache_clear()

    def test_medications_do_not_infer_pets(self):
        context = get_household_context({
            "special_needs": "Uses medications and backup power for a medical device.",
            "household_tags": [],
        })
        self.assertIn("medical", context["tags"])
        self.assertNotIn("pets", context["tags"])

    def test_hayward_near_fault_gets_area_based_context(self):
        plan = build_resident_plan(
            location_context(
                "Hayward",
                "777 Mission Boulevard, Hayward, California, 94541",
                37.672,
                -122.083,
                "94541",
            ),
            [hazard("earthquake")],
        )
        result = plan["hazards"][0]
        self.assertEqual(result["evidence_tier"], "area_based")
        self.assertIn("Hayward Fault corridor", result["location_matches"][0]["label"])
        self.assertIn("Hayward's Local Resilience Plan", result["why_it_matters"])

    def test_alameda_shoreline_match_keeps_address_evidence(self):
        plan = build_resident_plan(
            location_context(
                "Alameda",
                "Example address, Bay Farm Island, Alameda, California, 94501",
                37.768,
                -122.276,
                "94501",
            ),
            [hazard("flood", "The address matched an official FEMA flood polygon.")],
        )
        result = plan["hazards"][0]
        self.assertIn("Address point checked", result["evidence_badges"])
        self.assertIn("Area-based source", result["evidence_badges"])
        self.assertTrue(result["why_it_matters"].startswith("The address matched an official FEMA"))

    def test_oakland_hills_match_is_not_parcel_claim(self):
        plan = build_resident_plan(
            location_context(
                "Oakland",
                "Example address, Oakland Hills, Oakland, California, 94619",
                37.81,
                -122.18,
                "94619",
            ),
            [hazard("wildfire")],
        )
        result = plan["hazards"][0]
        self.assertEqual(result["evidence_tier"], "area_based")
        self.assertTrue(any("not a parcel" in item.lower() for item in result["limitations"]))

    def test_unknown_area_falls_back_without_area_match(self):
        plan = build_resident_plan(
            location_context(
                "Fremont",
                "Central Fremont address, Fremont, California, 94538",
                37.548,
                -121.988,
                "94538",
            ),
            [hazard("flood")],
        )
        result = plan["hazards"][0]
        self.assertFalse(result["location_matches"])
        self.assertEqual(result["evidence_tier"], "citywide")

    def test_fremont_hills_and_baylands_produce_different_context(self):
        hills = build_resident_plan(
            location_context(
                "Fremont",
                "Example address, Mission Peak, Fremont, California, 94539",
                37.53,
                -121.91,
                "94539",
            ),
            [hazard("wildfire")],
        )["hazards"][0]
        baylands = build_resident_plan(
            location_context(
                "Fremont",
                "Example address, Fremont Baylands, Fremont, California, 94538",
                37.52,
                -122.04,
                "94538",
            ),
            [hazard("flood")],
        )["hazards"][0]
        self.assertIn("eastern hills", hills["location_matches"][0]["label"].lower())
        self.assertIn("baylands", baylands["location_matches"][0]["label"].lower())

    def test_oakland_hills_and_shoreline_produce_different_context(self):
        hills = build_resident_plan(
            location_context("Oakland", "Example address, Oakland Hills, Oakland, California, 94619", 37.81, -122.18, "94619"),
            [hazard("wildfire")],
        )["hazards"][0]
        shoreline = build_resident_plan(
            location_context("Oakland", "Example address, Jack London Square, Oakland, California, 94607", 37.78, -122.29, "94607"),
            [hazard("flood")],
        )["hazards"][0]
        self.assertIn("oakland hills", hills["location_matches"][0]["label"].lower())
        self.assertIn("shoreline", shoreline["location_matches"][0]["label"].lower())

    def test_berkeley_hills_and_west_berkeley_produce_different_context(self):
        hills = build_resident_plan(
            location_context("Berkeley", "Example address, Berkeley Hills, Berkeley, California, 94707", 37.89, -122.255, "94707"),
            [hazard("wildfire")],
        )["hazards"][0]
        west = build_resident_plan(
            location_context("Berkeley", "West Berkeley, Berkeley, California, 94710", 37.86, -122.30, "94710"),
            [hazard("earthquake")],
        )["hazards"][0]
        self.assertIn("fire zones", hills["location_matches"][0]["label"].lower())
        self.assertIn("west berkeley", west["location_matches"][0]["label"].lower())

    def test_area_match_contains_page_citation(self):
        plan = build_resident_plan(
            location_context("Alameda", "Example address, Bay Farm Island, Alameda, California, 94501", 37.77, -122.28, "94501"),
            [hazard("flood")],
        )
        match = plan["hazards"][0]["location_matches"][0]
        self.assertTrue(match["source_document"])
        self.assertTrue(match["source_page"])

    def test_household_tags_change_actions_not_evidence(self):
        base_location = location_context(
            "Hayward",
            "777 Mission Boulevard, Hayward, California, 94541",
            37.672,
            -122.083,
            "94541",
        )
        plan = build_resident_plan(
            base_location,
            [hazard("earthquake")],
            session_data={"household_tags": ["pets", "no_car", "renter"]},
        )
        self.assertEqual(plan["hazards"][0]["evidence_tier"], "area_based")
        labels = {item["label"] for item in plan["household_priorities"]}
        self.assertEqual(labels, {"Pets", "No reliable car access", "Renter recovery"})

    def test_city_detection_does_not_confuse_alameda_county_for_alameda_city(self):
        result = location_from_session({
            "address": "123 Example Street, Castro Valley, Alameda County, California, 94546",
            "zip_code": "94546",
            "lat": 37.69,
            "lon": -122.09,
            "location_mode": "address",
        })
        self.assertEqual(result.city, "Castro Valley")


if __name__ == "__main__":
    unittest.main()
