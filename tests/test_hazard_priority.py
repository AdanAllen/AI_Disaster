import json
import os
import unittest

from shapely.geometry import shape

from hazard_priority import (
    build_hazard_priority_results,
    calculate_citywide_priority,
    combine_scenario_ratings,
    rank_hazards_for_risk_summary,
)


def hazard_result(slug, **overrides):
    base = {
        "slug": slug,
        "data_status": "not_in_layer",
        "is_in_hazard_zone": False,
        "additional_geospatial_evidence": [],
        "matched_layers": [],
        "priority_score": 10,
        "local_risk_score": 10,
    }
    base.update(overrides)
    return base


def cgs_evidence(dataset_id, layer_key, matched=True):
    return {
        "dataset_id": dataset_id,
        "map_layer_key": layer_key,
        "dataset_name": f"CGS {layer_key}",
        "checked": True,
        "data_available": True,
        "matched": matched,
        "result_label": f"Inside a CGS mapped {layer_key} zone." if matched else "No mapped match found.",
        "status_label": "Mapped match found" if matched else "No mapped match found",
        "checked_at": "2026-06-17T12:00:00Z",
    }


def coordinates_for_sub_area(name):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hazard_priority", "oakland_plan_areas.geojson")
    with open(path, "r", encoding="utf-8") as source:
        geojson = json.load(source)
    for feature in geojson["features"]:
        if feature["properties"].get("OAKLAND_PE", "").replace("/ ", "/") == name:
            point = shape(feature["geometry"]).representative_point()
            return {"lat": point.y, "lon": point.x}
    raise AssertionError(f"Sub-area not found: {name}")


class HazardPriorityRankingTests(unittest.TestCase):
    def test_probability_impact_matrix(self):
        self.assertEqual(calculate_citywide_priority("Low", "High"), "Medium")
        self.assertEqual(calculate_citywide_priority("Medium", "High"), "High")
        self.assertEqual(calculate_citywide_priority("High", "Low"), "Medium")

    def test_missing_probability_produces_unknown(self):
        ranked = build_hazard_priority_results(
            "Example",
            [],
            citywide_lhmp_data=[{
                "jurisdiction": "Example",
                "hazard": "flood",
                "probability": "Unknown",
                "impact": "High",
                "confidence": "Low",
            }],
        )
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(flood["calculated_citywide_priority"], "Unknown")

    def test_missing_impact_produces_unknown(self):
        ranked = build_hazard_priority_results(
            "Example",
            [],
            citywide_lhmp_data=[{
                "jurisdiction": "Example",
                "hazard": "wildfire",
                "probability": "High",
                "impact": "Unknown",
                "confidence": "Low",
            }],
        )
        wildfire = next(item for item in ranked if item["slug"] == "wildfire")
        self.assertEqual(wildfire["calculated_citywide_priority"], "Unknown")

    def test_legacy_numeric_values_do_not_affect_result(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [
                hazard_result("wildfire", priority_score=1, local_risk_score=1),
                hazard_result("flood", priority_score=10, local_risk_score=10),
            ],
        )
        wildfire = next(item for item in ranked if item["slug"] == "wildfire")
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(wildfire["calculated_citywide_priority"], "High")
        self.assertEqual(flood["calculated_citywide_priority"], "High")
        self.assertNotIn("10", wildfire["why_this_rating"])

    def test_official_lhmp_priority_is_preserved_when_simplified_method_differs(self):
        flood = next(item for item in rank_hazards_for_risk_summary("Oakland", []) if item["slug"] == "flood")
        self.assertEqual(flood["calculated_citywide_priority"], "High")
        self.assertEqual(flood["official_lhmp_priority"], "Medium")
        self.assertTrue(flood["official_priority_differs"])

    def test_earthquake_remains_high_outside_liquefaction(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("earthquake", additional_geospatial_evidence=[
                cgs_evidence("cgs_liquefaction_remote", "liquefaction", matched=False)
            ])],
        )
        earthquake = next(item for item in ranked if item["slug"] == "earthquake")
        self.assertEqual(earthquake["calculated_citywide_priority"], "High")
        self.assertEqual(earthquake["local_exposure_status"], "General citywide exposure")

    def test_liquefaction_intersection_adds_local_earthquake_concern(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("earthquake", additional_geospatial_evidence=[
                cgs_evidence("cgs_liquefaction_remote", "liquefaction", matched=True)
            ])],
        )
        earthquake = next(item for item in ranked if item["slug"] == "earthquake")
        self.assertEqual(earthquake["local_exposure_status"], "Additional mapped concern")
        self.assertEqual(earthquake["calculated_citywide_priority"], "High")

    def test_wildfire_fhsz_intersection_adds_direct_exposure(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("wildfire", data_status="checked", is_in_hazard_zone=True, matched_layers=[
                {"name": "CAL FIRE Fire Hazard Severity Zone: Very High", "hazard_class": "Very High"}
            ])],
        )
        wildfire = next(item for item in ranked if item["slug"] == "wildfire")
        self.assertEqual(wildfire["local_exposure_status"], "Direct mapped exposure")
        self.assertIn("Very High", " ".join(wildfire["local_exposure"]["source_specific_terminology"]))

    def test_outside_fhsz_does_not_claim_zero_wildfire_risk(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("wildfire", data_status="not_in_layer", is_in_hazard_zone=False)],
        )
        wildfire = next(item for item in ranked if item["slug"] == "wildfire")
        self.assertEqual(wildfire["local_exposure_status"], "No mapped exposure identified in checked layers")
        self.assertIn("does not mean", " ".join(wildfire["limitations"]))

    def test_fema_flood_intersection_adds_direct_exposure(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("flood", data_status="checked", is_in_hazard_zone=True, matched_layers=[
                {"name": "FEMA Zone AE", "zone": "AE"}
            ])],
        )
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(flood["local_exposure_status"], "Direct mapped exposure")
        self.assertIn("AE", flood["local_exposure"]["source_specific_terminology"])

    def test_failed_fema_request_returns_data_unavailable(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("flood", data_status="data_unavailable", is_in_hazard_zone=None)],
        )
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(flood["local_exposure_status"], "Data unavailable")
        self.assertEqual(flood["final_rating"], "Unknown")

    def test_tsunami_zone_intersection_can_move_tsunami_into_top_four(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("tsunami", additional_geospatial_evidence=[
                cgs_evidence("cgs_tsunami_hazard_area_remote", "tsunami", matched=True)
            ])],
            coordinates=coordinates_for_sub_area("West Oakland"),
        )
        tsunami = next(item for item in ranked if item["slug"] == "tsunami")
        self.assertFalse(tsunami["is_top_four"])
        self.assertEqual(tsunami["displayed_level_status"], "unsupported")
        self.assertEqual(tsunami["local_exposure_status"], "Direct mapped exposure")
        self.assertEqual(tsunami["displayed_hazard_level"], "Unknown")

    def test_hillside_landslide_exposure_can_move_landslide_higher(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("earthquake", additional_geospatial_evidence=[
                cgs_evidence("cgs_earthquake_landslide_remote", "earthquake-landslide", matched=True)
            ])],
            coordinates=coordinates_for_sub_area("East Oakland Hills"),
        )
        landslide = next(item for item in ranked if item["slug"] == "landslide")
        self.assertFalse(landslide["is_top_four"])
        self.assertEqual(landslide["displayed_level_status"], "unsupported")
        self.assertEqual(landslide["local_exposure_status"], "Additional mapped concern")
        self.assertEqual(landslide["displayed_hazard_level"], "Unknown")

    def test_all_five_hazards_returned_and_only_four_top(self):
        ranked = rank_hazards_for_risk_summary("Oakland", [])
        self.assertEqual({item["slug"] for item in ranked}, {"earthquake", "wildfire", "flood", "landslide", "tsunami"})
        prominent = [item for item in ranked if item["is_top_four"]]
        self.assertTrue(all(item["final_rating"] in {"High", "Medium"} for item in prominent))
        self.assertTrue(all(item["is_top_four"] for item in ranked if item["final_rating"] in {"High", "Medium"}))

    def test_population_exposure_percentages_do_not_determine_individual_rating(self):
        ranked = build_hazard_priority_results(
            "Oakland",
            [],
            optional_sub_area_match={
                "sub_area": "West Oakland",
                "sub_area_status": "Matched reliable boundary",
                "basis": "Test boundary",
                "epc_percent_exposed": "17.7%",
            },
        )
        earthquake = next(item for item in ranked if item["slug"] == "earthquake")
        self.assertEqual(earthquake["calculated_citywide_priority"], "High")
        self.assertNotIn("17.7", earthquake["why_this_rating"])

    def test_draft_sources_are_visibly_labeled(self):
        earthquake = next(item for item in rank_hazards_for_risk_summary("Oakland", []) if item["slug"] == "earthquake")
        self.assertEqual(earthquake["document_status"], "draft")
        self.assertEqual(earthquake["sources_used"][0]["status_label"], "Draft")

    def test_unknown_sub_area_boundaries_are_not_guessed(self):
        earthquake = next(item for item in rank_hazards_for_risk_summary("Oakland", []) if item["slug"] == "earthquake")
        self.assertEqual(earthquake["sub_area_context"]["sub_area"], "Unknown")
        self.assertEqual(earthquake["sub_area_context"]["sub_area_status"], "Coordinates unavailable; boundary not evaluated")

    def test_adding_another_jurisdiction_uses_data_not_core_edits(self):
        earthquake = next(item for item in rank_hazards_for_risk_summary("Test Jurisdiction", []) if item["slug"] == "earthquake")
        self.assertEqual(earthquake["calculated_citywide_priority"], "High")
        self.assertEqual(earthquake["probability"], "Medium")
        self.assertEqual(earthquake["impact"], "High")

    def test_official_oakland_polygon_assigns_one_sub_area(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [],
            coordinates=coordinates_for_sub_area("East Oakland Hills"),
        )
        sub_areas = {item["sub_area_context"]["sub_area"] for item in ranked}
        self.assertEqual(sub_areas, {"East Oakland Hills"})
        self.assertEqual(ranked[0]["sub_area_context"]["sub_area_match_status"], "Matched official Oakland plan-area polygon")

    def test_outside_official_polygons_keeps_sub_area_unknown(self):
        earthquake = next(item for item in rank_hazards_for_risk_summary(
            "Oakland",
            [],
            coordinates={"lat": 36.0, "lon": -120.0},
        ) if item["slug"] == "earthquake")
        self.assertEqual(earthquake["sub_area_context"]["sub_area"], "Unknown")
        self.assertEqual(earthquake["sub_area_context"]["sub_area_match_status"], "No official sub-area polygon match")

    def test_zip_only_never_assigns_sub_area(self):
        earthquake = next(item for item in rank_hazards_for_risk_summary(
            "Oakland",
            [],
            coordinates={"zip_code": "94605"},
        ) if item["slug"] == "earthquake")
        self.assertEqual(earthquake["sub_area_context"]["sub_area"], "Unknown")
        self.assertIn("Coordinates unavailable", earthquake["sub_area_context"]["sub_area_match_status"])

    def test_epc_context_does_not_change_probability_or_rating(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [],
            coordinates=coordinates_for_sub_area("East Oakland Hills"),
        )
        earthquake = next(item for item in ranked if item["slug"] == "earthquake")
        self.assertEqual(earthquake["probability"], "High")
        self.assertEqual(earthquake["final_rating"], "Unknown")
        self.assertEqual(earthquake["displayed_level_status"], "unsupported")
        self.assertEqual(earthquake["community_context"]["status"], "Available")
        self.assertIn("community context only", earthquake["community_context"]["summary"])

    def test_epc_zero_does_not_create_low_rating(self):
        wildfire = next(item for item in rank_hazards_for_risk_summary(
            "Oakland",
            [],
            coordinates=coordinates_for_sub_area("East Oakland Hills"),
        ) if item["slug"] == "wildfire")
        self.assertEqual(wildfire["probability"], "High")
        self.assertEqual(wildfire["final_rating"], "Unknown")
        self.assertEqual(wildfire["displayed_level_status"], "unsupported")
        self.assertNotEqual(wildfire["final_rating"], "Low")

    def test_physical_zone_can_raise_local_level_without_epc(self):
        ranked = build_hazard_priority_results(
            "Example",
            [hazard_result("tsunami", additional_geospatial_evidence=[
                cgs_evidence("cgs_tsunami_hazard_area_remote", "tsunami", matched=True)
            ])],
            citywide_lhmp_data=[{
                "jurisdiction": "Example",
                "hazard": "tsunami",
                "probability": "Low",
                "impact": "Low",
                "confidence": "Medium",
            }],
        )
        tsunami = next(item for item in ranked if item["slug"] == "tsunami")
        self.assertEqual(tsunami["lhmp_rating"], "Low")
        self.assertEqual(tsunami["final_rating"], "Low")
        self.assertEqual(tsunami["physical_exposure_status"], "Direct mapped exposure")

    def test_fema_annual_chance_category_can_raise_flood_level(self):
        ranked = build_hazard_priority_results(
            "Example",
            [hazard_result("flood", data_status="checked", is_in_hazard_zone=True, matched_layers=[
                {"name": "FEMA Zone AE", "zone": "AE"}
            ])],
            citywide_lhmp_data=[{
                "jurisdiction": "Example",
                "hazard": "flood",
                "probability": "Low",
                "impact": "Low",
                "confidence": "Medium",
            }],
        )
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(flood["lhmp_rating"], "Low")
        self.assertEqual(flood["final_rating"], "Low")
        self.assertEqual(flood["physical_exposure_status"], "Direct mapped exposure")

    def test_fhsz_severity_does_not_rewrite_probability(self):
        ranked = build_hazard_priority_results(
            "Example",
            [hazard_result("wildfire", data_status="checked", is_in_hazard_zone=True, matched_layers=[
                {"name": "CAL FIRE Fire Hazard Severity Zone: Very High", "hazard_class": "Very High"}
            ])],
            citywide_lhmp_data=[{
                "jurisdiction": "Example",
                "hazard": "wildfire",
                "probability": "Low",
                "impact": "Low",
                "confidence": "Medium",
            }],
        )
        wildfire = next(item for item in ranked if item["slug"] == "wildfire")
        self.assertEqual(wildfire["probability"], "Low")
        self.assertEqual(wildfire["final_rating"], "Low")
        self.assertEqual(wildfire["official_zone_category"], "Very High, CAL FIRE Fire Hazard Severity Zone: Very High")

    def test_unverified_oakland_hazards_are_not_displayed_as_high_or_medium(self):
        ranked = build_hazard_priority_results(
            "Oakland",
            [],
            coordinates=coordinates_for_sub_area("East Oakland Hills"),
            display_limit=1,
        )
        high_medium = [item for item in ranked if item["final_rating"] in {"High", "Medium"}]
        self.assertEqual(high_medium, [])
        self.assertTrue(all(item["displayed_level_status"] == "unsupported" for item in ranked))

    def test_medium_high_high_combines_to_high(self):
        result = combine_scenario_ratings([
            {"scenario_name": "a", "official_rating": "Medium"},
            {"scenario_name": "b", "official_rating": "High"},
            {"scenario_name": "c", "official_rating": "High"},
        ])
        self.assertEqual(result["combined_area_rating"], "High")

    def test_low_medium_combines_to_medium(self):
        result = combine_scenario_ratings([
            {"scenario_name": "a", "official_rating": "Low"},
            {"scenario_name": "b", "official_rating": "Medium"},
        ])
        self.assertEqual(result["combined_area_rating"], "Medium")

    def test_low_low_medium_combines_to_low(self):
        result = combine_scenario_ratings([
            {"scenario_name": "a", "official_rating": "Low"},
            {"scenario_name": "b", "official_rating": "Low"},
            {"scenario_name": "c", "official_rating": "Medium"},
        ])
        self.assertEqual(result["combined_area_rating"], "Low")

    def test_unknown_scenarios_are_excluded_and_disclosed(self):
        result = combine_scenario_ratings([
            {"scenario_name": "a", "official_rating": "Low"},
            {"scenario_name": "b", "official_rating": "Unknown"},
        ])
        self.assertEqual(result["combined_area_rating"], "Low")
        self.assertEqual(result["valid_scenario_count"], 1)
        self.assertEqual(result["excluded_scenario_count"], 1)

    def test_all_unknown_scenarios_produce_unknown(self):
        result = combine_scenario_ratings([
            {"scenario_name": "a", "official_rating": "Unknown"},
            {"scenario_name": "b", "official_rating": None},
        ])
        self.assertEqual(result["combined_area_rating"], "Unknown")
        self.assertEqual(result["valid_scenario_count"], 0)

    def test_one_valid_scenario_does_not_pretend_average_needed(self):
        result = combine_scenario_ratings([
            {"scenario_name": "a", "official_rating": "Medium"},
        ])
        self.assertEqual(result["combined_area_rating"], "Medium")
        self.assertIn("no averaging was required", result["combination_explanation"])

    def test_oakland_area_rating_preferred_over_probability_impact(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [],
            coordinates=coordinates_for_sub_area("Coliseum/Airport"),
        )
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(flood["official_lhmp_area_rating"], "Unknown")
        self.assertEqual(flood["stayready_fallback_rating"], "High")
        self.assertEqual(flood["displayed_hazard_level"], "Unknown")
        self.assertEqual(flood["displayed_level_status"], "unsupported")

    def test_different_plan_areas_do_not_display_unverified_categories(self):
        east = rank_hazards_for_risk_summary("Oakland", [], coordinates=coordinates_for_sub_area("East Oakland Hills"))
        west = rank_hazards_for_risk_summary("Oakland", [], coordinates=coordinates_for_sub_area("West Oakland"))
        east_flood = next(item for item in east if item["slug"] == "flood")
        west_flood = next(item for item in west if item["slug"] == "flood")
        self.assertEqual(east_flood["displayed_hazard_level"], "Unknown")
        self.assertEqual(west_flood["displayed_hazard_level"], "Unknown")
        self.assertEqual(east_flood["displayed_level_status"], "unsupported")
        self.assertEqual(west_flood["displayed_level_status"], "unsupported")

    def test_failed_gis_does_not_erase_valid_lhmp_area_rating(self):
        ranked = rank_hazards_for_risk_summary(
            "Oakland",
            [hazard_result("flood", data_status="data_unavailable", is_in_hazard_zone=None)],
            coordinates=coordinates_for_sub_area("West Oakland"),
        )
        flood = next(item for item in ranked if item["slug"] == "flood")
        self.assertEqual(flood["displayed_hazard_level"], "Unknown")
        self.assertEqual(flood["displayed_level_status"], "unsupported")
        self.assertEqual(flood["physical_exposure_status"], "Data unavailable")


if __name__ == "__main__":
    unittest.main()
