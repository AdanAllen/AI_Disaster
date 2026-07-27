import json
from pathlib import Path

from hazard_engine import check_flood_layer, check_wildfire_layer


ROOT = Path(__file__).resolve().parents[1]


def test_fema_regression_controls_match_current_official_categories():
    sunol = check_flood_layer(37.5944011, -121.8876726)
    boundary = check_flood_layer(37.58550595, -122.1412945052)
    zone_d = check_flood_layer(37.5519146164, -122.0868773141)

    assert sunol["evaluation_status"] == "not_matched"
    assert {item["zone"] for item in sunol["layers"]} == {"X"}
    assert boundary["evaluation_status"] == "matched"
    assert {item["zone"] for item in boundary["layers"]} == {"AE"}
    assert zone_d["evaluation_status"] == "not_evaluated"
    assert {item["zone"] for item in zone_d["layers"]} == {"D"}


def test_calfire_selects_effective_sra_after_responsibility_check():
    result = check_wildfire_layer(37.5944011, -121.8876726)

    assert result["responsibility_area"] == "SRA"
    assert result["applicability_status"] == "effective_sra"
    assert result["evaluation_status"] == "matched"
    assert {item["hazard_class"] for item in result["layers"]} == {"High"}
    assert all(item["authority"] == "effective_sra" for item in result["layers"])


def test_calfire_lra_recommendation_is_not_an_adopted_match():
    result = check_wildfire_layer(37.7664278, -122.2436494)

    assert result["responsibility_area"] == "LRA"
    assert result["applicability_status"] == "lra_recommendation_only"
    assert result["inside"] is False
    assert result["layers"] == []


def test_calfire_fra_is_not_misreported_as_a_non_match():
    result = check_wildfire_layer(37.6803311499, -121.7075976692)

    assert result["responsibility_area"] == "FRA"
    assert result["evaluation_status"] == "not_evaluated"
    assert result["checked"] is False
    assert result["inside"] is False


def test_lra_adoption_matrix_is_complete_and_fail_closed():
    matrix = json.loads(
        (ROOT / "data/geospatial/calfire_lra_adoption_matrix.json").read_text()
    )

    assert len(matrix["jurisdictions"]) == 15
    assert {
        item["adoption_status"] for item in matrix["jurisdictions"]
    } == {"ordinance_not_verified"}
    assert all(item["adoption_evidence_url"] is None for item in matrix["jurisdictions"])
