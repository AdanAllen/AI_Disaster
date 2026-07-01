import re
import unittest
from pathlib import Path
from unittest.mock import patch

import hazard_priority
from risk_summary_view_model import build_canonical_risk_summary


BASE_DIR = Path(__file__).resolve().parents[1]


def location(city="Berkeley"):
    return {
        "location_mode": "address",
        "has_precise_location": True,
        "display_name": f"{city} City Hall",
        "city": city,
        "county": "Alameda County",
    }


def evidence(*, matched, dataset, label, claim_type="regulatory_zone"):
    return {
        "matched": matched,
        "dataset_id": dataset.lower().replace(" ", "_"),
        "dataset_name": dataset,
        "result_label": label,
        "status_label": "Mapped match found" if matched else "No mapped match found",
        "claim_type": claim_type,
        "source_agency": "Official test agency",
        "source_url": "https://example.gov/dataset",
        "public_claim_status": "official_provisional",
    }


def hazard(slug="earthquake", *, exposure="no_mapped_match", data_status="not_in_layer", records=None, why="Preparedness remains important."):
    return {
        "slug": slug,
        "name": slug.replace("-", " ").title(),
        "scope": "address_level",
        "hazard_exposure": exposure,
        "data_status": data_status,
        "normalized_mapped_evidence": records or [],
        "additional_geospatial_evidence": records or [],
        "priority_reasons": [why],
        "why_shown": why,
        "review_status": "draft_reviewed",
        "limitations": ["Not a property-damage prediction."],
        "sources": [{"name": "Official test source", "url": "https://example.gov/source"}],
        "recommended_actions": [{
            "instruction": "Practice the official protective action.",
            "time_buckets": ["before"],
        }],
    }


def oakland_priority():
    return {
        "slug": "earthquake",
        "probability": "High",
        "impact": "Medium",
        "official_lhmp_priority": "High",
        "document_status": "draft",
        "source_document": "Draft Oakland 2026-2031 Local Hazard Mitigation Plan",
        "source_page": 316,
        "source_table": "Table 17-1",
        "sources_used": [{
            "name": "Draft Oakland 2026-2031 Local Hazard Mitigation Plan",
            "url": "https://www.oaklandca.gov/topics/local-hazard-mitigation-plan",
            "reference": "Table 17-1; page 316",
            "document_status": "draft",
        }],
        "limitations": ["Draft source."],
        "sub_area_context": {
            "sub_area": "Downtown",
            "sub_area_status": "Matched official Oakland plan-area polygon",
            "sub_area_match_status": "Matched official Oakland plan-area polygon",
            "boundary_warning": "",
        },
    }


class CanonicalRiskSummaryTests(unittest.TestCase):
    def test_berkeley_nonmatch_has_unknown_vulnerability_and_no_address_rating(self):
        record = evidence(matched=False, dataset="CGS Liquefaction Zones", label="No mapped match found.")
        model = build_canonical_risk_summary(location(), [hazard(records=[record])], [])
        item = model["hazards"][0]
        self.assertEqual(item["address_evidence"]["outcome"], "checked_no_match")
        self.assertEqual(item["regional_context"]["status"], "unavailable")
        self.assertEqual(item["vulnerability"]["status"], "unknown")
        self.assertNotRegex(str(item["address_evidence"]), re.compile(r"address.{0,12}(High|Medium|Low)", re.I))

    def test_oakland_draft_citywide_rating_remains_visibly_citywide(self):
        record = evidence(matched=False, dataset="CGS Liquefaction Zones", label="No mapped match found.")
        model = build_canonical_risk_summary(location("Oakland"), [hazard(records=[record])], [oakland_priority()])
        item = model["hazards"][0]
        self.assertEqual(item["regional_context"]["priority_label"], "High")
        self.assertEqual(item["regional_context"]["source_status"], "draft")
        self.assertIn("citywide", item["regional_context"]["explanation"])
        self.assertEqual(item["address_evidence"]["outcome"], "checked_no_match")
        self.assertNotIn("High", item["address_evidence"]["label"])

    def test_oakland_subarea_context_is_separate_reviewed_and_cited(self):
        record = evidence(matched=False, dataset="CGS Liquefaction Zones", label="No mapped match found.")
        item = build_canonical_risk_summary(location("Oakland"), [hazard(records=[record])], [oakland_priority()])["hazards"][0]
        self.assertEqual(item["subarea_context"]["status"], "available")
        self.assertEqual(item["subarea_context"]["subarea_name"], "Downtown")
        self.assertEqual(item["subarea_context"]["source_status"], "draft")
        self.assertIn("Hayward M7.05 scenario", item["subarea_context"]["findings"][0]["text"])
        self.assertIn("Table 9-6", item["subarea_context"]["findings"][0]["reference"])
        self.assertIn("Table 9-6", item["subarea_context"]["claims"][0]["reference"])
        self.assertIn("PDF page 186", item["subarea_context"]["claims"][0]["reference"])
        self.assertEqual(item["regional_context"]["priority_label"], "High")
        self.assertEqual(item["address_evidence"]["outcome"], "checked_no_match")
        self.assertNotRegex(str(item["subarea_context"]), re.compile(r"property.{0,12}(High|Medium|Low)", re.I))

    def test_positive_address_finding_outranks_subarea_attention(self):
        record = evidence(matched=True, dataset="CGS Liquefaction Zones", label="Inside a CGS mapped liquefaction zone.")
        item = build_canonical_risk_summary(
            location("Oakland"),
            [hazard(exposure="mapped_match", data_status="checked", records=[record])],
            [oakland_priority()],
        )["hazards"][0]
        self.assertEqual(item["attention"]["label"], "Address finding")
        self.assertEqual(item["subarea_context"]["status"], "available")

    def test_unreviewed_subarea_value_is_withheld(self):
        source = hazard_priority.load_priority_data()
        fake = {**source, "sub_area_evidence": {
            "review_gate": {"production_statuses": ["reviewed"], "required_fields": ["reviewer"]},
            "records": [{
                "jurisdiction": "Oakland", "subarea_name": "Downtown", "hazard_id": "earthquake",
                "review_status": "candidate", "reviewer": "", "permitted_use": "subarea_context_only",
                "display_text": "Invented candidate value.",
            }],
        }}
        hazard_priority._community_context_for_hazard.__globals__["load_priority_data"].cache_clear()
        with patch("hazard_priority.load_priority_data", return_value=fake):
            result = hazard_priority.reviewed_subarea_context_for_hazard("Oakland", "Downtown", "earthquake")
        self.assertEqual(result["status"], "Unavailable")
        self.assertNotIn("Invented", str(result))

    def test_positive_flood_preserves_official_category_without_overall_score(self):
        record = evidence(matched=True, dataset="FEMA NFHL", label="Special Flood Hazard Area, Zone AE")
        item = build_canonical_risk_summary(location(), [hazard("flood", exposure="mapped_match", data_status="checked", records=[record])], [])["hazards"][0]
        self.assertEqual(item["address_evidence"]["outcome"], "mapped_match")
        self.assertIn("Zone AE", item["address_evidence"]["findings"][0])
        self.assertEqual(item["address_evidence"]["claims"][0]["dataset_or_document"], "FEMA NFHL")
        self.assertEqual(item["attention"]["label"], "Address finding")
        self.assertNotIn("risk", item["address_evidence"]["label"].lower())
        self.assertEqual(len(item["address_evidence"]["findings"]), 1)

    def test_positive_wildfire_preserves_severity_as_mapped_category(self):
        record = evidence(matched=True, dataset="CAL FIRE FHSZ", label="Very High Fire Hazard Severity Zone")
        item = build_canonical_risk_summary(location(), [hazard("wildfire", exposure="mapped_match", data_status="checked", records=[record])], [])["hazards"][0]
        self.assertIn("Very High Fire Hazard Severity Zone", item["address_evidence"]["findings"])
        self.assertIn("fire-hazard-severity", item["address_evidence"]["explanation"])
        self.assertIn("does not estimate personal annual wildfire probability", item["address_evidence"]["explanation"].lower())

    def test_fault_proximity_preserves_distance_without_zone_membership(self):
        record = evidence(matched=True, dataset="USGS fault traces", label="Hayward Fault — 0.4 km away", claim_type="proximity")
        item = build_canonical_risk_summary(location(), [hazard(exposure="proximity_context", data_status="checked", records=[record])], [])["hazards"][0]
        self.assertEqual(item["address_evidence"]["outcome"], "proximity_context")
        self.assertIn("0.4 km", item["address_evidence"]["findings"][0])
        self.assertIn("not polygon", item["address_evidence"]["explanation"])
        self.assertNotIn("High", item["address_evidence"]["label"])
        self.assertEqual(item["address_evidence"]["claims"][0]["dataset_or_document"], "USGS fault traces")
        self.assertEqual(item["address_evidence"]["claims"][0]["url"], "https://example.gov/dataset")

    def test_unavailable_does_not_become_low_or_use_zip(self):
        unavailable = hazard(exposure="not_checked", data_status="data_unavailable")
        unavailable["legacy_score"] = 1
        item = build_canonical_risk_summary(location(), [unavailable], [])["hazards"][0]
        self.assertEqual(item["address_evidence"]["outcome"], "data_unavailable")
        self.assertNotRegex(str(item["address_evidence"]), re.compile(r"\bLow\b"))
        self.assertNotIn("ZIP", str(item["address_evidence"]))

    def test_false_match_with_intersection_words_fails_closed(self):
        contradictory = evidence(matched=False, dataset="CGS Liquefaction Zones", label="Inside mapped liquefaction zone")
        item = build_canonical_risk_summary(location(), [hazard(records=[contradictory])], [])["hazards"][0]
        self.assertEqual(item["address_evidence"]["outcome"], "checked_no_match")
        self.assertEqual(item["address_evidence"]["findings"], [])
        self.assertEqual(item["address_evidence"]["claims"], [])

    def test_template_and_css_define_separate_subarea_column_and_mobile_cards(self):
        template = (BASE_DIR / "templates" / "_risk_summary_reference.html").read_text(encoding="utf-8")
        css = (BASE_DIR / "static" / "css" / "stayready.css").read_text(encoding="utf-8")
        for label in (
            "Regional / city context",
            "Oakland sub-area context",
            "What we found at your address",
            "Building &amp; household vulnerability",
            "Why this matters / what to do",
        ):
            self.assertIn(label, template)
        self.assertIn("hazard.vulnerability.label", template)
        self.assertIn("hazard.subarea_context", template)
        self.assertIn("hazard.detail_url", template)
        self.assertIn(".sr-risk-table thead { display: none; }", css)
        self.assertIn('content: attr(data-label)', css)


if __name__ == "__main__":
    unittest.main()
