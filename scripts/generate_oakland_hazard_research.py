#!/usr/bin/env python3
"""Generate Phase 1 Oakland hazard-assessment research artifacts.

The generator audits existing local data and extraction candidates, but it does
not visually verify records. All generated assessment records remain
ineligible until a human review supplies complete page/table/row provenance.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "oakland_hazard_assessment"
TODAY = date.today().isoformat()
ADOPTED_PAGE_DIR = "research/oakland_hazard_assessment/page-images/adopted"

PLAN_AREAS = [
    "Central East Oakland",
    "Coliseum/Airport",
    "Downtown",
    "East Oakland Hills",
    "Eastlake/Fruitvale",
    "Glenview/Redwood Heights",
    "North Oakland Hills",
    "North Oakland/Adams Point",
    "West Oakland",
]
HAZARDS = ["earthquake", "wildfire", "flood", "landslide", "tsunami"]
METRIC_TYPES = [
    "official_hazard_priority",
    "probability",
    "impact",
    "scenario_hazard_rating",
    "physical_exposure",
    "population_exposure",
    "property_exposure",
    "modeled_loss",
    "community_vulnerability",
    "EPC_context",
    "historical_frequency",
    "preparedness_context",
    "unknown_metric",
]


def load_json(path: str) -> Any:
    with (ROOT / path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(name: str, payload: Any) -> None:
    with (OUT / name).open("w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write("\n")


def write_text(name: str, content: str) -> None:
    (OUT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def infer_hazard(text: str, fallback: str | None = None) -> str | None:
    haystack = text.lower()
    for hazard in HAZARDS:
        if hazard in haystack:
            return hazard
    if "fire hazard" in haystack:
        return "wildfire"
    if "seiche" in haystack:
        return "tsunami"
    return fallback


def infer_metric(text: str) -> str:
    haystack = text.lower()
    if "equity priority" in haystack or "epc" in haystack:
        return "EPC_context"
    if "population" in haystack or "people" in haystack or "household" in haystack:
        return "population_exposure"
    if "property" in haystack or "building" in haystack or "structure" in haystack:
        return "property_exposure"
    if "loss" in haystack or "damage" in haystack:
        return "modeled_loss"
    if "probability" in haystack or "likelihood" in haystack:
        return "probability"
    if "impact" in haystack or "severity" in haystack:
        return "impact"
    if "risk ranking" in haystack or "hazard rating" in haystack:
        return "scenario_hazard_rating"
    if "historical" in haystack or "past hazard" in haystack:
        return "historical_frequency"
    if "zone" in haystack or "inundation" in haystack or "susceptibility" in haystack:
        return "physical_exposure"
    return "unknown_metric"


def review_priority(metric_type: str, text: str) -> str:
    haystack = text.lower()
    if metric_type in {
        "official_hazard_priority",
        "probability",
        "impact",
        "scenario_hazard_rating",
    } or "risk ranking score" in haystack or "hazard risk rating" in haystack:
        return "A"
    if metric_type in {
        "physical_exposure",
        "population_exposure",
        "property_exposure",
        "modeled_loss",
    } or "critical facilit" in haystack or "lifeline" in haystack or "shelter" in haystack:
        return "B"
    if metric_type in {"EPC_context", "community_vulnerability"}:
        return "C"
    if metric_type in {"historical_frequency", "preparedness_context"}:
        return "D"
    return "E"


def plan_version_for_status(source_status: str) -> str:
    if source_status == "adopted":
        return "2021-2026"
    if source_status == "draft":
        return "2026-2031"
    return ""


def adopted_page_image(page: int | str | None) -> str:
    if not page:
        return ""
    return f"{ADOPTED_PAGE_DIR}/page-{int(page):04d}.png"


def source_pages_for_inventory(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages = {}
    for record in records:
        page_image = record.get("page_image_reference", "")
        key = (record["source_document"], record["source_status"], record.get("source_page"), page_image)
        if not record.get("source_page"):
            continue
        pages[key] = {
            "source_document": record["source_document"],
            "source_status": record["source_status"],
            "plan_version": record.get("plan_version", plan_version_for_status(record["source_status"])),
            "source_page": record.get("source_page"),
            "page_image_reference": page_image,
            "page_image_exists": bool(page_image and (ROOT / page_image).exists()),
        }
    return sorted(pages.values(), key=lambda item: (item["source_status"], item["source_document"], int(item["source_page"] or 0), item["page_image_reference"]))


def source_inventory_from_existing() -> list[dict[str, Any]]:
    records = []
    scenario = load_json("data/hazard_priority/oakland_lhmp_area_scenario_ratings.json")["records"]
    for idx, record in enumerate(scenario, 1):
        records.append({
            "record_id": f"adopted_scenario_{idx:04d}",
            "hazard": record["hazard"],
            "geographic_unit": "oakland_plan_area",
            "plan_area": record["plan_area"],
            "scenario": record["scenario_name"],
            "metric_type": "scenario_hazard_rating",
            "raw_value": record.get("official_numeric_risk_ranking_value"),
            "raw_category": record.get("official_rating"),
            "unit": "Oakland LHMP risk-ranking value and category",
            "denominator": "",
            "source_document": record["source_plan"],
            "source_status": record["document_status"],
            "plan_version": record.get("plan_version") or "2021-2026",
            "source_page": record.get("source_page"),
            "printed_page": "",
            "chapter": record.get("source_chapter", ""),
            "table_or_figure_number": "",
            "table_or_figure_title": record.get("source_table", ""),
            "source_table": record.get("source_table", ""),
            "row": record["plan_area"],
            "source_row": record["plan_area"],
            "column": "official_rating; official_numeric_risk_ranking_value",
            "source_column": "Risk Ranking Score; Hazard Risk Rating",
            "extracted_text": "",
            "what_the_metric_actually_measures": "Official adopted LHMP scenario rating for one hazard scenario and one Oakland plan area.",
            "permitted_use": "requires_visual_verification_before_assessment",
            "limitations": record.get("limitations", ""),
            "verification_status": "needs_more_review",
            "review_priority": "A",
            "page_image_reference": adopted_page_image(record.get("source_page")),
            "review_blockers": [] if (ROOT / adopted_page_image(record.get("source_page"))).exists() else ["missing_rendered_page_image"],
            "review_action_required": True,
        })

    for group_name, path in [
        ("draft_evidence", "data/lhmp/extracted/oakland/evidence_candidates.json"),
        ("draft_table", "data/lhmp/extracted/oakland/tables_candidates.json"),
        ("draft_visual", "data/lhmp/extracted/oakland/visuals_candidates.json"),
    ]:
        for idx, item in enumerate(load_json(path), 1):
            text = item.get("caption") or item.get("extracted_text") or item.get("extracted_snippet") or ""
            hazard = infer_hazard(text, item.get("suggested_hazard"))
            if hazard not in HAZARDS:
                continue
            metric = infer_metric(text)
            records.append({
                "record_id": f"{group_name}_{idx:04d}",
                "hazard": hazard,
                "geographic_unit": "candidate_text_or_visual",
                "plan_area": "",
                "scenario": "",
                "metric_type": metric,
                "raw_value": "",
                "raw_category": "",
                "unit": "",
                "denominator": "",
                "source_document": item.get("source_document", "oakland-draft-lhmp-2026-2031.pdf"),
            "source_status": "draft",
            "plan_version": "2026-2031",
            "source_page": item.get("pdf_page"),
                "printed_page": item.get("page_label", ""),
                "chapter": item.get("section_heading", ""),
                "table_or_figure_number": "",
            "table_or_figure_title": item.get("caption", ""),
            "source_table": item.get("caption", ""),
            "row": "",
            "source_row": "",
            "column": "",
            "source_column": "",
            "extracted_text": text[:1200],
            "what_the_metric_actually_measures": "Candidate extracted draft LHMP evidence; requires human interpretation before use.",
            "permitted_use": "candidate_review_only" if metric != "EPC_context" else "context_only",
            "limitations": "Machine-extracted candidate. It may be a table-of-contents reference, narrative context, OCR artifact, or incomplete row.",
            "verification_status": "context_only" if metric == "EPC_context" else "extracted_unverified",
            "page_image_reference": item.get("page_image_path", ""),
            "review_priority": review_priority(metric, text),
            "review_blockers": ["draft_not_adopted", "requires_human_review"],
            "review_action_required": True,
        })
    return records


def existing_data_audit() -> list[dict[str, Any]]:
    audited = []
    scenario = load_json("data/hazard_priority/oakland_lhmp_area_scenario_ratings.json")["records"]
    for idx, record in enumerate(scenario, 1):
        has_page = bool(record.get("source_page") and record.get("source_table"))
        audited.append({
            "audit_id": f"oakland_scenario_rating_{idx:04d}",
            "hazard": record.get("hazard"),
            "oakland_plan_area": record.get("plan_area"),
            "scenario": record.get("scenario_name"),
            "stored_value": record.get("official_numeric_risk_ranking_value"),
            "stored_category": record.get("official_rating"),
            "file": "data/hazard_priority/oakland_lhmp_area_scenario_ratings.json",
            "exact_key": f"records[{idx - 1}]",
            "claimed_source": record.get("source_plan"),
            "claimed_source_page": record.get("source_page"),
            "origin": "manual_or_generated_local_json",
            "current_verification_status": record.get("verification_status", "extracted_unverified"),
            "used_by_production": False,
            "supported": False,
            "recommended_action": "retain_for_research_visual_review",
            "reason": "Has source page/table metadata but lacks row-level visual verification fields required by Phase 1.",
            "has_page_table_provenance": has_page,
            "legacy_value_must_not_drive_research": True,
        })

    for filename in [
        "jurisdiction_hazard_rankings.json",
        "sub_area_evidence.json",
        "source_documents.json",
        "official_gis_layer_rules.json",
        "top_four_rules.json",
    ]:
        path = f"data/hazard_priority/{filename}"
        payload = load_json(path)
        collection = payload.get("records") if isinstance(payload, dict) else None
        if collection is None:
            collection = payload.get("rules") if isinstance(payload, dict) else None
        if collection is None:
            collection = list((payload.get("sources") or {}).values()) if isinstance(payload, dict) else []
        for idx, record in enumerate(collection, 1):
            hazard = infer_hazard(json.dumps(record), record.get("hazard") if isinstance(record, dict) else None)
            audited.append({
                "audit_id": f"{slug(filename)}_{idx:04d}",
                "hazard": hazard or "",
                "oakland_plan_area": record.get("plan_area", "") if isinstance(record, dict) else "",
                "scenario": record.get("scenario", "") if isinstance(record, dict) else "",
                "stored_value": record.get("score", record.get("priority_score", "")) if isinstance(record, dict) else "",
                "stored_category": record.get("priority", record.get("rating", "")) if isinstance(record, dict) else "",
                "file": path,
                "exact_key": f"records_or_rules[{idx - 1}]",
                "claimed_source": record.get("source_document", record.get("source_name", "")) if isinstance(record, dict) else "",
                "claimed_source_page": record.get("source_page", "") if isinstance(record, dict) else "",
                "origin": "configuration_or_supporting_metadata",
                "current_verification_status": record.get("verification_status", "not_research_verified") if isinstance(record, dict) else "not_research_verified",
                "used_by_production": filename != "source_documents.json",
                "supported": filename in {"source_documents.json", "official_gis_layer_rules.json"},
                "recommended_action": "retain_as_production_configuration_but_do_not_trust_for_research_assessment",
                "reason": "Production support/config data is not a visually verified research assessment record.",
                "legacy_value_must_not_drive_research": True,
            })
    return audited


def matrix_for(records: list[dict[str, Any]], source_status: str) -> dict[str, Any]:
    cells = []
    for area in PLAN_AREAS:
        for hazard in HAZARDS:
            candidates = [
                r for r in records
                if r["source_status"] == source_status and r["hazard"] == hazard and r.get("plan_area") == area
            ]
            verified = [r for r in candidates if r["verification_status"] in {"visually_verified", "corrected_after_visual_review"}]
            cells.append({
                "plan_area": area,
                "hazard": hazard,
                "official_area_rating_exists": any(r["metric_type"] == "scenario_hazard_rating" for r in candidates),
                "multiple_scenarios_exist": len({r.get("scenario") for r in candidates if r.get("scenario")}) > 1,
                "probability_exists": any(r["metric_type"] == "probability" for r in candidates),
                "impact_exists": any(r["metric_type"] == "impact" for r in candidates),
                "only_exposure_or_context_exists": bool(candidates) and all(r["metric_type"] in {"physical_exposure", "population_exposure", "property_exposure", "EPC_context", "community_vulnerability"} for r in candidates),
                "source_provenance_record_ids": [r["record_id"] for r in candidates],
                "missing_evidence": [] if verified else ["visual_verification", "complete_row_column_provenance"],
                "conflicts": [],
                "future_user_facing_assessment_supportable": bool(verified),
            })
    return {
        "schema_version": 1,
        "jurisdiction": "oakland",
        "source_status": source_status,
        "generated_at": TODAY,
        "cells": cells,
    }


def coverage_report(adopted: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    cells = adopted["cells"] + draft["cells"]
    counts = Counter()
    by_hazard = defaultdict(Counter)
    for cell in cells:
        if cell["future_user_facing_assessment_supportable"]:
            bucket = "supportable"
        elif cell["source_provenance_record_ids"]:
            bucket = "candidate_or_context_only"
        else:
            bucket = "missing"
        counts[bucket] += 1
        by_hazard[cell["hazard"]][bucket] += 1
    return {
        "schema_version": 1,
        "generated_at": TODAY,
        "total_cells": len(cells),
        "counts": dict(counts),
        "percentages": {key: round(value / len(cells) * 100, 2) for key, value in counts.items()},
        "by_hazard": {hazard: dict(counter) for hazard, counter in sorted(by_hazard.items())},
        "note": "No generated cell is supportable until records are visually verified by a human reviewer.",
    }


def triage_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_priority = Counter(record.get("review_priority", "E") for record in records)
    by_hazard = Counter(record["hazard"] for record in records)
    by_plan_version = Counter(record.get("plan_version", "") for record in records)
    by_plan_area = Counter(record.get("plan_area") or "not_area_specific" for record in records)
    by_metric_type = Counter(record["metric_type"] for record in records)
    by_verification_status = Counter(record["verification_status"] for record in records)
    cross = defaultdict(Counter)
    for record in records:
        cross[record.get("review_priority", "E")][record["hazard"]] += 1
    return {
        "schema_version": 1,
        "generated_at": TODAY,
        "priority_definitions": {
            "A": "assessment-critical",
            "B": "physical exposure and consequence",
            "C": "community context",
            "D": "supporting context",
            "E": "unclear or unusable",
        },
        "counts": {
            "by_priority": dict(sorted(by_priority.items())),
            "by_hazard": dict(sorted(by_hazard.items())),
            "by_plan_version": dict(sorted(by_plan_version.items())),
            "by_plan_area": dict(sorted(by_plan_area.items())),
            "by_metric_type": dict(sorted(by_metric_type.items())),
            "by_verification_status": dict(sorted(by_verification_status.items())),
            "by_priority_and_hazard": {priority: dict(sorted(counter.items())) for priority, counter in sorted(cross.items())},
        },
    }


def review_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    return sorted(
        records,
        key=lambda record: (
            priority_order.get(record.get("review_priority", "E"), 9),
            record.get("source_status", ""),
            record.get("hazard", ""),
            record.get("source_page") or 99999,
            record.get("plan_area", ""),
        ),
    )


def adopted_draft_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    adopted = [record for record in records if record["source_status"] == "adopted"]
    draft = [record for record in records if record["source_status"] == "draft"]
    draft_by_hazard = defaultdict(list)
    for record in draft:
        draft_by_hazard[record["hazard"]].append(record)
    comparisons = []
    for record in adopted:
        matches = [
            draft_record for draft_record in draft_by_hazard.get(record["hazard"], [])
            if draft_record["metric_type"] == record["metric_type"]
        ]
        comparisons.append({
            "adopted_record_id": record["record_id"],
            "hazard": record["hazard"],
            "plan_area": record.get("plan_area", ""),
            "scenario": record.get("scenario", ""),
            "adopted_value": record.get("raw_value"),
            "adopted_category": record.get("raw_category"),
            "draft_candidate_record_ids": [item["record_id"] for item in matches[:10]],
            "value_changed": "not_reviewed",
            "methodology_changed": "not_reviewed",
            "plan_area_boundaries_changed": "not_reviewed",
            "scenarios_changed": "not_reviewed",
            "draft_provides_absent_information": bool(matches),
        })
    return {
        "schema_version": 1,
        "generated_at": TODAY,
        "adopted_records_compared": len(adopted),
        "draft_records_available": len(draft),
        "status": "comparison_candidates_only_no_draft_recommendation",
        "comparisons": comparisons,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = source_inventory_from_existing()
    audit = existing_data_audit()
    adopted = matrix_for(inventory, "adopted")
    draft = matrix_for(inventory, "draft")
    coverage = coverage_report(adopted, draft)
    triage = triage_report(inventory)
    pages = source_pages_for_inventory(inventory)
    comparison = adopted_draft_comparison(inventory)
    counts = Counter(r["verification_status"] for r in inventory)
    hazard_counts = Counter(r["hazard"] for r in inventory)

    write_json("existing_data_audit.json", {"schema_version": 1, "generated_at": TODAY, "records": audit})
    write_json("source_inventory.json", {"schema_version": 1, "generated_at": TODAY, "allowed_metric_types": METRIC_TYPES, "records": inventory})
    write_json("source_page_catalog.json", {"schema_version": 1, "generated_at": TODAY, "pages": pages})
    write_json("triage_report.json", triage)
    write_json("visual_verification_queue.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "eligible_statuses_after_review": ["visually_verified", "corrected_after_visual_review"],
        "allowed_review_statuses": [
            "visually_verified",
            "corrected_after_visual_review",
            "context_only",
            "rejected",
            "superseded",
            "needs_more_review",
        ],
        "records": [
            {
                "record_id": r["record_id"],
                "review_priority": r.get("review_priority", "E"),
                "hazard": r["hazard"],
                "plan_area": r.get("plan_area", ""),
                "scenario": r.get("scenario", ""),
                "metric_type": r["metric_type"],
                "plan_version": r.get("plan_version", ""),
                "source_status": r.get("source_status", ""),
                "source_document": r["source_document"],
                "source_page": r["source_page"],
                "source_table": r.get("table_or_figure_title", ""),
                "source_row": r.get("source_row", r.get("row", "")),
                "source_column": r.get("source_column", r.get("column", "")),
                "raw_value": r.get("raw_value", ""),
                "raw_category": r.get("raw_category", ""),
                "page_image_reference": r.get("page_image_reference", ""),
                "extracted_text": r.get("extracted_text", ""),
                "proposed_interpretation": r.get("what_the_metric_actually_measures", ""),
                "proposed_permitted_use": r.get("permitted_use", ""),
                "review_blockers": r.get("review_blockers", []),
                "approval_status": r["verification_status"],
            }
            for r in review_queue(inventory)
        ],
    })
    write_json("verified_adopted_matrix.json", adopted)
    write_json("verified_draft_matrix.json", draft)
    write_json("matrix_coverage_report.json", coverage)
    write_json("adopted_draft_comparison.json", comparison)
    write_json("methodology_report.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "hazards": {
            hazard: {
                "recommendation": "mapped_findings_only_until_visual_verification_and_method_approval",
                "official_citywide_rating_found": "unknown_pending_inventory_review",
                "plan_area_ratings_found": hazard == "earthquake",
                "multiple_scenarios_found": hazard == "earthquake",
                "probability_varies_geographically": "not_determined",
                "impact_varies_geographically": "not_determined",
                "documented_combination_method": "not_approved",
                "prohibited_methods": [
                    "scenario_averaging_without_scale_confirmation",
                    "combining_adopted_and_draft_records",
                    "using_context_only_metrics_as_hazard_categories",
                ],
            }
            for hazard in HAZARDS
        },
    })
    write_json("plan_area_geometry_validation.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "source_service": "https://gismaps.oaklandca.gov/server/rest/services/Accela/Citywide_202410/MapServer/9",
        "layer_id": 9,
        "official_feature_names": PLAN_AREAS,
        "geometry_version": "Citywide_202410 MapServer layer 9; fetched in existing repo metadata 2026-06-17",
        "validation_status": "fixture_pending",
        "fail_closed_rules": [
            "zip_only_input_cannot_assign_area",
            "multiple_match_requires_warning",
            "outside_oakland_returns_no_area",
            "invalid_geometry_disables_area_assignment",
        ],
    })
    write_json("gis_source_validation.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "hazards": {hazard: {"validation_status": "source_semantics_documented_tests_pending_live_review"} for hazard in HAZARDS},
        "fail_closed_rules": [
            "failed_gis_is_data_unavailable_not_low",
            "non_intersection_is_not_guaranteed_safety",
            "fault_proximity_is_not_fault_zone_intersection",
            "urban_unzoned_is_not_no_wildfire_risk",
            "tsunami_evacuation_and_inundation_terms_remain_separate",
        ],
    })
    write_json("shadow_report.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "mode": "development_only",
        "production_connected": False,
        "addresses": [],
        "required_coverage": PLAN_AREAS,
        "status": "fixture_collection_pending",
    })
    write_json("manual_spot_check_plan.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "status": "pending_human_review",
        "required_sample": [
            "at_least_three_verified_records_per_hazard",
            "every_scenario_type",
            "every_plan_area",
            "adopted_and_draft_examples",
            "corrected_records",
            "rejected_records",
            "context_only_records",
        ],
        "spot_checked_records": [],
    })
    write_json("observatory_ave_diagnostic.json", {
        "schema_version": 1,
        "generated_at": TODAY,
        "address": "4183 Observatory Ave, Oakland, CA",
        "status": "permanent_regression_case_created_fixture_pending",
        "production_connected": False,
        "required_trace_fields": [
            "normalized_address",
            "geocoder_coordinates",
            "matched_plan_area",
            "boundary_distance",
            "AP_zone_result",
            "fault_proximity",
            "liquefaction_result",
            "earthquake_induced_landslide_result",
            "wildfire_FHSZ",
            "FEMA_result",
            "tsunami_result",
            "verified_LHMP_records",
            "proposed_assessment",
            "provenance",
            "warnings",
        ],
    })

    write_text("existing_data_audit.md", f"""# Existing Oakland Data Audit

Generated: {TODAY}

Audited records: {len(audit)}

The existing Oakland scenario ratings are retained only as research candidates. They are not research-active because they lack complete visual-verification fields such as exact row/column review, page-image approval, reviewer, and verified date.

Production use remains disconnected from these research artifacts.
""")
    write_text("source_inventory.md", f"""# Oakland LHMP Source Inventory

Generated: {TODAY}

Candidate records: {len(inventory)}

By verification status: {dict(counts)}

By hazard: {dict(hazard_counts)}

This inventory distinguishes adopted and draft records. Machine-extracted draft candidates remain unverified or context-only until visual review.
""")
    write_text("triage_report.md", f"""# Oakland Candidate Triage

Generated: {TODAY}

Priority counts: {triage['counts']['by_priority']}

Records by hazard: {triage['counts']['by_hazard']}

Records by plan version: {triage['counts']['by_plan_version']}

Review order: Priority A adopted records first, then official GIS and agency evidence, then draft records, then context-only evidence.
""")
    write_text("visual_verification_review.html", """<!doctype html>
<meta charset="utf-8">
<title>Oakland Hazard Assessment Visual Review Queue</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0; color: #1f2933; }
header { padding: 16px 24px; border-bottom: 1px solid #d9e2ec; }
main { display: grid; grid-template-columns: minmax(360px, 1fr) minmax(420px, 0.95fr); gap: 16px; padding: 16px; }
img { max-width: 100%; border: 1px solid #bcccdc; background: #fff; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f0f4f8; padding: 12px; }
select, button { font: inherit; }
.queue { max-height: 84vh; overflow: auto; border: 1px solid #d9e2ec; }
.record { display: block; width: 100%; text-align: left; padding: 10px; border: 0; border-bottom: 1px solid #d9e2ec; background: white; }
.record:focus, .record:hover { background: #f0f4f8; }
.meta { color: #52606d; font-size: 13px; }
.status { font-weight: 700; }
</style>
<header>
<h1>Oakland Hazard Assessment Visual Review Queue</h1>
<p>This research-only page loads <code>visual_verification_queue.json</code>. It does not write approvals. Reviewers must record explicit review actions separately.</p>
</header>
<main>
  <section>
    <h2>Queue</h2>
    <div id="queue" class="queue"></div>
  </section>
  <section>
    <h2 id="title">Select a record</h2>
    <p class="meta" id="meta"></p>
    <img id="page" alt="PDF page image unavailable">
    <h3>Extracted text</h3>
    <pre id="text"></pre>
    <h3>Structured record</h3>
    <pre id="json"></pre>
  </section>
</main>
<script>
async function loadQueue() {
  const response = await fetch('visual_verification_queue.json');
  const data = await response.json();
  const queue = document.getElementById('queue');
  const title = document.getElementById('title');
  const meta = document.getElementById('meta');
  const page = document.getElementById('page');
  const text = document.getElementById('text');
  const json = document.getElementById('json');
  data.records.forEach((record, index) => {
    const button = document.createElement('button');
    button.className = 'record';
    button.innerHTML = `<span class="status">Priority ${record.review_priority}</span> ${record.hazard} ${record.plan_area || 'citywide/candidate'}<br><span class="meta">${record.source_status} ${record.plan_version} page ${record.source_page || 'unknown'} - ${record.approval_status}</span>`;
    button.addEventListener('click', () => {
      title.textContent = `${record.record_id}: ${record.hazard}`;
      meta.textContent = `${record.source_document} | ${record.source_table || 'No table'} | row: ${record.source_row || 'unidentified'} | column: ${record.source_column || 'unidentified'}`;
      page.src = record.page_image_reference || '';
      text.textContent = record.extracted_text || '(No extracted text captured for this candidate.)';
      json.textContent = JSON.stringify(record, null, 2);
    });
    queue.appendChild(button);
    if (index === 0) button.click();
  });
}
loadQueue();
</script>
""")
    write_text("matrix_coverage_report.md", f"""# Oakland Matrix Coverage

Generated: {TODAY}

Total adopted plus draft cells: {coverage['total_cells']}

Coverage counts: {coverage['counts']}

No matrix cell is supportable until visually verified source records are present with complete provenance.
""")
    write_text("methodology_report.md", """# Oakland Methodology Review

Recommendation for all five hazards in Phase 1: keep production on mapped findings only. Do not combine adopted and draft values, do not average context-only metrics, and do not emit official-sounding Low / Medium / High categories until hazard-specific methods are approved.

- Earthquake: adopted scenario candidates exist, but scenario averaging remains unapproved for research until visual verification and scale review.
- Wildfire: use official GIS semantics and LHMP evidence separately; no combined category yet.
- Flood: keep FEMA/NFHL terms separate from probability and impact.
- Landslide: keep earthquake-induced landslide and broader landslide evidence separate.
- Tsunami: do not merge evacuation-area and inundation terminology.
""")
    write_text("adopted_draft_comparison.md", f"""# Adopted Versus Draft Comparison

Generated: {TODAY}

Adopted records compared: {comparison['adopted_records_compared']}

Draft records available: {comparison['draft_records_available']}

Status: candidate comparison only. Draft values are not recommended for production and do not overwrite adopted values.
""")
    write_text("manual_spot_check_plan.md", """# Manual Spot Check Plan

Manual spot checks are pending. The required sample includes at least three verified records per hazard, every scenario type, every plan area, adopted and draft examples, corrected records, rejected records, and context-only records.
""")
    write_text("plan_area_geometry_validation.md", """# Oakland Plan-Area Geometry Validation

The official City of Oakland Plan Area CW layer is documented for research use, but representative-point and boundary fixtures still require human/live-service validation. ZIP-only and neighborhood-only inputs must fail closed.
""")
    write_text("gis_source_validation.md", """# Oakland GIS Source Validation

Official GIS evidence is tracked separately from LHMP area evidence. Failed services become Data unavailable, successful non-intersections are not safety guarantees, and proximity results are not polygon intersections.
""")
    write_text("shadow_report.md", """# Oakland Shadow Report

Shadow mode is development-only and disconnected from production. Address fixtures, including 4183 Observatory Ave, are initialized as pending until provenance-labeled official responses are captured.
""")
    write_text("human_review_report.md", f"""# Oakland Hazard Assessment Human Review Package

Generated: {TODAY}

Artifacts included:

- existing data audit
- source inventory
- visual verification queue
- adopted and draft matrices
- matrix coverage report
- methodology report
- plan-area geometry validation notes
- GIS source validation notes
- shadow-mode scaffold
- 4183 Observatory Ave diagnostic scaffold

Official vs StayReady status:

- Official adopted and draft source candidates are stored separately.
- No StayReady-created research category is active.
- Context-only records cannot drive physical hazard ratings.
- Unsupported and missing evidence remains explicit.
- Human approval is still required before any research category can become supportable.
""")


if __name__ == "__main__":
    main()
