"""Development-only Oakland hazard assessment engine.

This engine is deliberately disconnected from production routes. It only emits
categories from visually verified source records with full provenance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .constants import HAZARDS
from .validators import (
    detect_conflicting_records,
    is_record_eligible_for_research_assessment,
)

ASSESSMENT_STATUSES = {
    "verified_official",
    "verified_stayready_summary",
    "source_verified_method_pending",
    "incomplete",
    "unsupported",
    "conflicting_sources",
    "data_unavailable",
}


def _records_for(records: list[dict[str, Any]], hazard: str, plan_area: str, source_status: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("hazard") == hazard
        and record.get("plan_area") == plan_area
        and record.get("source_status") == source_status
    ]


def _public_trace(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record.get("record_id"),
        "source_document": record.get("source_document"),
        "source_status": record.get("source_status"),
        "source_page": record.get("source_page"),
        "printed_page": record.get("printed_page"),
        "source_table": record.get("source_table"),
        "source_row": record.get("source_row"),
        "source_column": record.get("source_column"),
        "raw_value": record.get("raw_value"),
        "raw_category": record.get("raw_category"),
        "verification_status": record.get("verification_status"),
        "permitted_use": record.get("permitted_use"),
    }


def build_research_assessment(
    *,
    jurisdiction: str,
    plan_area: str,
    source_status: str,
    source_records: list[dict[str, Any]],
    gis_results: dict[str, Any] | None = None,
    methodology_config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed research assessment for all five hazards."""

    methodology_config = methodology_config or {}
    gis_results = gis_results or {}
    hazard_results: dict[str, dict[str, Any]] = {}

    for hazard in HAZARDS:
        hazard_records = _records_for(source_records, hazard, plan_area, source_status)
        eligible = [record for record in hazard_records if is_record_eligible_for_research_assessment(record)]
        conflicts = detect_conflicting_records(eligible)
        context = [
            record for record in hazard_records
            if record.get("verification_status") == "context_only"
        ]
        if conflicts:
            status = "conflicting_sources"
        elif not eligible:
            status = "incomplete" if hazard_records or context else "unsupported"
        else:
            method = methodology_config.get(hazard, {}).get("methodology_name")
            status = "source_verified_method_pending" if not method else "verified_official"

        hazard_results[hazard] = {
            "hazard": hazard,
            "official_citywide_evidence": [
                _public_trace(record) for record in eligible if not record.get("plan_area")
            ],
            "official_plan_area_evidence": [
                _public_trace(record) for record in eligible if record.get("plan_area") == plan_area
            ],
            "official_scenario_evidence": [
                _public_trace(record)
                for record in eligible
                if record.get("metric_type") == "scenario_hazard_rating"
            ],
            "address_specific_mapped_findings": gis_results.get(hazard, {}),
            "community_impact_context": [_public_trace(record) for record in context],
            "proposed_area_assessment": None,
            "assessment_status": status,
            "methodology_name": methodology_config.get(hazard, {}).get("methodology_name", ""),
            "full_provenance": [_public_trace(record) for record in eligible],
            "confidence": "medium" if eligible and status != "conflicting_sources" else "low",
            "limitations": [
                "Research-only result; not connected to production.",
                "Missing or unverified source records fail closed.",
            ],
            "unresolved_conflicts": conflicts,
        }

    by_status: dict[str, int] = defaultdict(int)
    for result in hazard_results.values():
        by_status[result["assessment_status"]] += 1

    return {
        "jurisdiction": jurisdiction,
        "plan_area": plan_area,
        "source_status": source_status,
        "production_connected": False,
        "hazards": hazard_results,
        "status_counts": dict(sorted(by_status.items())),
    }
