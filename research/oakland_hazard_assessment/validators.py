"""Validation gates for Oakland hazard-assessment research records.

The functions in this module intentionally do not import production code. They
describe research eligibility only, and fail closed whenever provenance is
missing or source status is ambiguous.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .constants import HAZARDS, METRIC_TYPES, PLAN_AREAS, SOURCE_STATUSES

ACTIVE_VERIFICATION_STATUSES = {
    "visually_verified",
    "corrected_after_visual_review",
}
CONTEXT_ONLY_STATUS = "context_only"
INACTIVE_VERIFICATION_STATUSES = {
    "extracted_unverified",
    "rejected",
    "superseded",
    CONTEXT_ONLY_STATUS,
}


def _present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def is_record_eligible_for_research_assessment(record: dict[str, Any]) -> bool:
    """Return True only for visually verified records with complete provenance."""

    if record.get("verification_status") not in ACTIVE_VERIFICATION_STATUSES:
        return False
    if record.get("source_status") not in SOURCE_STATUSES:
        return False
    if record.get("source_status") in {"draft", "superseded"} and record.get("dataset_status") == "adopted_active":
        return False
    required = [
        "jurisdiction",
        "hazard",
        "metric_type",
        "source_document",
        "source_status",
        "source_page",
        "source_table",
        "source_row",
        "source_column",
        "raw_value",
        "page_image_reference",
        "verified_by",
        "verified_date",
        "permitted_use",
    ]
    return all(_present(record.get(field)) for field in required)


def validate_source_record(record: dict[str, Any]) -> list[str]:
    """Return validation errors. An empty list means structurally usable."""

    errors: list[str] = []
    if record.get("hazard") not in HAZARDS:
        errors.append("unsupported_hazard")
    if record.get("plan_area") and record.get("plan_area") not in PLAN_AREAS:
        errors.append("unsupported_plan_area")
    if record.get("metric_type") not in METRIC_TYPES:
        errors.append("unsupported_metric_type")
    if not _present(record.get("source_document")):
        errors.append("missing_source_document")
    if record.get("source_status") not in SOURCE_STATUSES:
        errors.append("missing_or_unsupported_source_status")
    if record.get("source_table") and not _present(record.get("source_row")):
        errors.append("table_record_missing_row")
    if record.get("metric_type") in {"EPC_context", "community_vulnerability"}:
        if record.get("permitted_use") != "context_only":
            errors.append("context_metric_has_non_context_use")
    if record.get("verification_status") in ACTIVE_VERIFICATION_STATUSES:
        if not is_record_eligible_for_research_assessment(record):
            errors.append("active_record_missing_complete_visual_provenance")
    if record.get("verification_status") == "corrected_after_visual_review":
        if not _present(record.get("original_extracted_value")):
            errors.append("corrected_record_missing_original_extracted_value")
    return errors


def detect_duplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for record in records:
        key = (
            record.get("jurisdiction"),
            record.get("source_document"),
            record.get("source_status"),
            record.get("hazard"),
            record.get("plan_area"),
            record.get("scenario"),
            record.get("metric_type"),
            record.get("source_page"),
            record.get("source_table"),
            record.get("source_row"),
            record.get("source_column"),
        )
        buckets[key].append(record.get("record_id", "unknown"))
    return [
        {"key": list(key), "record_ids": ids}
        for key, ids in buckets.items()
        if len(ids) > 1
    ]


def detect_conflicting_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    ids: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for record in records:
        if not is_record_eligible_for_research_assessment(record):
            continue
        key = (
            record.get("source_status"),
            record.get("hazard"),
            record.get("plan_area"),
            record.get("scenario"),
            record.get("metric_type"),
        )
        buckets[key].add(record.get("raw_value") or record.get("raw_category"))
        ids[key].append(record.get("record_id", "unknown"))
    return [
        {"key": list(key), "record_ids": ids[key], "values": sorted(str(v) for v in values)}
        for key, values in buckets.items()
        if len(values) > 1
    ]
