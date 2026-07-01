"""Table-level review helpers for adopted Oakland LHMP Priority A evidence."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

from .constants import PLAN_AREAS
from .fingerprints import evidence_fingerprint_fields
from .second_pass import second_pass_complete

ALLOWED_BATCH_DECISIONS = {
    "approve_table_extraction",
    "approve_with_corrections",
    "context_only",
    "reject_table_for_assessment",
    "needs_more_review",
    "second_pass_approved",
    "second_pass_rejected",
}

ACTIVE_BATCH_DECISIONS = {"approve_table_extraction", "approve_with_corrections"}


def validate_table_batch_decision(batch: dict[str, Any], decision: dict[str, Any], base_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    decision_name = decision.get("decision")
    if decision_name not in ALLOWED_BATCH_DECISIONS:
        errors.append("unsupported_batch_decision")
    if decision_name in ACTIVE_BATCH_DECISIONS:
        if not decision.get("reviewer"):
            errors.append("missing_reviewer")
        if not decision.get("reviewed_at"):
            errors.append("missing_review_timestamp")
        if batch.get("source_status") != "adopted":
            errors.append("source_is_not_adopted")
        if batch.get("another_page_needed"):
            errors.append("missing_continuation_page")
        if batch.get("table_title_status") != "visible":
            errors.append("table_title_not_visible")
        if batch.get("headers_status") != "visible":
            errors.append("headers_not_visible")
        if batch.get("hidden_or_truncated_rows"):
            errors.append("hidden_or_truncated_rows")
        if not batch.get("assessment_eligible"):
            errors.append("table_not_assessment_eligible")
        if batch.get("metric_type") in {"population_exposure", "EPC_context", "property_exposure"}:
            errors.append("context_or_exposure_metric_not_assessment_category")
        if decision.get("fingerprints") and decision.get("fingerprints") != evidence_fingerprint_fields(batch.get("fingerprints", {})):
            errors.append("source_fingerprint_mismatch")

        page_image = batch.get("page_image_reference")
        if not page_image:
            errors.append("missing_page_image_reference")
        elif base_dir and not (base_dir / page_image).exists():
            errors.append("page_image_reference_missing_on_disk")

        expected_rows = batch.get("expected_row_coverage", {}).get("expected_rows") or []
        row_names = [row.get("plan_area") for row in batch.get("extracted_rows", [])]
        if expected_rows and row_names != expected_rows:
            errors.append("table_rows_do_not_match_expected_coverage")
        missing_policy = batch.get("expected_row_coverage", {}).get("missing_row_policy", "fail_closed")
        if batch.get("missing_expected_rows") and missing_policy != "documented_allowed":
            errors.append("missing_expected_rows_not_explained")
        required_columns = batch.get("required_value_columns") or []
        visible_columns = batch.get("extracted_column_headers") or []
        for column in required_columns:
            if column not in visible_columns:
                errors.append("required_column_missing_from_source")

        for candidate in batch.get("candidate_records", []):
            if candidate.get("hazard") != batch.get("hazard"):
                errors.append("candidate_hazard_mismatch")
            if candidate.get("scenario") != batch.get("scenario"):
                errors.append("candidate_scenario_mismatch")
            if candidate.get("source_page") != batch.get("source_page"):
                errors.append("candidate_source_page_mismatch")
            if candidate.get("plan_area") not in PLAN_AREAS:
                errors.append("candidate_plan_area_mismatch")
            row = next((item for item in batch.get("extracted_rows", []) if item.get("plan_area") == candidate.get("plan_area")), None)
            if not row:
                errors.append("candidate_row_not_visible")
                continue
            value_field = batch.get("value_field", "risk_ranking_score")
            category_field = batch.get("category_field", "hazard_risk_rating")
            if value_field and str(candidate.get("raw_value")) != str(row.get(value_field)):
                errors.append("candidate_value_mismatch")
            if category_field and str(candidate.get("raw_category")) != str(row.get(category_field)):
                errors.append("candidate_category_mismatch")
    return sorted(set(errors))


def derive_record_updates_from_batch(batch: dict[str, Any], decision: dict[str, Any], base_dir: Path | None = None) -> list[dict[str, Any]]:
    errors = validate_table_batch_decision(batch, decision, base_dir)
    if errors:
        raise ValueError(f"table batch decision is invalid: {errors}")

    status = "visually_verified" if decision["decision"] == "approve_table_extraction" else "corrected_after_visual_review"
    updates: list[dict[str, Any]] = []
    corrections = decision.get("corrections") or {}
    for candidate in batch.get("candidate_records", []):
        update = deepcopy(candidate)
        row = next(row for row in batch["extracted_rows"] if row["plan_area"] == candidate["plan_area"])
        update["verification_status"] = status
        update["review_action_id"] = decision["decision_id"]
        update["approved_table_batch_id"] = batch["batch_id"]
        update["verified_by"] = decision["reviewer"]
        update["verified_date"] = decision.get("reviewed_at") or date.today().isoformat()
        update["source_table"] = batch["table_title"]
        update["source_row"] = row["plan_area"]
        update["source_column"] = "Risk Ranking Score; Hazard Risk Rating"
        update["page_image_reference"] = batch["page_image_reference"]
        update["metric_interpretation"] = batch["metric_interpretation"]
        update["permitted_use"] = batch["permitted_use"]
        correction = corrections.get(candidate["record_id"])
        if correction:
            update["original_extracted_value"] = correction.get("original_extracted_value", candidate.get("raw_value"))
            update["corrected_value"] = correction.get("corrected_value")
            update["correction_reason"] = correction.get("correction_reason", "")
            update["raw_value"] = correction.get("corrected_value", candidate.get("raw_value"))
        updates.append(update)
    return updates


def records_may_be_assessment_eligible(
    batch: dict[str, Any],
    decision: dict[str, Any],
    second_pass_event: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> bool:
    if validate_table_batch_decision(batch, decision, base_dir):
        return False
    if decision.get("decision") not in ACTIVE_BATCH_DECISIONS:
        return False
    return second_pass_complete(batch, decision, second_pass_event)
