"""Explicit human-review actions for Oakland research records."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from typing import Any

from .validators import ALLOWED_REVIEW_STATUSES, ACTIVE_VERIFICATION_STATUSES, validate_source_record


def apply_review_action(record: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Return a reviewed copy of a record after an explicit review action.

    The generator never calls this automatically. Tests use this function to
    prove that visually verified status requires an action with reviewer,
    review date, page image, and row/column provenance.
    """

    if action.get("record_id") != record.get("record_id"):
        raise ValueError("review action targets a different record")
    status = action.get("review_status")
    if status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError("unsupported review status")
    if status in ACTIVE_VERIFICATION_STATUSES and not action.get("reviewer"):
        raise ValueError("active review status requires reviewer")

    reviewed = deepcopy(record)
    reviewed["verification_status"] = status
    reviewed["review_action_id"] = action.get("review_action_id")
    reviewed["verified_by"] = action.get("reviewer", "")
    reviewed["verified_date"] = action.get("review_date") or date.today().isoformat()
    reviewed["page_image_reference"] = action.get("page_image_reference") or reviewed.get("page_image_reference", "")
    reviewed["source_row"] = action.get("source_row") or reviewed.get("source_row") or reviewed.get("row", "")
    reviewed["source_column"] = action.get("source_column") or reviewed.get("source_column") or reviewed.get("column", "")
    reviewed["source_table"] = action.get("source_table") or reviewed.get("source_table") or reviewed.get("table_or_figure_title", "")
    reviewed["permitted_use"] = action.get("permitted_use") or reviewed.get("permitted_use", "")

    if status == "corrected_after_visual_review":
        reviewed["original_extracted_value"] = action.get("original_extracted_value") or reviewed.get("raw_value") or reviewed.get("raw_category")
        reviewed["corrected_value"] = action.get("corrected_value")
        reviewed["correction_reason"] = action.get("correction_reason", "")
        if action.get("corrected_value") is not None:
            reviewed["raw_value"] = action["corrected_value"]

    errors = validate_source_record(reviewed)
    if status in ACTIVE_VERIFICATION_STATUSES and errors:
        raise ValueError(f"review action did not produce a valid active record: {errors}")
    return reviewed
