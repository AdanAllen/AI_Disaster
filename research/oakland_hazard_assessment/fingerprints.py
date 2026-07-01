"""Canonical fingerprints for Oakland table-review evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


VOLATILE_KEYS = {
    "created_at",
    "generated_at",
    "reviewed_at",
    "review_date",
    "verified_date",
    "decision",
    "decisions",
    "decision_id",
    "event_fingerprint",
    "fingerprints",
    "review_history",
    "reviewer",
    "superseded_by",
    "supersedes",
    "stale_status",
    "updated_at",
}

PRESENTATION_KEYS = {
    "full_resolution_page_link",
    "suggested_review_reason",
    "suggested_review_decision",
    "verification_status_effect",
}


def _repo_relative(value: str, base_dir: Path | None) -> str:
    text = value.replace("\\", "/")
    if not base_dir:
        return text
    try:
        path = Path(value)
        if path.is_absolute():
            return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return text
    return text


def normalize_text(value: str, base_dir: Path | None = None) -> str:
    text = _repo_relative(value, base_dir)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def canonicalize(value: Any, base_dir: Path | None = None, *, drop_keys: set[str] | None = None) -> Any:
    drop = VOLATILE_KEYS | (drop_keys or set())
    if isinstance(value, dict):
        return {
            str(key): canonicalize(item, base_dir, drop_keys=drop_keys)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in drop
        }
    if isinstance(value, list):
        return [canonicalize(item, base_dir, drop_keys=drop_keys) for item in value]
    if isinstance(value, str):
        return normalize_text(value, base_dir)
    return value


def canonical_json(value: Any, base_dir: Path | None = None, *, drop_keys: set[str] | None = None) -> str:
    return json.dumps(
        canonicalize(value, base_dir, drop_keys=drop_keys),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint_payload(value: Any, base_dir: Path | None = None, *, drop_keys: set[str] | None = None) -> str:
    return hashlib.sha256(canonical_json(value, base_dir, drop_keys=drop_keys).encode("utf-8")).hexdigest()


def fingerprint_file(path: str | Path, base_dir: Path | None = None) -> str:
    file_path = Path(path)
    if not file_path.is_absolute() and base_dir:
        file_path = base_dir / file_path
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_batch_payload(batch: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "schema_version",
        "batch_id",
        "source_document",
        "source_status",
        "source_page",
        "printed_page",
        "chapter",
        "page_heading",
        "table_identifier",
        "table_number",
        "table_title",
        "table_title_status",
        "headers_status",
        "hazard",
        "scenario",
        "metric_type",
        "geographic_unit",
        "contains_low_medium_high_values",
        "final_category_is_official_hazard_risk_rating",
        "another_page_needed",
        "hidden_or_truncated_rows",
        "extracted_column_headers",
        "extracted_rows",
        "candidate_records",
        "candidate_record_count",
        "expected_row_coverage",
        "plan_areas_represented",
        "missing_expected_rows",
        "duplicated_rows",
        "unmatched_candidate_record_ids",
        "claimed_page_mismatch_record_ids",
        "metric_type_uncertainty_record_ids",
        "metric_interpretation",
        "assessment_eligible",
        "permitted_use",
        "interpretation_reason",
        "methodology_source_page",
        "proposed_column_interpretations",
        "warnings",
        "review_readiness",
        "interpretation_confidence",
    ]
    return {key: batch.get(key) for key in keys if key in batch}


def candidate_payload(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": batch.get("batch_id"),
        "table_identifier": batch.get("table_identifier"),
        "candidate_records": batch.get("candidate_records", []),
        "extracted_rows": batch.get("extracted_rows", []),
    }


def review_presentation_payload(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": batch.get("batch_id"),
        "table_title": batch.get("table_title"),
        "source_page": batch.get("source_page"),
        "page_image_reference": batch.get("page_image_reference"),
        "extracted_column_headers": batch.get("extracted_column_headers", []),
        "extracted_rows": batch.get("extracted_rows", []),
        "candidate_records": batch.get("candidate_records", []),
        "warnings": batch.get("warnings", []),
    }


def table_identifier_payload(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": batch.get("batch_id"),
        "source_document": batch.get("source_document"),
        "source_status": batch.get("source_status"),
        "source_page": batch.get("source_page"),
        "table_identifier": batch.get("table_identifier"),
        "table_title": batch.get("table_title"),
        "hazard": batch.get("hazard"),
        "scenario": batch.get("scenario"),
    }


def batch_fingerprints(batch: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    source_pdf = batch.get("source_url_or_local_file", "")
    page_image = batch.get("page_image_reference", "")
    table_crop = batch.get("table_crop_reference", "")
    fingerprints = {
        "source_pdf_fingerprint": fingerprint_file(source_pdf, base_dir) if source_pdf else "",
        "rendered_page_fingerprint": fingerprint_file(page_image, base_dir) if page_image else "",
        "table_crop_fingerprint": fingerprint_file(table_crop, base_dir) if table_crop else None,
        "candidate_payload_fingerprint": fingerprint_payload(candidate_payload(batch), base_dir),
        "semantic_batch_evidence_fingerprint": fingerprint_payload(semantic_batch_payload(batch), base_dir),
        "review_presentation_fingerprint": fingerprint_payload(review_presentation_payload(batch), base_dir),
        "table_identifier_fingerprint": fingerprint_payload(table_identifier_payload(batch), base_dir),
    }
    batch_without_fingerprints = {key: value for key, value in batch.items() if key != "fingerprints"}
    fingerprints["batch_json_fingerprint"] = fingerprint_payload(batch_without_fingerprints, base_dir)
    return fingerprints


def decision_event_fingerprint(event: dict[str, Any], base_dir: Path | None = None) -> str:
    return fingerprint_payload(event, base_dir, drop_keys={"event_fingerprint"})


def evidence_fingerprint_fields(fingerprints: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "source_pdf_fingerprint",
        "rendered_page_fingerprint",
        "table_crop_fingerprint",
        "candidate_payload_fingerprint",
        "semantic_batch_evidence_fingerprint",
        "table_identifier_fingerprint",
    ]
    return {key: fingerprints.get(key) for key in keys}

