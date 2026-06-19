"""Deterministic second-pass sampling for Oakland table review."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

SECOND_PASS_SEED = "oakland-adopted-priority-a-second-pass-v1"


def row_id(batch_id: str, row: dict[str, Any]) -> str:
    return "|".join([
        batch_id,
        str(row.get("plan_area", "")),
        str(row.get("candidate_record_id", "")),
    ])


def deterministic_sample_row_ids(
    batch: dict[str, Any],
    *,
    seed: str = SECOND_PASS_SEED,
    selection_date: str | None = None,
) -> dict[str, Any]:
    evidence = batch.get("fingerprints", {}).get("semantic_batch_evidence_fingerprint", "")
    rows = batch.get("extracted_rows", [])
    candidates = [row_id(batch["batch_id"], row) for row in rows]
    if not candidates:
        selected: list[str] = []
    else:
        selected = [min(candidates, key=lambda item: hashlib.sha256(f"{seed}|{evidence}|{item}".encode("utf-8")).hexdigest())]
    return {
        "seed": seed,
        "selected_row_ids": selected,
        "batch_evidence_fingerprint": evidence,
        "selection_date": selection_date or date.today().isoformat(),
    }


def required_second_pass_row_ids(batch: dict[str, Any], decision: dict[str, Any] | None = None) -> dict[str, Any]:
    required = set()
    rows = batch.get("extracted_rows", [])
    for index, row in enumerate(rows):
        rid = row_id(batch["batch_id"], row)
        category = str(row.get("hazard_risk_rating", "")).lower()
        if category in {"high", "low"}:
            required.add(rid)
        if index in {0, len(rows) - 1}:
            required.add(rid)
        if batch.get("source_page") == 480:
            required.add(rid)

    corrections = (decision or {}).get("corrections") or {}
    corrected_record_ids = set(corrections)
    for row in rows:
        if row.get("candidate_record_id") in corrected_record_ids:
            required.add(row_id(batch["batch_id"], row))

    for row in rows:
        flags = set(row.get("row_review_flags") or [])
        if flags.intersection({"missing", "shortened", "combined", "aliased", "near_table_break"}):
            required.add(row_id(batch["batch_id"], row))

    sample = deterministic_sample_row_ids(batch)
    required.update(sample["selected_row_ids"])
    return {
        **sample,
        "required_row_ids": sorted(required),
    }


def second_pass_complete(batch: dict[str, Any], decision: dict[str, Any], second_pass_event: dict[str, Any] | None) -> bool:
    if not second_pass_event:
        return False
    if second_pass_event.get("batch_id") != batch.get("batch_id"):
        return False
    if second_pass_event.get("batch_evidence_fingerprint") != batch.get("fingerprints", {}).get("semantic_batch_evidence_fingerprint"):
        return False
    required = set(required_second_pass_row_ids(batch, decision)["required_row_ids"])
    reviewed = set(second_pass_event.get("reviewed_row_ids") or [])
    return required.issubset(reviewed) and second_pass_event.get("outcome") == "second_pass_approved"

