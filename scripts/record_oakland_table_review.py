#!/usr/bin/env python3
"""Record a human table-review decision without editing JSON by hand."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.oakland_hazard_assessment.table_review import validate_table_batch_decision
from research.oakland_hazard_assessment.review_history import (
    EVENTS_DIR,
    build_decision_event,
    load_events,
    validate_history,
    write_event_atomic,
)
from research.oakland_hazard_assessment.second_pass import required_second_pass_row_ids

RESEARCH = ROOT / "research" / "oakland_hazard_assessment"


def parse_correction(value: str) -> tuple[str, dict]:
    parts = value.split("|", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "correction must be record_id|original_extracted_value|corrected_value|correction_reason"
        )
    record_id, original, corrected, reason = parts
    return record_id, {
        "original_extracted_value": original,
        "corrected_value": corrected,
        "correction_reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an Oakland table review decision.")
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--decision", required=True, choices=[
        "approve_table_extraction",
        "approve_with_corrections",
        "context_only",
        "reject_table_for_assessment",
        "needs_more_review",
        "second_pass_approved",
        "second_pass_rejected",
    ])
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--reviewed-at", default="")
    parser.add_argument(
        "--correction",
        action="append",
        default=[],
        type=parse_correction,
        help="Record correction as record_id|original_extracted_value|corrected_value|correction_reason",
    )
    parser.add_argument(
        "--reviewed-row-id",
        action="append",
        default=[],
        help="Row id reviewed in a second-pass decision.",
    )
    args = parser.parse_args()

    batch_path = RESEARCH / "table_review_batches" / f"{args.batch_id}.json"
    if not batch_path.exists():
        raise SystemExit(f"Unknown batch id: {args.batch_id}")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    reviewed_at = args.reviewed_at or datetime.now(timezone.utc).isoformat()
    corrections = {record_id: correction for record_id, correction in args.correction}
    events_dir = ROOT / EVENTS_DIR
    history = validate_history(load_events(events_dir, args.batch_id), batch)
    if not history.valid:
        raise SystemExit(f"Existing review history is malformed: {', '.join(history.errors)}")
    previous_decision_id = history.latest_event.get("decision_id") if history.latest_event else ""
    extra = {}
    if args.decision.startswith("second_pass"):
        required = set(required_second_pass_row_ids(batch)["required_row_ids"])
        reviewed = set(args.reviewed_row_id)
        if not required.issubset(reviewed):
            missing = ", ".join(sorted(required - reviewed))
            raise SystemExit(f"Second-pass decision is missing required rows: {missing}")
        extra = {
            "reviewed_row_ids": sorted(reviewed),
            "batch_evidence_fingerprint": batch.get("fingerprints", {}).get("semantic_batch_evidence_fingerprint", ""),
            "outcome": args.decision,
        }
    event = build_decision_event(
        batch=batch,
        decision=args.decision,
        reviewer=args.reviewer,
        reason=args.reason,
        previous_decision_id=previous_decision_id,
        corrections=corrections,
        reviewed_at=reviewed_at,
        extra_fields=extra,
    )
    if not args.decision.startswith("second_pass"):
        errors = validate_table_batch_decision(batch, event, ROOT)
        if errors:
            raise SystemExit(f"Decision did not pass validation: {', '.join(errors)}")
    decision_path = write_event_atomic(events_dir, event)
    print(decision_path)


if __name__ == "__main__":
    main()
