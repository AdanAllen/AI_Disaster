#!/usr/bin/env python3
"""Record a human table-review decision without editing JSON by hand."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from research.oakland_hazard_assessment.table_review import validate_table_batch_decision

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "oakland_hazard_assessment"
DECISIONS = RESEARCH / "table_review_decisions"


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
    args = parser.parse_args()

    batch_path = RESEARCH / "table_review_batches" / f"{args.batch_id}.json"
    if not batch_path.exists():
        raise SystemExit(f"Unknown batch id: {args.batch_id}")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    reviewed_at = args.reviewed_at or datetime.now(timezone.utc).isoformat()
    corrections = {record_id: correction for record_id, correction in args.correction}
    decision = {
        "schema_version": 1,
        "decision_id": f"{args.batch_id}_{reviewed_at.replace(':', '').replace('.', '')}",
        "batch_id": args.batch_id,
        "decision": args.decision,
        "reviewer": args.reviewer,
        "reviewed_at": reviewed_at,
        "reason": args.reason,
        "corrections": corrections,
    }
    errors = validate_table_batch_decision(batch, decision, ROOT)
    if errors:
        raise SystemExit(f"Decision did not pass validation: {', '.join(errors)}")
    DECISIONS.mkdir(parents=True, exist_ok=True)
    decision_path = DECISIONS / f"{args.batch_id}.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(decision_path)


if __name__ == "__main__":
    main()
