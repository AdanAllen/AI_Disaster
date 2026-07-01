#!/usr/bin/env python3
"""Dry-run validation for Oakland table review state."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.oakland_hazard_assessment.review_history import EVENTS_DIR, load_events, validate_history
from research.oakland_hazard_assessment.second_pass import required_second_pass_row_ids, second_pass_complete

RESEARCH = ROOT / "research" / "oakland_hazard_assessment"


def main() -> None:
    batch_dir = RESEARCH / "table_review_batches"
    report = {
        "schema_version": 1,
        "mode": "dry_run",
        "production_connected": False,
        "batches": [],
        "would_activate_record_count": 0,
        "errors": [],
    }
    for path in sorted(batch_dir.glob("*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        events = load_events(ROOT / EVENTS_DIR, batch["batch_id"])
        history = validate_history(events, batch)
        batch_report = {
            "batch_id": batch["batch_id"],
            "event_count": len(events),
            "history_valid": history.valid,
            "history_errors": history.errors,
            "latest_decision": history.latest_event.get("decision") if history.latest_event else "",
            "latest_stale_status": history.latest_event.get("stale_status") if history.latest_event else "",
            "second_pass_required": required_second_pass_row_ids(batch),
            "would_activate_records": False,
        }
        if history.errors:
            report["errors"].extend([f"{batch['batch_id']}:{error}" for error in history.errors])
        if history.latest_event and history.latest_event.get("decision") in {"approve_table_extraction", "approve_with_corrections"}:
            if not second_pass_complete(batch, history.latest_event, None):
                batch_report["history_errors"].append("missing_second_pass_approval")
            elif history.latest_event.get("stale_status") == "current":
                batch_report["would_activate_records"] = True
                report["would_activate_record_count"] += len(batch.get("candidate_records", []))
        report["batches"].append(batch_report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
