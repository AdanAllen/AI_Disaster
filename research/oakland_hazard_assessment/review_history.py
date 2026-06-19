"""Append-only table-review history for Oakland research evidence."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fingerprints import decision_event_fingerprint, evidence_fingerprint_fields


EVENTS_DIR = Path("research/oakland_hazard_assessment/table_review_decisions/events")


@dataclass
class ReviewHistory:
    events: list[dict[str, Any]]
    errors: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def latest_event(self) -> dict[str, Any] | None:
        if not self.valid or not self.events:
            return None
        return self.events[-1]


def build_decision_event(
    *,
    batch: dict[str, Any],
    decision: str,
    reviewer: str,
    reason: str,
    previous_decision_id: str = "",
    corrections: dict[str, Any] | None = None,
    reviewed_at: str | None = None,
    supersedes: list[str] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat()
    decision_id = f"{batch['batch_id']}_{timestamp.replace(':', '').replace('.', '')}_{uuid.uuid4().hex[:8]}"
    event = {
        "schema_version": 1,
        "decision_id": decision_id,
        "batch_id": batch["batch_id"],
        "previous_decision_id": previous_decision_id,
        "decision": decision,
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "reason": reason,
        "fingerprints": evidence_fingerprint_fields(batch.get("fingerprints", {})),
        "corrections": corrections or {},
        "supersedes": supersedes or ([previous_decision_id] if previous_decision_id else []),
        "superseded_by": "",
        "stale_status": "current",
    }
    if extra_fields:
        event.update(extra_fields)
    event["event_fingerprint"] = decision_event_fingerprint(event)
    return event


def event_path(events_dir: Path, event: dict[str, Any]) -> Path:
    return events_dir / event["batch_id"] / f"{event['decision_id']}.json"


def write_event_atomic(events_dir: Path, event: dict[str, Any]) -> Path:
    target = event_path(events_dir, event)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    temp = target.with_name(f".{target.name}.tmp.{uuid.uuid4().hex}")
    payload = json.dumps(event, indent=2, sort_keys=True) + "\n"
    with temp.open("x", encoding="utf-8") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, target)
    return target


def load_events(events_dir: Path, batch_id: str | None = None) -> list[dict[str, Any]]:
    root = events_dir / batch_id if batch_id else events_dir
    if not root.exists():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.rglob("*.json"))
        if not path.name.startswith(".")
    ]


def validate_history(events: list[dict[str, Any]], current_batch: dict[str, Any] | None = None) -> ReviewHistory:
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for event in events:
        decision_id = event.get("decision_id")
        if not decision_id:
            errors.append("event_missing_decision_id")
            continue
        if decision_id in by_id:
            errors.append("duplicate_decision_id")
        by_id[decision_id] = event
        expected = decision_event_fingerprint(event)
        if event.get("event_fingerprint") != expected:
            errors.append("event_fingerprint_mismatch")

    children: dict[str, list[str]] = {}
    for event in events:
        previous = event.get("previous_decision_id") or ""
        if previous:
            if previous not in by_id:
                errors.append("missing_previous_decision")
            children.setdefault(previous, []).append(event.get("decision_id", ""))

    for parent, child_ids in children.items():
        if len(child_ids) > 1:
            errors.append("conflicting_successor_decisions")

    roots = [event for event in events if not event.get("previous_decision_id")]
    if len(roots) > 1:
        errors.append("multiple_history_roots")

    sorted_events = sorted(events, key=lambda item: (item.get("reviewed_at", ""), item.get("decision_id", "")))
    if current_batch:
        current = evidence_fingerprint_fields(current_batch.get("fingerprints", {}))
        for event in sorted_events:
            if event.get("fingerprints") != current:
                event["stale_status"] = "stale"
            else:
                event["stale_status"] = event.get("stale_status") or "current"

    superseded_ids = {
        item
        for event in sorted_events
        for item in (event.get("supersedes") or [])
        if item
    }
    active = [
        event for event in sorted_events
        if event.get("stale_status") == "current"
        and not event.get("superseded_by")
        and event.get("decision_id") not in superseded_ids
    ]
    if len(active) > 1:
        errors.append("duplicate_active_decisions")
    return ReviewHistory(sorted_events, sorted(set(errors)))
