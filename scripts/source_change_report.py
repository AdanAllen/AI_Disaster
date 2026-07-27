#!/usr/bin/env python3
"""Compare registered geospatial provenance with current local/remote sources.

This command never changes a dataset's review status and never publishes a new
claim. It emits a review queue when checksums, schemas, service versions, or
availability change.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from geospatial.registry import DatasetRegistry  # noqa: E402


def canonical_schema(payload):
    fields = [
        {
            "name": field.get("name"),
            "type": field.get("type"),
            "length": field.get("length"),
        }
        for field in payload.get("fields", [])
    ]
    value = {
        "fields": fields,
        "geometryType": payload.get("geometryType"),
        "objectIdField": payload.get("objectIdField"),
        "maxRecordCount": payload.get("maxRecordCount"),
        "capabilities": payload.get("capabilities"),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def remote_check(dataset):
    response = requests.get(
        dataset.exact_service_or_download_url,
        params={"f": "json"},
        headers={"User-Agent": "StayReadyNow/2.0 source-monitor"},
        timeout=(3.05, 12),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error") or not payload.get("fields"):
        raise ValueError("Feature service metadata was missing or invalid")
    fingerprint = canonical_schema(payload)
    current_version = str(payload.get("currentVersion") or "")
    changes = []
    if dataset.schema_fingerprint and dataset.schema_fingerprint != fingerprint:
        changes.append("schema_fingerprint_changed")
    if dataset.service_version and dataset.service_version != current_version:
        changes.append("service_version_changed")
    return {
        "available": True,
        "schema_fingerprint": fingerprint,
        "service_version": current_version,
        "changes": changes,
    }


def local_check(dataset):
    path = PROJECT_ROOT / dataset.local_path
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw)
    feature_count = len(payload.get("features") or [])
    schema_values = {
        "geometry_types": sorted(
            {
                str((feature.get("geometry") or {}).get("type") or "")
                for feature in payload.get("features") or []
            }
        ),
        "property_fields": sorted(
            {
                key
                for feature in payload.get("features") or []
                for key in (feature.get("properties") or {})
            }
        ),
    }
    schema_fingerprint = hashlib.sha256(
        json.dumps(schema_values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    changes = []
    if checksum != dataset.sha256:
        changes.append("checksum_changed")
    if feature_count != dataset.record_count:
        changes.append("record_count_changed")
    if dataset.schema_fingerprint and dataset.schema_fingerprint != schema_fingerprint:
        changes.append("schema_fingerprint_changed")
    return {
        "available": True,
        "sha256": checksum,
        "record_count": feature_count,
        "schema_fingerprint": schema_fingerprint,
        "changes": changes,
    }


def _snapshot_check(label, snapshot):
    path = PROJECT_ROOT / snapshot["path"]
    raw = path.read_bytes()
    payload = json.loads(raw)
    changes = []
    checksum = hashlib.sha256(raw).hexdigest()
    record_count = len(payload.get("features") or [])
    if checksum != snapshot["sha256"]:
        changes.append("checksum_changed")
    if record_count != snapshot["record_count"]:
        changes.append("record_count_changed")
    return {
        "source_id": label,
        "source_type": "component_snapshot",
        "available": True,
        "sha256": checksum,
        "record_count": record_count,
        "changes": changes,
        "review_task_required": bool(changes),
    }


def supplemental_checks():
    checks = []
    import_manifest = json.loads(
        (PROJECT_ROOT / "data/geospatial/fema_calfire_import_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    checks.append(
        _snapshot_check(
            "fema_effective_panels",
            import_manifest["fema"]["panel_snapshot"],
        )
    )
    fema_changes = []
    fema_available = True
    try:
        response = requests.get(
            import_manifest["fema"]["hazard_service"],
            params={"f": "json"},
            headers={"User-Agent": "StayReadyNow/2.0 source-monitor"},
            timeout=(3.05, 20),
        )
        response.raise_for_status()
        payload = response.json()
        if (
            str(payload.get("currentVersion") or "")
            != import_manifest["fema"]["service_metadata"]["service_version"]
        ):
            fema_changes.append("service_version_changed")
    except Exception as error:
        fema_available = False
        fema_changes.append("source_unavailable")
        fema_error = type(error).__name__
    checks.append(
        {
            "source_id": "fema_nfhl_mapserver_layer_28",
            "source_type": "official_service",
            "available": fema_available,
            "service_url": import_manifest["fema"]["hazard_service"],
            "layer_id": 28,
            "resolved_archive": import_manifest["fema"]["resolved_archive"],
            "changes": fema_changes,
            "review_task_required": bool(fema_changes),
            **({"error": fema_error} if not fema_available else {}),
        }
    )
    for source_id in ("sra", "lra_recommendation", "responsibility_area"):
        component = import_manifest["calfire"][source_id]
        check = _snapshot_check(f"calfire_{source_id}", component["snapshot"])
        try:
            response = requests.get(
                component["service"],
                params={"f": "json"},
                headers={"User-Agent": "StayReadyNow/2.0 source-monitor"},
                timeout=(3.05, 20),
            )
            response.raise_for_status()
            payload = response.json()
            editing = payload.get("editingInfo") or {}
            if str(payload.get("currentVersion") or "") != component["service_version"]:
                check["changes"].append("service_version_changed")
            if (
                component.get("data_last_edit_epoch_ms") is not None
                and editing.get("dataLastEditDate")
                != component["data_last_edit_epoch_ms"]
            ):
                check["changes"].append("effective_data_edit_changed")
        except Exception as error:
            check["available"] = False
            check["changes"].append("source_unavailable")
            check["error"] = type(error).__name__
        check["review_task_required"] = bool(check["changes"])
        checks.append(check)

    adoption = json.loads(
        (
            PROJECT_ROOT / "data/geospatial/calfire_lra_adoption_matrix.json"
        ).read_text(encoding="utf-8")
    )
    adoption_changes = []
    if len(adoption.get("jurisdictions") or []) != 15:
        adoption_changes.append("jurisdiction_matrix_changed")
    checks.append(
        {
            "source_id": "calfire_lra_adoption_matrix",
            "source_type": "adoption_matrix",
            "available": True,
            "jurisdiction_count": len(adoption.get("jurisdictions") or []),
            "unresolved_adoption_count": sum(
                item.get("adoption_status") == "ordinance_not_verified"
                for item in adoption.get("jurisdictions") or []
            ),
            "changes": adoption_changes,
            "review_task_required": bool(adoption_changes),
        }
    )

    plan_manifest = json.loads(
        (PROJECT_ROOT / "data/lhmp/source_manifest.json").read_text(encoding="utf-8")
    )
    for source in plan_manifest["sources"]:
        path = PROJECT_ROOT / source["local_pdf_path"]
        try:
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            changes = (
                []
                if checksum == source["pdf_sha256"]
                else ["official_document_checksum_changed"]
            )
            available = True
        except OSError:
            checksum = ""
            changes = ["official_document_missing"]
            available = False
        checks.append(
            {
                "source_id": f"local_plan_{source['slug']}",
                "source_type": "official_document",
                "available": available,
                "sha256": checksum,
                "registered_sha256": source["pdf_sha256"],
                "official_page_url": source["official_page_url"],
                "official_pdf_url": source["official_pdf_url"],
                "publication_date": source.get("publication_date"),
                "publication_status": source.get("publication_status"),
                "selection_reason": source.get("selection_reason"),
                "adoption_status": source["adoption_status"],
                "changes": changes,
                "review_task_required": bool(changes),
            }
        )
    return checks


def build_report():
    checked_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for dataset in DatasetRegistry().all():
        entry = {
            "dataset_id": dataset.dataset_id,
            "registered_version": dataset.dataset_version,
            "registered_status": dataset.status,
            "checked_at": checked_at,
            "review_task_required": False,
        }
        try:
            current = (
                local_check(dataset)
                if dataset.source_type == "local_snapshot"
                else remote_check(dataset)
            )
            entry.update(current)
        except Exception as error:
            entry.update(
                {
                    "available": False,
                    "changes": ["source_unavailable"],
                    "error": type(error).__name__,
                }
            )
        entry["review_task_required"] = bool(entry.get("changes"))
        entries.append(entry)
    supplemental = supplemental_checks()
    return {
        "generated_at": checked_at,
        "policy": (
            "Changes require named human review. This report never updates review "
            "status or publishes claims automatically."
        ),
        "review_task_count": (
            sum(item["review_task_required"] for item in entries)
            + sum(item["review_task_required"] for item in supplemental)
        ),
        "datasets": entries,
        "supplemental_sources": supplemental,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Exit 2 when a source change or outage needs review.",
    )
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.fail_on_change and report["review_task_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
