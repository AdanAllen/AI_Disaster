#!/usr/bin/env python3
"""Run the offline LHMP candidate extractor for each enabled registry plan."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lhmp_models import PlanRegistryRecord  # noqa: E402
from scripts.lhmp_extract import extract_lhmp  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "lhmp" / "plan_registry.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "lhmp" / "batch_summary.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_project_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_registry(registry_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Plan registry must contain a JSON array.")
    return payload


def run_batch(
    registry_path: Path,
    *,
    summary_path: Path = DEFAULT_SUMMARY,
    project_root: Path = PROJECT_ROOT,
    replace_candidates: bool = False,
    extractor: Callable[..., Dict[str, Any]] = extract_lhmp,
) -> Dict[str, Any]:
    raw_records = load_registry(registry_path)
    results: List[Dict[str, Any]] = []

    for index, raw_record in enumerate(raw_records):
        fallback_city = (
            str(raw_record.get("city", "")).strip()
            if isinstance(raw_record, dict)
            else ""
        ) or f"registry-row-{index + 1}"
        try:
            plan = PlanRegistryRecord.model_validate(raw_record)
        except ValidationError as exc:
            results.append(
                {
                    "city": fallback_city,
                    "enabled": bool(
                        raw_record.get("enabled", True)
                        if isinstance(raw_record, dict)
                        else True
                    ),
                    "status": "failed",
                    "error": f"Invalid registry record: {exc}",
                }
            )
            continue

        if not plan.enabled:
            results.append(
                {
                    "city": plan.city,
                    "city_display_name": plan.city_display_name,
                    "enabled": False,
                    "status": "skipped_disabled",
                }
            )
            continue

        pdf_path = _resolve_project_path(plan.local_pdf_path, project_root)
        markdown_path = (
            _resolve_project_path(plan.markdown_path, project_root)
            if plan.markdown_path
            else None
        )
        try:
            result = extractor(
                city=plan.city,
                pdf_path=pdf_path,
                markdown_path=markdown_path,
                source_url=str(plan.source_url) if plan.source_url else "",
                project_root=project_root,
                replace_candidates=replace_candidates,
                max_render_pages=plan.max_render_pages,
                max_evidence_candidates=plan.max_evidence_candidates,
            )
            results.append(
                {
                    "city": plan.city,
                    "city_display_name": plan.city_display_name,
                    "enabled": True,
                    "status": "succeeded",
                    "pdf_path": _portable_path(pdf_path, project_root),
                    "candidate_counts": {
                        "evidence": result.get("evidence_candidates", 0),
                        "visuals": result.get("visual_candidates", 0),
                        "tables": result.get("table_candidates", 0),
                        "data_sources": result.get("data_source_candidates", 0),
                    },
                    "rendered_pages": result.get("rendered_pages", []),
                    "warnings": result.get("warnings", []),
                }
            )
        except Exception as exc:  # One plan must not stop the countywide audit.
            results.append(
                {
                    "city": plan.city,
                    "city_display_name": plan.city_display_name,
                    "enabled": True,
                    "status": "failed",
                    "pdf_path": _portable_path(pdf_path, project_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "registry_path": _portable_path(registry_path, project_root),
        "replace_candidates": replace_candidates,
        "totals": {
            "registered": len(results),
            "enabled": sum(result.get("enabled") is True for result in results),
            "disabled": sum(result.get("enabled") is False for result in results),
            "succeeded": sum(result["status"] == "succeeded" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
        },
        "results": results,
    }
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract LHMP review candidates for each enabled registry plan."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--replace-candidates",
        action="store_true",
        help="Replace existing extracted candidates for enabled plans.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_batch(
            args.registry.resolve(),
            summary_path=args.summary.resolve(),
            project_root=PROJECT_ROOT,
            replace_candidates=args.replace_candidates,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"LHMP batch extraction failed: {exc}", file=sys.stderr)
        return 1

    totals = summary["totals"]
    print(
        "LHMP batch extraction complete: "
        f"{totals['succeeded']} succeeded, {totals['failed']} failed, "
        f"{totals['disabled']} disabled. Summary: {args.summary.resolve()}"
    )
    return 0 if totals["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
