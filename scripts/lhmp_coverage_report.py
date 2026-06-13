#!/usr/bin/env python3
"""Generate a countywide audit from non-public LHMP extraction candidates."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lhmp_models import PlanRegistryRecord  # noqa: E402

DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "lhmp" / "plan_registry.json"
DEFAULT_EXTRACTED_ROOT = PROJECT_ROOT / "data" / "lhmp" / "extracted"
DEFAULT_BATCH_SUMMARY = PROJECT_ROOT / "data" / "lhmp" / "batch_summary.json"
DEFAULT_MATRIX = PROJECT_ROOT / "data" / "lhmp" / "coverage_matrix.json"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "lhmp" / "coverage_report.html"

CANDIDATE_FILES = {
    "evidence": "evidence_candidates.json",
    "visuals": "visuals_candidates.json",
    "tables": "tables_candidates.json",
    "data_sources": "data_sources_candidates.json",
}
TIER_ORDER = {"none": 0, "basic": 1, "standard": 2, "strong": 3, "full": 4}
MITIGATION_RE = re.compile(
    r"\b(mitigat(?:e|ion)|action(?:s)?|strateg(?:y|ies)|project(?:s)?|"
    r"preparedness|response|recovery|implementation)\b",
    re.IGNORECASE,
)
RISK_PROFILE_RE = re.compile(
    r"\b(risk|vulnerab(?:ility|le)|exposure|loss(?:es)?|impact(?:s)?|"
    r"hazard profile|probability|severity)\b",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _record_text(record: Mapping[str, Any]) -> str:
    fields = (
        "section_heading",
        "original_text",
        "extracted_snippet",
        "caption",
        "extracted_text",
        "source_text",
        "extraction_reason",
    )
    return " ".join(str(record.get(field, "")) for field in fields).strip()


def _load_candidate_list(
    path: Path,
    kind: str,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    if not path.is_file():
        warnings.append(f"Missing {kind} candidate file: {path.name}")
        return []
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return []
    if not isinstance(payload, list):
        warnings.append(f"{path.name} must contain a JSON array.")
        return []
    records = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            warnings.append(f"{path.name} record {index + 1} is not an object.")
            continue
        if item.get("review_status") != "candidate":
            warnings.append(
                f"{path.name} record {index + 1} was ignored because it is not a candidate."
            )
            continue
        records.append(item)
    return records


def _load_batch_errors(batch_summary_path: Path) -> Dict[str, str]:
    if not batch_summary_path.is_file():
        return {}
    try:
        payload = _read_json(batch_summary_path)
    except (OSError, json.JSONDecodeError):
        return {}
    errors = {}
    for result in payload.get("results", []) if isinstance(payload, dict) else []:
        if (
            isinstance(result, dict)
            and result.get("status") == "failed"
            and result.get("city")
        ):
            errors[str(result["city"])] = str(result.get("error", "Extraction failed."))
    return errors


def suggest_coverage_tier(metrics: Mapping[str, int]) -> str:
    evidence = metrics.get("evidence", 0)
    visuals = metrics.get("visuals", 0)
    tables = metrics.get("tables", 0)
    sources = metrics.get("data_sources", 0)
    headings = metrics.get("section_headings", 0)
    mitigation = metrics.get("mitigation_candidates", 0)
    risk_profile = metrics.get("risk_profile_candidates", 0)
    total = evidence + visuals + tables + sources

    if total == 0:
        return "none"
    if (
        evidence >= 8
        and headings >= 1
        and mitigation >= 1
        and risk_profile >= 1
        and visuals >= 1
        and tables >= 1
        and sources >= 1
    ):
        return "full"
    if (
        evidence >= 1
        and headings >= 1
        and mitigation >= 1
        and visuals + tables >= 1
    ):
        return "strong"
    if evidence >= 3 or (evidence >= 1 and headings >= 1):
        return "standard"
    return "basic"


def _group_records(
    records_by_kind: Mapping[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Counter]:
    grouped: MutableMapping[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "records": defaultdict(list),
            "headings": set(),
            "mitigation_candidates": 0,
            "risk_profile_candidates": 0,
        }
    )
    unresolved_counts = Counter()
    unresolved_examples: List[str] = []
    all_headings = Counter()

    for kind, records in records_by_kind.items():
        for record in records:
            heading = str(record.get("section_heading", "")).strip()
            if heading:
                all_headings[heading] += 1
            hazard = str(record.get("suggested_hazard") or "").strip()
            if not hazard:
                unresolved_counts[kind] += 1
                if len(unresolved_examples) < 8:
                    example = heading or _record_text(record)
                    if example:
                        unresolved_examples.append(example[:240])
                continue
            bucket = grouped[hazard]
            bucket["records"][kind].append(record)
            if heading:
                bucket["headings"].add(heading)
            text = _record_text(record)
            if MITIGATION_RE.search(text):
                bucket["mitigation_candidates"] += 1
            if RISK_PROFILE_RE.search(text):
                bucket["risk_profile_candidates"] += 1

    hazards: Dict[str, Dict[str, Any]] = {}
    for hazard, bucket in sorted(grouped.items()):
        metrics = {
            kind: len(bucket["records"].get(kind, []))
            for kind in CANDIDATE_FILES
        }
        metrics.update(
            {
                "section_headings": len(bucket["headings"]),
                "mitigation_candidates": bucket["mitigation_candidates"],
                "risk_profile_candidates": bucket["risk_profile_candidates"],
            }
        )
        hazards[hazard] = {
            "coverage_tier": suggest_coverage_tier(metrics),
            "metrics": metrics,
            "likely_section_headings": sorted(bucket["headings"])[:20],
        }

    unresolved = {
        "total": sum(unresolved_counts.values()),
        "counts": {
            kind: unresolved_counts.get(kind, 0) for kind in CANDIDATE_FILES
        },
        "examples": list(dict.fromkeys(unresolved_examples)),
        "note": (
            "These candidates did not map through hazard_aliases.json and were not "
            "assigned a guessed hazard."
        ),
    }
    return hazards, unresolved, all_headings


def _city_coverage(
    plan: PlanRegistryRecord,
    *,
    extracted_root: Path,
    batch_errors: Mapping[str, str],
) -> Dict[str, Any]:
    base = {
        "city": plan.city,
        "city_display_name": plan.city_display_name,
        "enabled": plan.enabled,
        "plan_title": plan.plan_title or "",
        "plan_year": plan.plan_year,
        "local_pdf_path": plan.local_pdf_path,
        "source_url": str(plan.source_url) if plan.source_url else "",
        "notes": plan.notes,
    }
    if not plan.enabled:
        return {
            **base,
            "status": "disabled",
            "coverage_tier": "none",
            "candidate_counts": {kind: 0 for kind in CANDIDATE_FILES},
            "hazards_detected": [],
            "hazards": {},
            "likely_section_headings": [],
            "unresolved": {
                "total": 0,
                "counts": {kind: 0 for kind in CANDIDATE_FILES},
                "examples": [],
                "note": "Disabled registry entries are not audited.",
            },
            "warnings": [],
            "errors": [],
        }

    warnings: List[str] = []
    errors = [batch_errors[plan.city]] if plan.city in batch_errors else []
    city_directory = extracted_root / plan.city
    records_by_kind = {
        kind: _load_candidate_list(city_directory / filename, kind, warnings)
        for kind, filename in CANDIDATE_FILES.items()
    }
    manifest_path = city_directory / "plan_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            candidate_manifest = _read_json(manifest_path)
            if (
                isinstance(candidate_manifest, dict)
                and candidate_manifest.get("review_status") == "candidate"
            ):
                manifest = candidate_manifest
                warnings.extend(
                    str(item) for item in candidate_manifest.get("warnings", [])
                )
            else:
                warnings.append("Extracted manifest was ignored because it is not a candidate.")
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read extracted plan manifest: {exc}")
    else:
        warnings.append("Missing extracted plan manifest.")

    hazards, unresolved, all_headings = _group_records(records_by_kind)
    tier = max(
        (details["coverage_tier"] for details in hazards.values()),
        key=lambda value: TIER_ORDER[value],
        default="none",
    )
    candidate_counts = {
        kind: len(records_by_kind[kind]) for kind in CANDIDATE_FILES
    }
    has_candidates = any(candidate_counts.values())
    status = "extracted" if manifest or has_candidates else "missing"
    if errors:
        status = "error"

    return {
        **base,
        "status": status,
        "coverage_tier": tier,
        "candidate_counts": candidate_counts,
        "hazards_detected": list(hazards),
        "hazards": hazards,
        "likely_section_headings": [
            {"heading": heading, "candidate_count": count}
            for heading, count in all_headings.most_common(30)
        ],
        "unresolved": unresolved,
        "manifest": {
            "source_document": manifest.get("source_document", ""),
            "pdf_page_count": manifest.get("pdf_page_count"),
            "source_pdf_sha256": manifest.get("source_pdf_sha256", ""),
            "extracted_at": manifest.get("extracted_at", ""),
        }
        if manifest
        else {},
        "warnings": list(dict.fromkeys(warnings)),
        "errors": errors,
    }


def build_coverage_matrix(
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    extracted_root: Path = DEFAULT_EXTRACTED_ROOT,
    batch_summary_path: Path = DEFAULT_BATCH_SUMMARY,
) -> Dict[str, Any]:
    raw_registry = _read_json(registry_path)
    if not isinstance(raw_registry, list):
        raise ValueError("Plan registry must contain a JSON array.")
    batch_errors = _load_batch_errors(batch_summary_path)
    cities = []
    registry_errors = []
    for index, record in enumerate(raw_registry):
        try:
            plan = PlanRegistryRecord.model_validate(record)
        except ValidationError as exc:
            registry_errors.append(
                {
                    "registry_row": index + 1,
                    "city": record.get("city", "") if isinstance(record, dict) else "",
                    "error": str(exc),
                }
            )
            continue
        cities.append(
            _city_coverage(
                plan,
                extracted_root=extracted_root,
                batch_errors=batch_errors,
            )
        )

    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "registry_path": _portable_path(registry_path),
        "candidate_source_root": _portable_path(extracted_root),
        "source_boundary": (
            "This audit reads extracted candidates only. It does not read reviewed "
            "records and does not publish website content."
        ),
        "tier_definitions": {
            "none": "No useful candidate plan data was found.",
            "basic": "A hazard is mentioned, with limited candidate detail.",
            "standard": "A hazard has a likely section or several evidence candidates.",
            "strong": "A hazard has section evidence, mitigation/action material, and a table or visual.",
            "full": "Candidate coverage includes profile/risk material, mitigation, visuals, tables, and data-source leads.",
        },
        "totals": {
            "registered": len(raw_registry),
            "reported": len(cities),
            "enabled": sum(city["enabled"] for city in cities),
            "disabled": sum(not city["enabled"] for city in cities),
            "with_extracted_candidates": sum(
                city["status"] == "extracted" for city in cities
            ),
            "with_errors": sum(bool(city["errors"]) for city in cities),
        },
        "registry_errors": registry_errors,
        "cities": cities,
    }


def _format_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in counts.items())


def render_coverage_html(matrix: Mapping[str, Any]) -> str:
    city_sections = []
    for city in matrix["cities"]:
        hazard_rows = []
        for hazard, details in city["hazards"].items():
            metrics = details["metrics"]
            hazard_rows.append(
                "<tr>"
                f"<td><strong>{html.escape(hazard.replace('_', ' ').title())}</strong></td>"
                f"<td><span class=\"tier tier-{html.escape(details['coverage_tier'])}\">"
                f"{html.escape(details['coverage_tier'])}</span></td>"
                f"<td>{metrics['evidence']}</td>"
                f"<td>{metrics['visuals']}</td>"
                f"<td>{metrics['tables']}</td>"
                f"<td>{metrics['data_sources']}</td>"
                f"<td>{metrics['section_headings']}</td>"
                f"<td>{metrics['mitigation_candidates']}</td>"
                "</tr>"
            )
        if not hazard_rows:
            hazard_rows.append(
                '<tr><td colspan="8" class="muted">No mapped hazard candidates.</td></tr>'
            )

        headings = ", ".join(
            html.escape(item["heading"])
            for item in city["likely_section_headings"][:12]
        ) or "None detected"
        warnings = city["warnings"] + city["errors"]
        warning_html = "".join(
            f"<li>{html.escape(message)}</li>" for message in warnings
        ) or "<li>None</li>"
        source_link = (
            f'<a href="{html.escape(city["source_url"])}">Official source</a>'
            if city["source_url"]
            else "No source URL recorded"
        )
        unresolved = city["unresolved"]
        city_sections.append(
            f"""
            <section class="city-card">
              <div class="city-heading">
                <div>
                  <p class="eyebrow">{html.escape(city["status"])}</p>
                  <h2>{html.escape(city["city_display_name"])}</h2>
                  <p>{html.escape(city["plan_title"] or "Plan title not recorded")}
                    {f'({city["plan_year"]})' if city["plan_year"] else ''}</p>
                </div>
                <span class="tier tier-{html.escape(city["coverage_tier"])}">
                  {html.escape(city["coverage_tier"])} candidate coverage
                </span>
              </div>
              <dl>
                <dt>Local PDF</dt><dd><code>{html.escape(city["local_pdf_path"])}</code></dd>
                <dt>Source</dt><dd>{source_link}</dd>
                <dt>Candidate totals</dt><dd>{html.escape(_format_counts(city["candidate_counts"]))}</dd>
                <dt>Likely headings</dt><dd>{headings}</dd>
                <dt>Unresolved</dt><dd>{unresolved["total"]} candidate(s); aliases were not guessed.</dd>
              </dl>
              <div class="table-wrap">
                <table>
                  <thead><tr><th>Hazard</th><th>Tier</th><th>Evidence</th><th>Visuals</th>
                    <th>Tables</th><th>Sources</th><th>Headings</th><th>Actions</th></tr></thead>
                  <tbody>{''.join(hazard_rows)}</tbody>
                </table>
              </div>
              <details>
                <summary>Warnings and extraction notes ({len(warnings)})</summary>
                <ul>{warning_html}</ul>
              </details>
            </section>
            """
        )

    generated_at = html.escape(str(matrix["generated_at"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StayReady LHMP Coverage Audit</title>
  <style>
    :root {{ color-scheme: light; --ink:#17252f; --muted:#5d6b75; --line:#d9e0e4;
      --paper:#fff; --wash:#f4f7f6; --accent:#087f5b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--wash);
      font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1180px, calc(100% - 32px)); margin:0 auto; padding:40px 0 64px; }}
    h1,h2 {{ line-height:1.15; margin:0 0 8px; }}
    .intro {{ max-width:780px; color:var(--muted); margin-bottom:28px; }}
    .city-card {{ background:var(--paper); border:1px solid var(--line); border-radius:10px;
      padding:22px; margin:16px 0; box-shadow:0 1px 2px rgba(23,37,47,.04); }}
    .city-heading {{ display:flex; align-items:flex-start; justify-content:space-between;
      gap:18px; flex-wrap:wrap; }}
    .eyebrow {{ margin:0 0 5px; color:var(--accent); font-size:12px;
      font-weight:700; letter-spacing:.08em; text-transform:uppercase; }}
    dl {{ display:grid; grid-template-columns:130px 1fr; gap:7px 16px; margin:18px 0; }}
    dt {{ font-weight:700; }} dd {{ margin:0; min-width:0; overflow-wrap:anywhere; }}
    code {{ font-size:13px; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; min-width:740px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    .tier {{ display:inline-block; padding:4px 8px; border-radius:999px;
      background:#e8eeec; font-size:12px; font-weight:700; white-space:nowrap; }}
    .tier-strong,.tier-full {{ color:#075c43; background:#dff3eb; }}
    .tier-standard {{ color:#4b5f12; background:#edf3d8; }}
    .tier-basic {{ color:#76520b; background:#fff0ca; }}
    .tier-none {{ color:#5d6b75; background:#edf0f2; }}
    .muted,.intro,details {{ color:var(--muted); }}
    details {{ margin-top:16px; }} summary {{ cursor:pointer; font-weight:700; color:var(--ink); }}
    a {{ color:#066a4c; }}
    @media (max-width:640px) {{
      main {{ width:min(100% - 20px, 1180px); padding-top:24px; }}
      .city-card {{ padding:16px; }}
      dl {{ grid-template-columns:1fr; gap:2px; }}
      dd {{ margin-bottom:8px; }}
    }}
  </style>
</head>
<body>
  <main>
    <p class="eyebrow">Offline review aid</p>
    <h1>StayReady LHMP coverage audit</h1>
    <p class="intro">Candidate coverage across registered Alameda County plans. This report
      reads extracted candidate files only. Tiers are workflow suggestions, not reviewed facts
      and not public hazard conclusions. Generated {generated_at}.</p>
    {''.join(city_sections)}
  </main>
</body>
</html>
"""


def generate_coverage_report(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    extracted_root: Path = DEFAULT_EXTRACTED_ROOT,
    batch_summary_path: Path = DEFAULT_BATCH_SUMMARY,
    matrix_path: Path = DEFAULT_MATRIX,
    report_path: Path = DEFAULT_REPORT,
) -> Dict[str, Any]:
    matrix = build_coverage_matrix(
        registry_path,
        extracted_root=extracted_root,
        batch_summary_path=batch_summary_path,
    )
    _write_json(matrix_path, matrix)
    _write_text(report_path, render_coverage_html(matrix))
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an HTML and JSON audit from extracted LHMP candidates."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED_ROOT)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        matrix = generate_coverage_report(
            registry_path=args.registry.resolve(),
            extracted_root=args.extracted_root.resolve(),
            batch_summary_path=args.batch_summary.resolve(),
            matrix_path=args.output_json.resolve(),
            report_path=args.output_html.resolve(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"LHMP coverage report failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"LHMP coverage report generated for {len(matrix['cities'])} registry records: "
        f"{args.output_json.resolve()} and {args.output_html.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
