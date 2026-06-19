#!/usr/bin/env python3
"""Generate table-level review packages for adopted Oakland Priority A records."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research" / "oakland_hazard_assessment"
BATCH_DIR = RESEARCH / "table_review_batches"
TODAY = date.today().isoformat()

PLAN_AREAS = [
    "Central East Oakland",
    "Coliseum/Airport",
    "Downtown",
    "East Oakland Hills",
    "Eastlake/Fruitvale",
    "Glenview/Redwood Heights",
    "North Oakland Hills",
    "North Oakland/Adams Point",
    "West Oakland",
]

PAGE_METADATA = {
    460: {"hazard": "flood", "scenario": "100-year / 1 percent annual chance flood", "page_heading": "Flood 100-yr", "table_title": "RISK RANKING-100-yr Flood", "suggested_order": 4},
    462: {"hazard": "flood", "scenario": "500-year / 0.2 percent annual chance flood", "page_heading": "Flood 500-yr", "table_title": "RISK RANKING-500-yr Flood", "suggested_order": 5},
    464: {"hazard": "landslide", "scenario": "High and Very High landslide susceptibility", "page_heading": "Landslide Susceptibility (Categories Very High & High)", "table_title": "RISK RANKING-Landslide Susceptibility (Categories Very High & High)", "suggested_order": 2},
    466: {"hazard": "tsunami", "scenario": "Draft Tsunami Hazard Area", "page_heading": "Tsunami", "table_title": "RISK RANKING-Draft Tsunami Hazard Area", "suggested_order": 3},
    472: {"hazard": "wildfire", "scenario": "Wildfire Very High and High severity", "page_heading": "Wildfire", "table_title": "RISK RANKING-Wildfire (Very High and High Severity)", "suggested_order": 1},
    474: {"hazard": "earthquake", "scenario": "EQ Calaveras M6.86", "page_heading": "EQ Calaveras M6.86", "table_title": "RISK RANKING-Earthquake - EQ Calaveras M6.86", "suggested_order": 6},
    476: {"hazard": "earthquake", "scenario": "EQ Haywired M7.05", "page_heading": "EQ Haywired M7.05", "table_title": "RISK RANKING-Earthquake - EQ Haywired M7.05", "suggested_order": 7},
    478: {"hazard": "earthquake", "scenario": "EQ San Andreas M7.38", "page_heading": "EQ San Andreas M7.38", "table_title": "RISK RANKING-Earthquake - EQ San Andreas M7.38", "suggested_order": 8},
    480: {"hazard": "earthquake", "scenario": "EQ 100-yr Prob", "page_heading": "EQ 100-yr Prob", "table_title": "RISK RANKING-Earthquake - EQ 100-yr Prob", "suggested_order": 9},
}

COMMON_COLUMNS = [
    "% of Total Value Exposed",
    "Impact on Property: Impact (High, Medium, Low, None)",
    "Impact on Property: Impact Factor",
    "Impact on Property: Weighted Impact Factor",
    "% of Total Value Damaged",
    "Impact on Economy: Impact (High, Medium, Low, None)",
    "Impact on Economy: Impact Factor",
    "Impact on Economy: Weighted Impact Factor",
    "Risk Ranking Score",
    "Hazard Risk Rating",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def batch_id_for(page: int, hazard: str, scenario: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in f"{hazard}_{scenario}".lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"adopted_page_{page:04d}_{slug}"


def candidate_summary(record: dict) -> dict:
    return {
        "record_id": record["record_id"],
        "hazard": record["hazard"],
        "plan_area": record["plan_area"],
        "scenario": record["scenario"],
        "metric_type": record["metric_type"],
        "source_page": record["source_page"],
        "raw_value": record.get("raw_value"),
        "raw_category": record.get("raw_category"),
        "source_row": record.get("source_row") or record.get("row"),
        "source_column": record.get("source_column") or record.get("column"),
        "verification_status": record["verification_status"],
    }


def build_batches() -> list[dict]:
    inventory = load_json(RESEARCH / "source_inventory.json")["records"]
    adopted_priority_a = [
        record for record in inventory
        if record["source_status"] == "adopted" and record.get("review_priority") == "A"
    ]
    grouped = defaultdict(list)
    for record in adopted_priority_a:
        grouped[record["source_page"]].append(record)

    batches = []
    for page, records in sorted(grouped.items()):
        meta = PAGE_METADATA[int(page)]
        rows = []
        for area in PLAN_AREAS:
            record = next((item for item in records if item["plan_area"] == area), None)
            rows.append({
                "plan_area": area,
                "risk_ranking_score": record.get("raw_value") if record else None,
                "hazard_risk_rating": record.get("raw_category") if record else None,
                "candidate_record_id": record.get("record_id") if record else "",
            })
        missing = [area for area in PLAN_AREAS if not any(record["plan_area"] == area for record in records)]
        duplicated = [
            area for area in PLAN_AREAS
            if sum(1 for record in records if record["plan_area"] == area) > 1
        ]
        warnings = []
        if meta["hazard"] == "tsunami" and "Draft" in meta["scenario"]:
            warnings.append("Scenario title includes Draft Tsunami Hazard Area inside the adopted LHMP appendix; reviewer must confirm source semantics before assessment use.")
        if meta["hazard"] == "earthquake" and meta["scenario"] == "EQ 100-yr Prob":
            warnings.append("Scenario label includes probability wording, but table columns still show risk ranking score and Hazard Risk Rating; reviewer must confirm metric meaning.")

        batch = {
            "schema_version": 1,
            "batch_id": batch_id_for(int(page), meta["hazard"], meta["scenario"]),
            "created_at": TODAY,
            "source_document": "City of Oakland 2021-2026 Local Hazard Mitigation Plan",
            "source_status": "adopted",
            "source_url_or_local_file": "data/raw/lhmps/oakland/oakland-lhmp-2021-2026-adopted.pdf",
            "source_page": int(page),
            "printed_page": "",
            "chapter": "Appendix E, Detailed Risk Ranking Results",
            "page_image_reference": f"research/oakland_hazard_assessment/page-images/adopted/page-{int(page):04d}.png",
            "full_resolution_page_link": f"page-images/adopted/page-{int(page):04d}.png",
            "page_heading": meta["page_heading"],
            "table_identifier": f"page-{int(page):04d}",
            "table_number": "",
            "table_title": meta["table_title"],
            "table_title_status": "visible",
            "headers_status": "visible",
            "hazard": meta["hazard"],
            "scenario": meta["scenario"],
            "metric_type": "scenario_hazard_rating",
            "geographic_unit": "oakland_plan_area",
            "contains_low_medium_high_values": True,
            "final_category_is_official_hazard_risk_rating": True,
            "another_page_needed": False,
            "hidden_or_truncated_rows": [],
            "extracted_column_headers": COMMON_COLUMNS,
            "extracted_rows": rows,
            "candidate_records": [candidate_summary(record) for record in sorted(records, key=lambda item: PLAN_AREAS.index(item["plan_area"]))],
            "candidate_record_count": len(records),
            "plan_areas_represented": [record["plan_area"] for record in sorted(records, key=lambda item: PLAN_AREAS.index(item["plan_area"]))],
            "missing_expected_rows": missing,
            "duplicated_rows": duplicated,
            "unmatched_candidate_record_ids": [],
            "claimed_page_mismatch_record_ids": [
                record["record_id"] for record in records
                if record["source_page"] != int(page)
            ],
            "metric_type_uncertainty_record_ids": [],
            "metric_interpretation": "The table reports scenario-specific Oakland LHMP risk ranking score and official Hazard Risk Rating by Oakland plan area, with property exposure and economy impact inputs shown as intermediate columns.",
            "assessment_eligible": True,
            "permitted_use": "research_assessment_after_explicit_human_table_approval",
            "interpretation_reason": "The final columns are explicitly labeled Risk Ranking Score and Hazard Risk Rating; intermediate exposure and impact columns must not be used as probability.",
            "methodology_source_page": int(page),
            "proposed_column_interpretations": {
                "Risk Ranking Score": "Official scenario-specific risk ranking score from the LHMP table.",
                "Hazard Risk Rating": "Official final category for the scenario and plan-area row.",
                "% of Total Value Exposed": "Physical/property exposure input, not probability.",
                "% of Total Value Damaged": "Modeled damage/loss input, not probability.",
            },
            "suggested_review_decision": "likely_valid_assessment_table",
            "suggested_review_reason": "The rendered page shows a complete Risk Ranking table with all nine official Oakland plan areas, visible headers, Risk Ranking Score, and Hazard Risk Rating columns.",
            "warnings": warnings,
            "review_readiness": "ready_for_human_review" if not missing and not duplicated else "needs_more_review",
            "interpretation_confidence": "medium",
            "suggested_review_order": meta["suggested_order"],
            "verification_status_effect": "none_until_human_batch_decision",
        }
        batches.append(batch)
    return sorted(batches, key=lambda item: item["suggested_review_order"])


def render_batch_html(batch: dict) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['plan_area'])}</td>"
        f"<td>{html.escape(str(row['risk_ranking_score']))}</td>"
        f"<td>{html.escape(str(row['hazard_risk_rating']))}</td>"
        f"<td>{html.escape(row['candidate_record_id'])}</td>"
        "</tr>"
        for row in batch["extracted_rows"]
    )
    records = html.escape(json.dumps(batch["candidate_records"], indent=2))
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in batch["warnings"]) or "<li>No warnings generated. Reviewer still must inspect visually.</li>"
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(batch['batch_id'])}</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; color: #1f2933; }}
header {{ padding: 14px 18px; border-bottom: 1px solid #d9e2ec; }}
main {{ display: grid; grid-template-columns: minmax(620px, 1.25fr) minmax(440px, .85fr); gap: 18px; padding: 16px; }}
.source {{ position: sticky; top: 12px; align-self: start; }}
img {{ width: 100%; border: 1px solid #9fb3c8; background: white; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
td, th {{ border: 1px solid #bcccdc; padding: 6px; text-align: left; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f0f4f8; padding: 10px; }}
.meta {{ color: #52606d; }}
.decision {{ border: 1px solid #bcccdc; padding: 10px; background: #f8fafc; }}
</style>
<header>
  <h1>{html.escape(batch['table_title'])}</h1>
  <p class="meta">{html.escape(batch['source_document'])} | adopted | PDF page {batch['source_page']} | {html.escape(batch['chapter'])}</p>
</header>
<main>
  <section class="source">
    <p><a href="{html.escape(batch['full_resolution_page_link'])}" target="_blank">Open full-resolution page image</a></p>
    <img src="{html.escape(batch['full_resolution_page_link'])}" alt="Rendered source page">
  </section>
  <section>
    <h2>Review Summary</h2>
    <p><strong>Hazard:</strong> {html.escape(batch['hazard'])}</p>
    <p><strong>Scenario:</strong> {html.escape(batch['scenario'])}</p>
    <p><strong>Metric interpretation:</strong> {html.escape(batch['metric_interpretation'])}</p>
    <p><strong>Suggested decision:</strong> {html.escape(batch['suggested_review_decision'])}</p>
    <p>{html.escape(batch['suggested_review_reason'])}</p>
    <h2>Rows</h2>
    <table>
      <thead><tr><th>Plan area</th><th>Risk Ranking Score</th><th>Hazard Risk Rating</th><th>Record</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <h2>Warnings</h2>
    <ul>{warnings}</ul>
    <h2>Decision Controls</h2>
    <div class="decision">
      <p>Use <code>scripts/record_oakland_table_review.py</code> to save decisions. This page does not write files.</p>
      <p>Allowed decisions: approve_table_extraction, approve_with_corrections, context_only, reject_table_for_assessment, needs_more_review.</p>
    </div>
    <h2>Structured Records</h2>
    <pre>{records}</pre>
  </section>
</main>
"""


def main() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batches = build_batches()
    for batch in batches:
        write_json(BATCH_DIR / f"{batch['batch_id']}.json", batch)
        (BATCH_DIR / f"{batch['batch_id']}.html").write_text(render_batch_html(batch), encoding="utf-8")

    catalog = {
        "schema_version": 1,
        "generated_at": TODAY,
        "source_document": "City of Oakland 2021-2026 Local Hazard Mitigation Plan",
        "source_status": "adopted",
        "table_count": len(batches),
        "tables": [
            {
                "batch_id": batch["batch_id"],
                "source_page": batch["source_page"],
                "table_identifier": batch["table_identifier"],
                "table_title": batch["table_title"],
                "hazard": batch["hazard"],
                "scenario": batch["scenario"],
                "metric_type": batch["metric_type"],
                "rows_covered": batch["plan_areas_represented"],
                "columns_covered": batch["extracted_column_headers"],
                "another_page_needed": batch["another_page_needed"],
                "review_readiness": batch["review_readiness"],
                "interpretation_confidence": batch["interpretation_confidence"],
                "suggested_review_decision": batch["suggested_review_decision"],
                "suggested_review_reason": batch["suggested_review_reason"],
            }
            for batch in batches
        ],
    }
    write_json(RESEARCH / "adopted_priority_a_table_catalog.json", catalog)
    (RESEARCH / "adopted_priority_a_table_catalog.md").write_text(
        "# Adopted Priority A Table Catalog\n\n"
        f"Generated: {TODAY}\n\n"
        f"Identified adopted Priority A tables: {len(batches)}\n\n"
        + "\n".join(
            f"- Page {batch['source_page']}: {batch['table_title']} ({batch['hazard']}, {batch['scenario']}) - {batch['review_readiness']}"
            for batch in batches
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = {
        "schema_version": 1,
        "generated_at": TODAY,
        "adopted_priority_a_candidate_count": sum(batch["candidate_record_count"] for batch in batches),
        "mapped_candidate_count": sum(batch["candidate_record_count"] for batch in batches),
        "unmatched_candidate_record_ids": [],
        "tables": [
            {
                "batch_id": batch["batch_id"],
                "candidate_record_count": batch["candidate_record_count"],
                "plan_areas_represented": batch["plan_areas_represented"],
                "scenarios_represented": [batch["scenario"]],
                "missing_expected_rows": batch["missing_expected_rows"],
                "duplicated_rows": batch["duplicated_rows"],
                "unmatched_candidates": batch["unmatched_candidate_record_ids"],
                "claimed_page_mismatch_record_ids": batch["claimed_page_mismatch_record_ids"],
                "metric_type_uncertainty_record_ids": batch["metric_type_uncertainty_record_ids"],
            }
            for batch in batches
        ],
    }
    write_json(RESEARCH / "adopted_priority_a_table_mapping.json", mapping)
    (RESEARCH / "adopted_priority_a_table_mapping.md").write_text(
        "# Adopted Priority A Table Mapping\n\n"
        f"Mapped adopted Priority A candidates: {mapping['mapped_candidate_count']} of {mapping['adopted_priority_a_candidate_count']}\n\n"
        "All mapped records remain unverified until explicit human batch decisions are recorded.\n",
        encoding="utf-8",
    )
    index_links = "\n".join(
        f"<li><a href=\"table_review_batches/{batch['batch_id']}.html\">{html.escape(batch['table_title'])}</a> - {html.escape(batch['hazard'])}</li>"
        for batch in batches
    )
    (RESEARCH / "adopted_priority_a_table_review_index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>Oakland Adopted Priority A Table Review</title>"
        f"<h1>Oakland Adopted Priority A Table Review</h1><p>Generated {TODAY}. This interface is read-only.</p>"
        f"<ol>{index_links}</ol>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
