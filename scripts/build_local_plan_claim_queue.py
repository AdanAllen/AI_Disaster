#!/usr/bin/env python3
"""Build a citation-level, human-review queue from extracted local plan text.

This script never promotes a candidate into ``local_plans.json``. The generated
records are deliberately limited to short excerpts and neutral paraphrases so a
named reviewer can compare each item with the cited PDF page before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data/lhmp/plan_registry.json"
LOCAL_PLANS_PATH = ROOT / "local_plans.json"
OUTPUT_PATH = ROOT / "data/lhmp/claim_candidates.json"

TRI_VALLEY_PAGE_RANGES = {
    "dublin": (14, 48),
    "livermore": (49, 83),
    "pleasanton": (84, 113),
}

SOURCE_ALIGNMENT = {
    "alameda": {
        "status": "current_document_adoption_pending",
        "detail": (
            "The current official PDF is the June 2026 Final Review Draft. Its "
            "filename reports Cal OES approval, but local adoption and FEMA approval "
            "have not been verified."
        ),
    },
    "piedmont": {
        "status": "adopted_with_draft_labeled_published_text",
        "detail": (
            "The city reports Council adoption on March 2, 2026, but the official "
            "plan page still links a PDF labeled as the 2025 public draft."
        ),
    },
    "san-leandro": {
        "status": "official_lhmp_pdf_unavailable",
        "detail": (
            "The official LHMP page linked an unrelated civic-center directory on "
            "2026-07-27. The archived 2025 Environmental Hazards Element is a "
            "different document type and remains withheld."
        ),
    },
}

JURISDICTION_ID_BY_SLUG = {
    "alameda-county": "unincorporated_alameda_county",
    "san-leandro": "san_leandro",
    "union-city": "union_city",
}

HAZARD_TERMS = {
    "dam_failure": ("dam failure", "dam breach", "inundation"),
    "drought": ("drought",),
    "earthquake": ("earthquake", "seismic", "fault", "liquefaction"),
    "extreme_heat": ("extreme heat", "heat event", "cooling center"),
    "flood": ("flood", "stormwater", "coastal flooding"),
    "landslide": ("landslide", "slope failure"),
    "poor_air_quality": ("poor air quality", "smoky air", "wildfire smoke"),
    "sea_level_rise": ("sea level rise", "groundwater rise"),
    "severe_weather": ("severe weather", "extreme weather", "storm"),
    "tsunami": ("tsunami",),
    "utility_disruption": (
        "power shutoff",
        "power outage",
        "utility disruption",
        "backup power",
    ),
    "wildfire": ("wildfire", "wildland", "fire severity"),
}

NOISE_MARKERS = (
    "table of contents",
    "acknowledgement",
    "photo credit",
    "appendix a—",
    "appendix a-",
)

SUBSTANTIVE_MARKERS = (
    "vulnerab",
    "risk",
    "exposure",
    "impact",
    "damage",
    "mitigation",
    "hazard",
    "evacuat",
    "located",
    "histor",
    "action",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _candidate_hazard(record: dict[str, Any]) -> str | None:
    suggested = record.get("suggested_hazard")
    if suggested in HAZARD_TERMS:
        return str(suggested)

    text = _normalize(record.get("extracted_snippet", "")).lower()
    for hazard, terms in HAZARD_TERMS.items():
        if any(term in text for term in terms):
            return hazard
    return None


def _short_excerpt(text: str, hazard: str, limit: int = 22) -> str:
    clean = _normalize(text)
    words = clean.split()
    if len(words) <= limit:
        return clean

    lowered = clean.lower()
    position = min(
        (
            lowered.find(term)
            for term in HAZARD_TERMS[hazard]
            if lowered.find(term) >= 0
        ),
        default=0,
    )
    word_index = len(clean[:position].split())
    start = max(0, word_index - 6)
    end = min(len(words), start + limit)
    if end - start < limit:
        start = max(0, end - limit)
    excerpt = " ".join(words[start:end]).strip(" ,;:.-")
    if start:
        excerpt = f"…{excerpt}"
    if end < len(words):
        excerpt = f"{excerpt}…"
    return excerpt


def _score(record: dict[str, Any], hazard: str) -> int:
    text = _normalize(record.get("extracted_snippet", ""))
    lowered = text.lower()
    score = 0
    score += min(len(text) // 160, 5)
    score += 4 * sum(marker in lowered for marker in SUBSTANTIVE_MARKERS)
    score += 3 * sum(term in lowered for term in HAZARD_TERMS[hazard])
    score += 2 if record.get("section_heading") else 0
    score -= 12 * sum(marker in lowered for marker in NOISE_MARKERS)
    score -= 8 if lowered.count("....") > 2 else 0
    score -= 5 if int(record.get("pdf_page", 0)) <= 6 else 0
    return score


def _within_plan_scope(slug: str, page: int) -> bool:
    page_range = TRI_VALLEY_PAGE_RANGES.get(slug)
    return not page_range or page_range[0] <= page <= page_range[1]


def _pick_candidates(slug: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for record in records:
        page = int(record.get("pdf_page", 0))
        if not _within_plan_scope(slug, page):
            continue
        hazard = _candidate_hazard(record)
        text = _normalize(record.get("extracted_snippet", ""))
        if not hazard or len(text) < 80:
            continue
        ranked.append((_score(record, hazard), hazard, record))

    ranked.sort(key=lambda item: (-item[0], int(item[2]["pdf_page"]), item[2]["id"]))
    selected: list[dict[str, Any]] = []
    selected_hazards: set[str] = set()
    selected_pages: set[int] = set()

    for _, hazard, record in ranked:
        page = int(record["pdf_page"])
        if hazard in selected_hazards or page in selected_pages:
            continue
        selected.append(_to_claim_candidate(slug, hazard, record))
        selected_hazards.add(hazard)
        selected_pages.add(page)
        if len(selected) == 5:
            break

    return selected


def _to_claim_candidate(
    slug: str, hazard: str, record: dict[str, Any]
) -> dict[str, Any]:
    text = _normalize(record["extracted_snippet"])
    excerpt = _short_excerpt(text, hazard)
    claim_id = hashlib.sha256(
        f"{slug}|{record['source_document']}|{record['pdf_page']}|{hazard}|{excerpt}".encode()
    ).hexdigest()[:16]
    return {
        "claim_candidate_id": f"local_plan_claim_{claim_id}",
        "hazard": hazard,
        "claim_type": "jurisdiction_plan_context",
        "geographic_scope": (
            "jurisdiction annex only; not address-level evidence"
            if slug in TRI_VALLEY_PAGE_RANGES
            else "jurisdiction-wide plan context; not address-level evidence"
        ),
        "source_document": record["source_document"],
        "pdf_page": int(record["pdf_page"]),
        "page_label": record.get("page_label") or None,
        "section_heading": record.get("section_heading") or None,
        "supporting_excerpt": excerpt,
        "supporting_excerpt_word_count": len(excerpt.replace("…", "").split()),
        "candidate_paraphrase": (
            f"The cited plan page discusses {hazard.replace('_', ' ')} in "
            "the jurisdiction's planning context."
        ),
        "permitted_public_claim": (
            "After named human approval, present only as jurisdiction-level plan "
            "context with this edition and page citation."
        ),
        "prohibited_claims": [
            "Do not treat this candidate as address-level evidence.",
            "Do not infer a property risk rating or probability.",
            "Do not publish before a named reviewer verifies the PDF page and scope.",
        ],
        "automated_verification_status": "source_pdf_text_located",
        "human_review_status": "pending_external_review",
        "human_reviewer": None,
        "human_reviewed_at": None,
        "limitations": [
            "Machine-selected excerpt; exact meaning and context require human review.",
            "Plan status and edition must be reconfirmed against the official page.",
        ],
    }


def build_queue() -> dict[str, Any]:
    registry = _load_json(REGISTRY_PATH)
    local_plans = {
        item["jurisdiction_id"]: item for item in _load_json(LOCAL_PLANS_PATH)
    }
    jurisdictions = []

    for plan in registry:
        if not plan.get("enabled", True):
            continue
        slug = plan["city"]
        jurisdiction_id = JURISDICTION_ID_BY_SLUG.get(slug, slug)
        local_plan = local_plans[jurisdiction_id]
        extracted_path = (
            ROOT / "data/lhmp/extracted" / slug / "evidence_candidates.json"
        )
        claims = _pick_candidates(slug, _load_json(extracted_path))
        official_url = plan.get("official_pdf_url") or plan["source_url"]
        document_type = plan.get(
            "document_type",
            "local_hazard_mitigation_plan_or_annex_requires_review",
        )
        adopted_draft_status = plan.get(
            "adoption_status",
            local_plan["adoption_status"],
        )
        for claim in claims:
            claim.update(
                {
                    "jurisdiction": local_plan["name"],
                    "plan_title": plan["plan_title"],
                    "edition": str(plan["plan_year"]),
                    "document_type": document_type,
                    "adopted_draft_status": adopted_draft_status,
                    "adoption_evidence": plan.get("adoption_evidence"),
                    "official_url": official_url,
                    "exact_section": claim.get("section_heading"),
                    "exact_table_or_figure": None,
                    "external_review_status": "pending_named_claim_review",
                }
            )
        source_alignment = SOURCE_ALIGNMENT.get(
            slug,
            {
                "status": "edition_not_externally_approved",
                "detail": (
                    "The local PDF identity is recorded, but named external review is "
                    "still required before any claim is published."
                ),
            },
        )
        jurisdictions.append(
            {
                "jurisdiction_id": jurisdiction_id,
                "jurisdiction_name": local_plan["name"],
                "plan_title": plan["plan_title"],
                "plan_year": plan["plan_year"],
                "adoption_status": local_plan["adoption_status"],
                "official_page": local_plan["url"],
                "local_pdf_path": plan["local_pdf_path"],
                "source_checked_at": local_plan["source_checked_at"],
                "source_alignment": source_alignment,
                "automated_claim_candidate_count": len(claims),
                "publication_status": (
                    "withheld_source_mismatch"
                    if source_alignment["status"]
                    in {
                        "superseded_local_pdf",
                        "different_document_type",
                        "official_lhmp_pdf_unavailable",
                    }
                    else "withheld_pending_external_review"
                ),
                "tri_valley_pdf_page_range": TRI_VALLEY_PAGE_RANGES.get(slug),
                "claims": claims,
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "Machine-generated citation queue for named human review; these records "
            "are not approved public claims."
        ),
        "publication_status": "withheld_pending_external_review",
        "jurisdiction_count": len(jurisdictions),
        "candidate_claim_count": sum(
            item["automated_claim_candidate_count"] for item in jurisdictions
        ),
        "jurisdictions": jurisdictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that generation succeeds without writing the output",
    )
    args = parser.parse_args()
    result = build_queue()

    short_jurisdictions = [
        item["jurisdiction_id"]
        for item in result["jurisdictions"]
        if item["automated_claim_candidate_count"] < 3
    ]
    if short_jurisdictions:
        print(
            "Insufficient meaningful claim candidates for: "
            + ", ".join(short_jurisdictions)
        )
        return 1

    if not args.check:
        OUTPUT_PATH.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"Local-plan queue: {result['candidate_claim_count']} candidates across "
        f"{result['jurisdiction_count']} jurisdictions; all withheld for human review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
