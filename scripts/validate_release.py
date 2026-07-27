#!/usr/bin/env python3
"""Deterministic release-contract checks with optional public-beta gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geospatial.registry import DatasetRegistry  # noqa: E402


EXPECTED_CITIES = {
    "Alameda",
    "Albany",
    "Berkeley",
    "Dublin",
    "Emeryville",
    "Fremont",
    "Hayward",
    "Livermore",
    "Newark",
    "Oakland",
    "Piedmont",
    "Pleasanton",
    "San Leandro",
    "Union City",
}
EXPECTED_COMMUNITIES = {
    "Ashland",
    "Castro Valley",
    "Cherryland",
    "Fairview",
    "Hayward Acres",
    "San Lorenzo",
    "Sunol",
}


def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def validate(public_beta=False):
    errors = []
    datasets = DatasetRegistry().all()
    for dataset in datasets:
        for field in (
            "authoritative_landing_url",
            "edition",
            "schema_fingerprint",
            "permitted_public_claim",
            "official_comparison_endpoints",
            "automated_verified_at",
            "known_limitations",
        ):
            if not getattr(dataset, field):
                errors.append(f"{dataset.dataset_id}: missing {field}")
        if dataset.status == "verified" and not dataset.human_reviewer:
            errors.append(f"{dataset.dataset_id}: verified without named reviewer")

    jurisdictions = load_json("jurisdictions.json")
    cities = {item["name"] for item in jurisdictions if item["type"] == "city"}
    if cities != EXPECTED_CITIES:
        errors.append("jurisdiction registry does not contain exactly the 14 incorporated cities")
    uninc = next(
        (item for item in jurisdictions if item["type"] == "unincorporated"),
        {},
    )
    if set(uninc.get("official_communities", [])) != EXPECTED_COMMUNITIES:
        errors.append("unincorporated registry does not contain the seven official communities")

    location_sources = (ROOT / "county_location_service.py").read_text(encoding="utf-8")
    if "nominatim.openstreetmap.org" in location_sources:
        errors.append("public Nominatim dependency remains")
    map_template = (ROOT / "templates/_map_reference.html").read_text(encoding="utf-8")
    if '<div class="layer-toggle' in map_template:
        errors.append("map layer controls are not native buttons")

    assets = (
        "static/vendor/bootstrap/bootstrap.min.css",
        "static/vendor/bootstrap/bootstrap.bundle.min.js",
        "static/vendor/fontawesome/css/all.min.css",
        "static/vendor/leaflet/leaflet.css",
        "static/vendor/leaflet/leaflet.js",
    )
    for relative in assets:
        if not (ROOT / relative).is_file():
            errors.append(f"missing self-hosted asset: {relative}")

    catalog = load_json("data/i18n/critical_journey.json")
    required_keys = set(catalog["required_keys"])
    if not required_keys.issubset(catalog["locales"]["en"]["messages"]):
        errors.append("English critical-journey catalog is incomplete")
    enabled_locales = {
        locale
        for locale, record in catalog["locales"].items()
        if record.get("resident_facing_enabled") is True
    }
    beta_locales = set(
        catalog.get("release_scopes", {})
        .get("initial_public_beta", {})
        .get("enabled_locales", [])
    )
    if enabled_locales != {"en"} or beta_locales != {"en"}:
        errors.append("initial public beta must expose English and no other locale")
    if catalog["locales"]["es"].get("resident_facing_enabled") is not False:
        errors.append("unreviewed Spanish must remain disabled")

    fixtures = load_json("data/location_acceptance_fixtures.json")
    fixture_cities = {
        item["jurisdiction"] for item in fixtures if not item.get("community")
    }
    fixture_communities = {
        item["community"] for item in fixtures if item.get("community")
    }
    if fixture_cities != EXPECTED_CITIES or fixture_communities != EXPECTED_COMMUNITIES:
        errors.append("location acceptance fixtures do not cover all 21 jurisdictions")

    review_cases = {}
    with (ROOT / "docs" / "evidence-review-worksheet.csv").open(
        newline="",
        encoding="utf-8",
    ) as review_file:
        for row in csv.DictReader(review_file):
            review_cases.setdefault(row["dataset_id"], set()).add(row["test_case_type"])
    required_case_types = {
        "positive",
        "negative",
        "boundary",
        "not_evaluated",
        "unavailable",
    }
    for dataset in datasets:
        if review_cases.get(dataset.dataset_id) != required_case_types:
            errors.append(
                f"{dataset.dataset_id}: evidence worksheet must contain all five case types"
            )

    fixtures = load_json("data/geospatial/evidence_gold_set_fixtures.json")
    fixture_cases = fixtures.get("cases", [])
    if len(fixture_cases) != len(datasets) * len(required_case_types):
        errors.append("mapped-evidence gold set must contain exactly 40 cases")
    for dataset in datasets:
        case_types = {
            case["test_case_type"]
            for case in fixture_cases
            if case["dataset_id"] == dataset.dataset_id
        }
        if case_types != required_case_types:
            errors.append(
                f"{dataset.dataset_id}: fixture must contain all five case types"
            )
    if fixtures.get("policy", {}).get("automated_review_is_human_approval") is not False:
        errors.append("gold-set policy must state that automation is not human approval")

    evidence_report = load_json("docs/evidence-verification-report.json")
    report_summary = evidence_report.get("summary", {})
    if report_summary.get("failed") != 0 or report_summary.get("total") != 40:
        errors.append("deterministic mapped-evidence report is not green for all 40 cases")

    plan_registry = load_json("data/lhmp/plan_registry.json")
    if len(plan_registry) != 15:
        errors.append("local-plan source registry must contain 15 selected publications")
    for plan in plan_registry:
        for field in (
            "publication_date",
            "document_type",
            "publication_status",
            "selection_reason",
            "retrieval_date",
            "limitations",
        ):
            if not plan.get(field):
                errors.append(f"{plan.get('city')}: selected plan is missing {field}")
        if plan.get("selection_reason") not in {
            "latest_official_publication",
            "latest_official_draft",
            "current_adopted_plan",
            "official_source_unresolved",
        }:
            errors.append(f"{plan.get('city')}: invalid plan selection reason")

    plan_queue = load_json("data/lhmp/claim_candidates.json")
    if plan_queue.get("jurisdiction_count") != 15:
        errors.append("local-plan claim queue must cover all 15 jurisdictions")
    for jurisdiction in plan_queue.get("jurisdictions", []):
        if jurisdiction.get("automated_claim_candidate_count", 0) < 3:
            errors.append(
                f"{jurisdiction.get('jurisdiction_id')}: fewer than three plan candidates"
            )
        for claim in jurisdiction.get("claims", []):
            if (
                claim.get("human_review_status") != "pending_targeted_review"
                or claim.get("human_reviewer") is not None
                or claim.get("human_reviewed_at") is not None
            ):
                errors.append(
                    f"{claim.get('claim_candidate_id')}: automated claim was presented "
                    "as human-reviewed"
                )
            for field in (
                "publication_date",
                "official_publication_status",
                "source_status_label",
                "source_badge",
                "selection_reason",
                "printed_page",
            ):
                if claim.get(field) in (None, ""):
                    errors.append(
                        f"{claim.get('claim_candidate_id')}: missing source-status field {field}"
                    )
            if claim.get("is_draft"):
                label_text = " ".join(
                    str(claim.get(field, ""))
                    for field in (
                        "source_status_label",
                        "source_badge",
                        "adopted_draft_status",
                    )
                ).lower()
                if "draft" not in label_text:
                    errors.append(
                        f"{claim.get('claim_candidate_id')}: draft lacks visible draft label"
                    )
                if claim.get("official_publication_status") == "adopted":
                    errors.append(
                        f"{claim.get('claim_candidate_id')}: draft was labeled adopted"
                    )

    oakland_context = load_json("data/hazard_priority/sub_area_evidence.json")
    if oakland_context.get("review_gate", {}).get(
        "automated_source_page_checks_are_human_approval"
    ) is not False:
        errors.append("Oakland source-page checks must not be labeled human approval")
    for record in oakland_context.get("records", []):
        if record.get("review_status") == "externally_reviewed" and (
            not record.get("human_reviewer") or not record.get("human_reviewed_at")
        ):
            errors.append("Oakland context marked external without a named human review")

    ranking_context = load_json(
        "data/hazard_priority/jurisdiction_hazard_rankings.json"
    )
    if ranking_context.get("review_gate", {}).get(
        "automated_extraction_is_human_approval"
    ) is not False:
        errors.append("citywide plan extraction must not be labeled human approval")
    for record in ranking_context.get("records", []):
        if record.get("review_status") == "externally_reviewed" and (
            not record.get("human_reviewer") or not record.get("human_reviewed_at")
        ):
            errors.append("citywide plan claim marked external without named review")

    release_approvals = load_json("data/release_approvals.json")
    if release_approvals.get("policy", {}).get(
        "automated_checks_are_external_approval"
    ) is not False:
        errors.append("release policy must keep automated checks separate from external approval")

    if public_beta:
        english = catalog["locales"]["en"]
        english_approval = release_approvals.get("english_critical_journey", {})
        if (
            english["review_status"] != "reviewed_for_english_public_beta"
            or english_approval.get("status") != "approved"
            or not english_approval.get("named_reviewer")
            or not english_approval.get("reviewed_at")
            or not english_approval.get("evidence_reference")
        ):
            errors.append("English critical journey lacks documented safety review")
        if any(dataset.status not in {"verified", "data_unavailable"} for dataset in datasets):
            errors.append("mapped datasets must be verified or explicitly unavailable")
        if any(
            not (
                str(jurisdiction.get("publication_status", "")).startswith("withheld_")
                or jurisdiction.get("publication_status") == "approved_for_publication"
            )
            for jurisdiction in plan_queue.get("jurisdictions", [])
        ):
            errors.append("local-plan claim publication state is not fail-closed")
        methodology = release_approvals.get("mapped_methodology", {})
        if (
            methodology.get("status") != "approved"
            or not methodology.get("named_reviewer")
            or not methodology.get("reviewed_at")
            or not methodology.get("evidence_reference")
        ):
            errors.append("mapped-data methodology lacks named external approval")
        research = (
            release_approvals.get("resident_research", {})
            .get("english_public_beta", {})
        )
        if (
            research.get("status") != "complete"
            or research.get("completed_sessions", 0)
            < research.get("minimum_sessions", 8)
            or research.get("county_regions_represented", 0)
            < research.get("minimum_county_regions", 4)
            or research.get("access_needs_sessions", 0)
            < research.get("minimum_access_needs_sessions", 3)
            or research.get("critical_comprehension_or_safety_findings_open") != 0
            or not research.get("findings_reference")
            or not research.get("approved_by")
        ):
            errors.append("English public-beta usability round is incomplete")
        accessibility = release_approvals.get("accessibility", {})
        if (
            accessibility.get("status") != "passed"
            or accessibility.get("standard") != "WCAG 2.2 AA"
            or not accessibility.get("auditor")
            or accessibility.get("critical_findings_open") != 0
            or accessibility.get("serious_findings_open") != 0
            or not accessibility.get("evidence_reference")
        ):
            errors.append("independent WCAG 2.2 AA audit approval is missing")
        operations = release_approvals.get("production_operations", {})
        if (
            operations.get("status") != "ready"
            or not operations.get("authorized_operator")
            or not all(
                operations.get(field) is True
                for field in (
                    "always_on_hosting_confirmed",
                    "shared_state_confirmed",
                    "monitoring_confirmed",
                    "rollback_confirmed",
                    "production_credentials_confirmed",
                )
            )
        ):
            errors.append("production operations and credentials are not confirmed")
        rollout = release_approvals.get("rollout", {})
        if (
            rollout.get("status") != "authorized"
            or not rollout.get("authorized_by")
            or not rollout.get("authorized_at")
            or not rollout.get("evidence_reference")
        ):
            errors.append("staged rollout has not been authorized")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-beta", action="store_true")
    args = parser.parse_args()
    errors = validate(public_beta=args.public_beta)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    if args.public_beta:
        print("Public-beta release gate passed.")
    else:
        print(
            "Technical release contracts passed. This does not grant the external "
            "human approvals required for public beta."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
