import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "data/lhmp/claim_candidates.json"


def load_queue():
    return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))


def test_local_plan_queue_covers_all_jurisdictions_without_fake_approval():
    queue = load_queue()

    assert queue["jurisdiction_count"] == 15
    assert queue["candidate_claim_count"] >= 45
    assert queue["publication_status"] == "withheld_pending_external_review"

    for jurisdiction in queue["jurisdictions"]:
        assert jurisdiction["automated_claim_candidate_count"] >= 3
        assert jurisdiction["publication_status"].startswith("withheld_")
        for claim in jurisdiction["claims"]:
            assert claim["pdf_page"] > 0
            assert 0 < claim["supporting_excerpt_word_count"] <= 22
            assert claim["automated_verification_status"] == "source_pdf_text_located"
            assert claim["human_review_status"] == "pending_external_review"
            assert claim["human_reviewer"] is None
            assert claim["human_reviewed_at"] is None
            assert "not address-level evidence" in claim["geographic_scope"]
            for field in (
                "jurisdiction",
                "plan_title",
                "edition",
                "document_type",
                "adopted_draft_status",
                "official_url",
                "exact_section",
                "exact_table_or_figure",
                "external_review_status",
            ):
                assert field in claim


def test_tri_valley_candidates_stay_inside_their_annex_ranges():
    ranges = {
        "dublin": (14, 48),
        "livermore": (49, 83),
        "pleasanton": (84, 113),
    }
    jurisdictions = {
        item["jurisdiction_id"]: item for item in load_queue()["jurisdictions"]
    }

    for jurisdiction_id, (start, end) in ranges.items():
        assert all(
            start <= claim["pdf_page"] <= end
            for claim in jurisdictions[jurisdiction_id]["claims"]
        )


def test_known_plan_source_mismatches_are_withheld():
    jurisdictions = {
        item["jurisdiction_id"]: item for item in load_queue()["jurisdictions"]
    }

    assert (
        jurisdictions["alameda"]["source_alignment"]["status"]
        == "current_document_adoption_pending"
    )
    assert (
        jurisdictions["piedmont"]["source_alignment"]["status"]
        == "adopted_with_draft_labeled_published_text"
    )
    assert (
        jurisdictions["san_leandro"]["source_alignment"]["status"]
        == "official_lhmp_pdf_unavailable"
    )
    assert jurisdictions["alameda"]["publication_status"] == "withheld_pending_external_review"
    assert jurisdictions["piedmont"]["publication_status"] == "withheld_pending_external_review"
    assert jurisdictions["san_leandro"]["publication_status"] == "withheld_source_mismatch"


def test_claim_queue_builder_validates_without_writing():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_local_plan_claim_queue.py",
            "--check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "all withheld for human review" in result.stdout


def test_automated_oakland_page_checks_are_not_human_approvals():
    evidence = json.loads(
        (ROOT / "data/hazard_priority/sub_area_evidence.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["review_gate"]["production_statuses"] == ["externally_reviewed"]
    assert evidence["review_gate"][
        "automated_source_page_checks_are_human_approval"
    ] is False
    assert all(
        record["review_status"] == "pending_external_review"
        and record["human_reviewer"] is None
        and record["human_reviewed_at"] is None
        for record in evidence["records"]
    )

    rankings = json.loads(
        (
            ROOT / "data/hazard_priority/jurisdiction_hazard_rankings.json"
        ).read_text(encoding="utf-8")
    )
    assert rankings["review_gate"]["production_statuses"] == [
        "externally_reviewed"
    ]
    assert rankings["review_gate"]["automated_extraction_is_human_approval"] is False
    assert all(
        record["review_status"] == "pending_external_review"
        and record["human_reviewer"] is None
        and record["human_reviewed_at"] is None
        for record in rankings["records"]
    )
