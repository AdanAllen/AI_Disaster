import hashlib
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    ROOT
    / "data/raw/lhmps/san-leandro"
    / "san-leandro-lhmp-2026-07-01-public-review-draft.pdf"
)
EXPECTED_SHA256 = (
    "c3019c3740e5c6a82fafd2d16edb3063ef99f9894765150597d24546aac96f94"
)


def test_san_leandro_source_identity_and_draft_status():
    assert hashlib.sha256(PDF_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256

    document = fitz.open(PDF_PATH)
    try:
        assert document.page_count == 253
        cover = document.load_page(0).get_text("text")
        revisions = document.load_page(2).get_text("text")
        adoption = document.load_page(175).get_text("text")
    finally:
        document.close()

    assert "CITY OF SAN LEANDRO" in cover
    assert "LOCAL HAZARD MITIGATION" in cover
    assert "July 2026" in cover
    assert "0.8" in revisions
    assert "July 1, 2026" in revisions
    assert "public review" in revisions.lower()
    assert "PLAN APPROVAL AND" in adoption
    assert "TBD" in adoption


def test_san_leandro_registry_selects_latest_official_draft():
    registry = json.loads(
        (ROOT / "data/lhmp/plan_registry.json").read_text(encoding="utf-8")
    )
    plan = next(item for item in registry if item["city"] == "san-leandro")

    assert plan["local_pdf_path"].endswith(
        "san-leandro-lhmp-2026-07-01-public-review-draft.pdf"
    )
    assert plan["publication_date"] == "2026-07-01"
    assert plan["publication_status"] == "public_review_draft"
    assert plan["selection_reason"] == "latest_official_draft"
    assert plan["document_type"] == "local_hazard_mitigation_plan_public_review_draft"
    assert plan["pdf_page_count"] == 253
    assert plan["pdf_sha256"] == EXPECTED_SHA256
    assert "TBD" in plan["adoption_status"]
    assert all(
        term not in plan["publication_status"]
        for term in ("adopted", "fema_approved")
    )


def test_san_leandro_source_manifest_matches_selected_pdf():
    manifest = json.loads(
        (ROOT / "data/lhmp/source_manifest.json").read_text(encoding="utf-8")
    )
    source = next(
        item for item in manifest["sources"] if item["slug"] == "san-leandro"
    )

    assert source["pdf_sha256"] == EXPECTED_SHA256
    assert source["publication_status"] == "public_review_draft"
    assert source["selection_reason"] == "latest_official_draft"
    assert source["claim_publication_status"] == "withheld_pending_targeted_review"
    assert source["official_pdf_url"] is None
