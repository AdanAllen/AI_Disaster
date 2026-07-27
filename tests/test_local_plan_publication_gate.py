from unittest.mock import patch

from app import get_additional_local_hazards


def _plan(**overrides):
    plan = {
        "name": "Oakland",
        "plan_name": "Example local plan",
        "url": "https://example.gov/plan",
        "hazards": ["extreme_heat"],
        "claim_records": [],
        "review_status": "pending_external_review",
        "human_reviewer": None,
        "human_reviewed_at": None,
        "review_evidence": None,
    }
    plan.update(overrides)
    return plan


def test_automation_only_local_plan_claim_is_withheld():
    with patch("app.get_local_plan_for_context", return_value=_plan()):
        assert get_additional_local_hazards({"city": "Oakland"}) == []


def _claim(**overrides):
    claim = {
        "hazard": "extreme_heat",
        "source_document": "example-lhmp.pdf",
        "pdf_page": 42,
        "candidate_paraphrase": "The draft discusses citywide heat planning.",
        "source_status_label": "Public Review Draft",
        "is_draft": True,
        "publication_status": "approved_for_context",
        "limitation": "Jurisdiction-level planning context only.",
    }
    claim.update(overrides)
    return claim


def test_named_external_approval_can_publish_exact_local_plan_claim():
    approved = _plan(
        review_status="externally_reviewed",
        human_reviewer="Named emergency-management reviewer",
        human_reviewed_at="2026-07-27",
        review_evidence="review/local-plan-batch-1",
        claim_records=[_claim()],
    )
    with patch("app.get_local_plan_for_context", return_value=approved):
        results = get_additional_local_hazards({"city": "Oakland"})

    assert [item["hazard_id"] for item in results] == ["extreme_heat"]
    assert "named external review" in results[0]["why_shown"]
    assert results[0]["source_status_label"] == "Public Review Draft"
    assert results[0]["source_reference"] == "example-lhmp.pdf, PDF page 42"


def test_targeted_beta_review_can_publish_a_cited_draft_without_adoption():
    reviewed = _plan(
        review_status="beta_context_reviewed",
        human_reviewer="Named English beta context reviewer",
        human_reviewed_at="2026-07-27",
        review_evidence="review/english-beta-plan-sample-1",
        claim_records=[_claim()],
    )
    with patch("app.get_local_plan_for_context", return_value=reviewed):
        results = get_additional_local_hazards({"city": "Oakland"})

    assert len(results) == 1
    assert results[0]["is_draft"] is True
    assert "targeted English-beta citation review" in results[0]["why_shown"]
    assert "Public Review Draft" in results[0]["source_status_label"]


def test_draft_claim_cannot_publish_with_an_adopted_status_label():
    reviewed = _plan(
        review_status="beta_context_reviewed",
        human_reviewer="Named English beta context reviewer",
        human_reviewed_at="2026-07-27",
        review_evidence="review/english-beta-plan-sample-1",
        claim_records=[_claim(source_status_label="Adopted")],
    )
    with patch("app.get_local_plan_for_context", return_value=reviewed):
        assert get_additional_local_hazards({"city": "Oakland"}) == []
