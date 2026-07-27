from unittest.mock import patch

from app import get_additional_local_hazards


def _plan(**overrides):
    plan = {
        "name": "Oakland",
        "plan_name": "Example local plan",
        "url": "https://example.gov/plan",
        "hazards": ["extreme_heat"],
        "review_status": "pending_external_review",
        "human_reviewer": None,
        "human_reviewed_at": None,
    }
    plan.update(overrides)
    return plan


def test_automation_only_local_plan_claim_is_withheld():
    with patch("app.get_local_plan_for_context", return_value=_plan()):
        assert get_additional_local_hazards({"city": "Oakland"}) == []


def test_named_external_approval_can_publish_local_plan_claim():
    approved = _plan(
        review_status="externally_reviewed",
        human_reviewer="Named emergency-management reviewer",
        human_reviewed_at="2026-07-27",
    )
    with patch("app.get_local_plan_for_context", return_value=approved):
        results = get_additional_local_hazards({"city": "Oakland"})

    assert [item["hazard_id"] for item in results] == ["extreme_heat"]
    assert "named external reviewer" in results[0]["why_shown"]
