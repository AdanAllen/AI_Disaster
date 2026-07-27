from scripts.validate_release import validate


def test_technical_contracts_do_not_claim_external_approval():
    assert validate(public_beta=False) == []


def test_public_beta_gate_includes_external_and_operational_approvals():
    errors = set(validate(public_beta=True))

    assert "English critical journey lacks documented safety review" in errors
    assert "mapped-data methodology lacks named external approval" in errors
    assert "English public-beta usability round is incomplete" in errors
    assert "independent WCAG 2.2 AA audit approval is missing" in errors
    assert "production operations and credentials are not confirmed" in errors
    assert "staged rollout has not been authorized" in errors


def test_spanish_and_withheld_plan_candidates_do_not_block_english_beta():
    errors = set(validate(public_beta=True))

    assert "Spanish critical journey lacks professional approval" not in errors
    assert "one or more local-plan jurisdictions still need claim review" not in errors
    assert "local-plan candidate queue still requires external approval" not in errors
    assert "Oakland sub-area plan context still requires external approval" not in errors
    assert "citywide plan rankings still require external approval" not in errors
