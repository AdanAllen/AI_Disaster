from scripts.validate_release import validate


def test_technical_contracts_do_not_claim_external_approval():
    assert validate(public_beta=False) == []


def test_public_beta_gate_includes_external_and_operational_approvals():
    errors = set(validate(public_beta=True))

    assert "mapped-data methodology lacks named external approval" in errors
    assert "resident usability session requirements are incomplete" in errors
    assert "independent WCAG 2.2 AA audit approval is missing" in errors
    assert "production operations and credentials are not confirmed" in errors
    assert "staged rollout has not been authorized" in errors
