import json
from pathlib import Path

from app import app


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_initial_public_beta_exposes_only_english():
    catalog = load_json("data/i18n/critical_journey.json")
    enabled = {
        locale
        for locale, record in catalog["locales"].items()
        if record.get("resident_facing_enabled") is True
    }

    assert enabled == {"en"}
    assert (
        catalog["release_scopes"]["initial_public_beta"]["enabled_locales"]
        == ["en"]
    )
    assert catalog["locales"]["es"]["resident_facing_enabled"] is False
    assert catalog["locales"]["es"]["messages"] == {}
    assert (
        catalog["locales"]["es"]["release_milestone"]
        == "future_spanish_release"
    )


def test_site_is_visibly_labeled_english_only_public_beta():
    app.config.update(TESTING=True)
    html = app.test_client().get("/").get_data(as_text=True)

    assert "Public beta · English only" in html
    assert "English-only public beta" in html
    assert "Español" not in html
    assert "bilingual" not in html.lower()
    assert 'lang="en"' in html


def test_release_approvals_separate_beta_and_future_language_milestones():
    approvals = load_json("data/release_approvals.json")
    scopes = approvals["release_scopes"]

    assert scopes["english_public_beta"]["enabled_locales"] == ["en"]
    assert scopes["english_public_beta"]["spanish_is_launch_blocker"] is False
    assert scopes["english_public_beta"]["multilingual_support_claim_allowed"] is False
    assert scopes["future_spanish_release"]["status"] == "planned"
    assert scopes["future_spanish_release"]["enabled_locales"] == []
    assert (
        scopes["future_spanish_release"]["professional_translation_required"]
        is True
    )


def test_english_beta_research_gate_has_no_spanish_quota():
    beta_research = load_json("data/release_approvals.json")["resident_research"][
        "english_public_beta"
    ]

    assert beta_research["minimum_sessions"] == 8
    assert beta_research["minimum_county_regions"] == 4
    assert beta_research["minimum_access_needs_sessions"] == 3
    assert "spanish_preferring_sessions" not in beta_research
