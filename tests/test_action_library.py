import unittest
from unittest.mock import patch

from flask import Flask, render_template
from pydantic import ValidationError

from action_library_service import load_action_library, publishable_actions, select_actions
from pydantic_models import ActionRecord
from resident_guidance_engine import build_resident_plan


def action_payload(**overrides):
    payload = {
        "action_id": "test_action",
        "title": "Test action",
        "instruction": "Complete a reviewed preparedness action.",
        "hazards": ["all"],
        "household_factors": [],
        "time_buckets": ["before"],
        "citation": {
            "source_id": "official_source",
            "source_name": "Official Source",
            "source_url": "https://example.gov/preparedness",
            "source_section": "Preparedness",
            "source_summary": "The official source supports this action.",
        },
        "confidence": "official_paraphrase",
        "review_status": "reviewed",
        "authority_scope": "national",
        "guidance_scope": "general",
        "trigger_type": "general",
        "applicable_jurisdictions": [],
        "required_household_factors": [],
        "excluded_household_factors": [],
        "required_evidence": {},
        "priority_category": "supplies",
        "last_source_verified": "2026-06-05",
    }
    payload.update(overrides)
    return payload


def location_context():
    return {
        "city": "Hayward",
        "county": "Alameda County",
        "address": "777 B Street, Hayward, California",
        "display_name": "777 B Street, Hayward, California",
        "zip_code": "94541",
        "precision_label": "Address point",
        "gis_status": "Address point ready for GIS checks",
        "location_result": {
            "formatted_address": "777 B Street, Hayward, California",
            "neighborhood": "",
            "lat": 37.67,
            "lon": -122.08,
        },
    }


def hazard():
    return {
        "slug": "earthquake",
        "name": "Earthquake",
        "scope": "address_level",
        "scope_label": "Address-level",
        "data_status": "checked",
        "data_status_label": "Checked",
        "exposure_level": "Unknown",
        "why_shown": "Fault proximity context was checked.",
        "limitations": ["This is not hazard-zone membership."],
        "sources": [],
        "specialized_guidance": {
            "resident_guidance": {},
            "city_context": [],
            "location_specific_context": [],
            "guidance_source_status": "local_reviewed",
            "recovery_needs": [],
        },
    }


class ActionModelTests(unittest.TestCase):
    def test_reviewed_action_requires_valid_source_url(self):
        payload = action_payload()
        payload["citation"]["source_url"] = ""
        with self.assertRaises(ValidationError):
            ActionRecord(**payload)

    def test_household_action_requires_factor(self):
        with self.assertRaises(ValidationError):
            ActionRecord(**action_payload(
                trigger_type="household",
                required_household_factors=[],
            ))

    def test_needs_source_cannot_be_reviewed(self):
        with self.assertRaises(ValidationError):
            ActionRecord(**action_payload(confidence="needs_source"))

    def test_committed_library_is_valid_and_publishable(self):
        load_action_library.cache_clear()
        records = load_action_library()
        self.assertGreaterEqual(len(records), 25)
        self.assertEqual(len(records), len(publishable_actions()))
        for record in records:
            self.assertTrue(record.displayable)
            self.assertTrue(str(record.citation.source_url).startswith("https://"))
            self.assertTrue(record.citation.source_summary)


class ActionSelectionTests(unittest.TestCase):
    def test_household_factor_selects_action_without_changing_hazard_evidence(self):
        base_plan = build_resident_plan(
            location_context(),
            [hazard()],
            session_data={},
        )
        plan = build_resident_plan(
            location_context(),
            [hazard()],
            session_data={"household_tags": ["medical", "pets"]},
        )
        self.assertEqual(
            plan["hazards"][0]["evidence_tier"],
            base_plan["hazards"][0]["evidence_tier"],
        )
        action_ids = {
            action["action_id"]
            for action in plan["household_priorities"]
        }
        self.assertIn("medical_prescription_plan", action_ids)
        self.assertIn("pet_disaster_kit", action_ids)

    def test_selected_actions_are_complete_and_cited(self):
        actions = select_actions(
            hazards=["flood"],
            time_buckets=["during"],
            city="Alameda",
            county="Alameda County",
            trigger_types=["hazard_result"],
        )
        self.assertTrue(actions)
        for action in actions:
            self.assertTrue(action.action_id)
            self.assertTrue(action.instruction)
            self.assertTrue(action.why_shown)
            self.assertTrue(action.citation.source_url)
            self.assertTrue(action.citation.source_summary)

    def test_deduplication_uses_action_id(self):
        selected = select_actions(
            hazards=["earthquake", "earthquake"],
            time_buckets=["before", "before"],
            county="Alameda County",
        )
        action_ids = [action.action_id for action in selected]
        self.assertEqual(len(action_ids), len(set(action_ids)))

    @patch("action_library_service.load_action_library")
    def test_needs_source_action_is_not_selected(self, load_library):
        load_library.return_value = [
            ActionRecord(**action_payload(
                review_status="needs_source",
                confidence="needs_source",
            ))
        ]
        self.assertEqual(
            select_actions(hazards=["all"], time_buckets=["before"]),
            [],
        )


class RenderedActionTests(unittest.TestCase):
    def test_action_source_and_reason_are_visible(self):
        app = Flask(__name__, template_folder="../templates")
        with app.app_context():
            action = select_actions(
                hazards=["earthquake"],
                time_buckets=["during"],
                trigger_types=["hazard_result"],
                limit=1,
            )[0].model_dump(mode="json")
            html = render_template("_action_macros.html")
            self.assertIsNotNone(html)

            template = app.jinja_env.from_string(
                '{% from "_action_macros.html" import action_item %}{{ action_item(action) }}'
            )
            rendered = template.render(action=action)
        self.assertIn("Source:", rendered)
        self.assertIn(action["citation"]["source_name"], rendered)
        self.assertIn(action["why_shown"], rendered)
        self.assertIn(action["citation"]["source_summary"], rendered)


if __name__ == "__main__":
    unittest.main()
