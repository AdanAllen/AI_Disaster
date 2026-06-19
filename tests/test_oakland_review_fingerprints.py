import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from research.oakland_hazard_assessment.fingerprints import (
    batch_fingerprints,
    evidence_fingerprint_fields,
    fingerprint_payload,
    semantic_batch_payload,
)
from research.oakland_hazard_assessment.review_history import (
    build_decision_event,
    validate_history,
    write_event_atomic,
)
from research.oakland_hazard_assessment.second_pass import (
    deterministic_sample_row_ids,
    required_second_pass_row_ids,
)
from research.oakland_hazard_assessment.table_review import (
    records_may_be_assessment_eligible,
    validate_table_batch_decision,
)


BASE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = BASE_DIR / "research" / "oakland_hazard_assessment"


def load_first_batch():
    catalog = json.loads((RESEARCH_DIR / "adopted_priority_a_table_catalog.json").read_text(encoding="utf-8"))
    batch_id = catalog["tables"][0]["batch_id"]
    return json.loads((RESEARCH_DIR / "table_review_batches" / f"{batch_id}.json").read_text(encoding="utf-8"))


def approval_event(batch, **overrides):
    event = build_decision_event(
        batch=batch,
        decision=overrides.pop("decision", "approve_table_extraction"),
        reviewer=overrides.pop("reviewer", "human-reviewer"),
        reason=overrides.pop("reason", "Reviewed current source evidence."),
        previous_decision_id=overrides.pop("previous_decision_id", ""),
        corrections=overrides.pop("corrections", {}),
        reviewed_at=overrides.pop("reviewed_at", "2026-06-19T12:00:00Z"),
        extra_fields=overrides.pop("extra_fields", None),
    )
    event.update(overrides)
    return event


class OaklandReviewFingerprintTests(unittest.TestCase):
    def test_identical_canonical_evidence_produces_identical_fingerprints(self):
        batch = load_first_batch()
        shuffled = json.loads(json.dumps(batch))
        self.assertEqual(
            fingerprint_payload(semantic_batch_payload(batch), BASE_DIR),
            fingerprint_payload(semantic_batch_payload(shuffled), BASE_DIR),
        )

    def test_generation_timestamps_and_absolute_paths_do_not_change_semantic_fingerprint(self):
        batch = load_first_batch()
        changed = deepcopy(batch)
        changed["created_at"] = "2099-01-01"
        changed["generated_at"] = "2099-01-01"
        changed["source_url_or_local_file"] = str((BASE_DIR / changed["source_url_or_local_file"]).resolve())
        self.assertEqual(
            fingerprint_payload(semantic_batch_payload(batch), BASE_DIR),
            fingerprint_payload(semantic_batch_payload(changed), BASE_DIR),
        )

    def test_review_history_changes_do_not_alter_underlying_evidence_fingerprint(self):
        batch = load_first_batch()
        changed = deepcopy(batch)
        changed["review_history"] = [{"decision": "approve_table_extraction"}]
        changed["decisions"] = [{"reviewer": "Somebody"}]
        self.assertEqual(
            fingerprint_payload(semantic_batch_payload(batch), BASE_DIR),
            fingerprint_payload(semantic_batch_payload(changed), BASE_DIR),
        )

    def test_changed_source_values_invalidate_prior_approval(self):
        batch = load_first_batch()
        event = approval_event(batch)
        changed = deepcopy(batch)
        changed["extracted_rows"][0]["risk_ranking_score"] = 999
        changed["candidate_records"][0]["raw_value"] = 999
        changed["fingerprints"] = batch_fingerprints(changed, BASE_DIR)
        history = validate_history([event], changed)
        self.assertTrue(history.valid)
        self.assertEqual(history.latest_event["stale_status"], "stale")

    def test_changed_candidate_payload_changes_fingerprint(self):
        batch = load_first_batch()
        changed = deepcopy(batch)
        changed["candidate_records"][0]["raw_category"] = "Changed"
        self.assertNotEqual(
            batch["fingerprints"]["candidate_payload_fingerprint"],
            batch_fingerprints(changed, BASE_DIR)["candidate_payload_fingerprint"],
        )

    def test_identical_regenerated_artifacts_preserve_approval_validity(self):
        batch = load_first_batch()
        event = approval_event(batch)
        regenerated = deepcopy(batch)
        regenerated["created_at"] = "2099-01-01"
        regenerated["fingerprints"] = batch_fingerprints(regenerated, BASE_DIR)
        history = validate_history([event], regenerated)
        self.assertTrue(history.valid)
        self.assertEqual(history.latest_event["stale_status"], "current")

    def test_deterministic_second_pass_selection_is_stable(self):
        batch = load_first_batch()
        first = deterministic_sample_row_ids(batch, selection_date="2026-06-19")
        second = deterministic_sample_row_ids(batch, selection_date="2026-06-19")
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], "oakland-adopted-priority-a-second-pass-v1")

    def test_stale_evidence_requires_new_second_pass_selection(self):
        batch = load_first_batch()
        changed = deepcopy(batch)
        changed["extracted_rows"][0]["risk_ranking_score"] = 999
        changed["fingerprints"] = batch_fingerprints(changed, BASE_DIR)
        self.assertNotEqual(
            deterministic_sample_row_ids(batch, selection_date="2026-06-19")["batch_evidence_fingerprint"],
            deterministic_sample_row_ids(changed, selection_date="2026-06-19")["batch_evidence_fingerprint"],
        )

    def test_generator_outputs_no_human_review_approval(self):
        batch = load_first_batch()
        self.assertTrue(all(record["verification_status"] == "needs_more_review" for record in batch["candidate_records"]))
        decision_dir = RESEARCH_DIR / "table_review_decisions"
        event_files = list((decision_dir / "events").glob("**/*.json")) if (decision_dir / "events").exists() else []
        self.assertEqual(event_files, [])

    def test_append_only_history_cannot_overwrite_event(self):
        batch = load_first_batch()
        event = approval_event(batch)
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            write_event_atomic(events_dir, event)
            with self.assertRaises(FileExistsError):
                write_event_atomic(events_dir, event)

    def test_old_decisions_remain_after_correction_or_reversal(self):
        batch = load_first_batch()
        first = approval_event(batch, reviewed_at="2026-06-19T12:00:00Z")
        second = approval_event(
            batch,
            decision="approve_with_corrections",
            previous_decision_id=first["decision_id"],
            reviewed_at="2026-06-19T12:30:00Z",
            corrections={batch["candidate_records"][0]["record_id"]: {"original_extracted_value": "1", "corrected_value": "1", "correction_reason": "fixture"}},
        )
        history = validate_history([first, second], batch)
        self.assertTrue(history.valid)
        self.assertEqual(len(history.events), 2)
        self.assertEqual(history.events[0]["decision_id"], first["decision_id"])

    def test_malformed_review_history_fails_closed(self):
        batch = load_first_batch()
        first = approval_event(batch)
        second = approval_event(batch, previous_decision_id="missing-decision-id", reviewed_at="2026-06-19T12:30:00Z")
        history = validate_history([first, second], batch)
        self.assertFalse(history.valid)
        self.assertIn("missing_previous_decision", history.errors)

    def test_second_pass_required_records_cannot_be_assessment_eligible_without_second_pass(self):
        batch = load_first_batch()
        event = approval_event(batch)
        self.assertFalse(records_may_be_assessment_eligible(batch, event, None, BASE_DIR))

    def test_table_with_documented_fewer_than_nine_rows_is_not_automatically_rejected(self):
        batch = load_first_batch()
        reduced = deepcopy(batch)
        reduced["extracted_rows"] = reduced["extracted_rows"][:2]
        reduced["candidate_records"] = reduced["candidate_records"][:2]
        reduced["expected_row_coverage"] = {
            "coverage_type": "official_limited_rows",
            "expected_rows": [row["plan_area"] for row in reduced["extracted_rows"]],
            "missing_row_policy": "documented_allowed",
        }
        reduced["missing_expected_rows"] = []
        reduced["fingerprints"] = batch_fingerprints(reduced, BASE_DIR)
        event = approval_event(reduced)
        self.assertNotIn("table_rows_do_not_match_expected_coverage", validate_table_batch_decision(reduced, event, BASE_DIR))

    def test_absent_columns_are_not_fabricated_or_required(self):
        batch = load_first_batch()
        no_score = deepcopy(batch)
        no_score["required_value_columns"] = ["Hazard Risk Rating"]
        no_score["value_field"] = ""
        no_score["extracted_column_headers"] = [column for column in no_score["extracted_column_headers"] if column != "Risk Ranking Score"]
        no_score["fingerprints"] = batch_fingerprints(no_score, BASE_DIR)
        event = approval_event(no_score)
        errors = validate_table_batch_decision(no_score, event, BASE_DIR)
        self.assertNotIn("required_column_missing_from_source", errors)
        self.assertNotIn("candidate_value_mismatch", errors)

    def test_decision_fingerprint_captures_event_content(self):
        batch = load_first_batch()
        first = approval_event(batch, reason="First reason")
        second = approval_event(batch, reason="Second reason")
        self.assertNotEqual(first["event_fingerprint"], second["event_fingerprint"])

    def test_evidence_fingerprint_fields_exclude_review_presentation_and_decisions(self):
        batch = load_first_batch()
        fields = evidence_fingerprint_fields(batch["fingerprints"])
        self.assertIn("semantic_batch_evidence_fingerprint", fields)
        self.assertNotIn("review_presentation_fingerprint", fields)
        self.assertNotIn("review_decision_event_fingerprint", fields)


if __name__ == "__main__":
    unittest.main()
