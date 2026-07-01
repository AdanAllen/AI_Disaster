import ast
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from research.oakland_hazard_assessment import apply_review_action, build_research_assessment
from research.oakland_hazard_assessment.constants import HAZARDS, PLAN_AREAS
from research.oakland_hazard_assessment.table_review import (
    derive_record_updates_from_batch,
    validate_table_batch_decision,
)
from research.oakland_hazard_assessment.validators import (
    detect_conflicting_records,
    detect_duplicate_records,
    is_record_eligible_for_research_assessment,
    source_page_reference_resolves,
    validate_source_record,
)


BASE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = BASE_DIR / "research" / "oakland_hazard_assessment"


def load_research_json(name):
    return json.loads((RESEARCH_DIR / name).read_text(encoding="utf-8"))


def verified_record(**overrides):
    record = {
        "record_id": "verified-test-record",
        "jurisdiction": "oakland",
        "hazard": "earthquake",
        "plan_area": "Downtown",
        "scenario": "test scenario",
        "metric_type": "scenario_hazard_rating",
        "raw_value": "36",
        "raw_category": "High",
        "source_document": "City of Oakland 2021-2026 Local Hazard Mitigation Plan",
        "source_status": "adopted",
        "source_page": 474,
        "printed_page": "Appendix E",
        "source_table": "Table RISK RANKING-Earthquake",
        "source_row": "Downtown",
        "source_column": "Risk Rating",
        "page_image_reference": "data/lhmp/extracted/oakland/page-images/page-0474.png",
        "verification_status": "visually_verified",
        "verified_by": "human-reviewer",
        "verified_date": "2026-06-19",
        "permitted_use": "research_assessment",
        "review_action_id": "review-action-001",
    }
    record.update(overrides)
    return record


def load_batch():
    catalog = load_research_json("adopted_priority_a_table_catalog.json")
    first_id = catalog["tables"][0]["batch_id"]
    return json.loads((RESEARCH_DIR / "table_review_batches" / f"{first_id}.json").read_text(encoding="utf-8"))


def valid_batch_decision(batch):
    return {
        "decision_id": f"{batch['batch_id']}-test-approval",
        "batch_id": batch["batch_id"],
        "decision": "approve_table_extraction",
        "reviewer": "human-reviewer",
        "reviewed_at": "2026-06-19T12:00:00Z",
        "reason": "Reviewed rendered table, headers, rows, and values.",
        "corrections": {},
    }


class OaklandHazardAssessmentResearchTests(unittest.TestCase):
    def test_generated_phase1_artifacts_exist(self):
        required = [
            "existing_data_audit.json",
            "existing_data_audit.md",
            "source_inventory.json",
            "source_inventory.md",
            "visual_verification_queue.json",
            "visual_verification_review.html",
            "verified_adopted_matrix.json",
            "verified_draft_matrix.json",
            "matrix_coverage_report.json",
            "matrix_coverage_report.md",
            "source_page_catalog.json",
            "triage_report.json",
            "triage_report.md",
            "adopted_draft_comparison.json",
            "adopted_draft_comparison.md",
            "manual_spot_check_plan.json",
            "manual_spot_check_plan.md",
            "adopted_priority_a_table_catalog.json",
            "adopted_priority_a_table_catalog.md",
            "adopted_priority_a_table_mapping.json",
            "adopted_priority_a_table_mapping.md",
            "adopted_priority_a_table_review_index.html",
            "methodology_report.json",
            "methodology_report.md",
            "plan_area_geometry_validation.json",
            "gis_source_validation.json",
            "shadow_report.json",
            "observatory_ave_diagnostic.json",
            "human_review_report.md",
        ]
        for name in required:
            self.assertTrue((RESEARCH_DIR / name).exists(), name)

    def test_adopted_table_batches_identify_pages_and_map_candidates(self):
        catalog = load_research_json("adopted_priority_a_table_catalog.json")
        mapping = load_research_json("adopted_priority_a_table_mapping.json")
        self.assertEqual(catalog["table_count"], 9)
        self.assertEqual(mapping["adopted_priority_a_candidate_count"], 81)
        self.assertEqual(mapping["mapped_candidate_count"], 81)
        self.assertEqual(mapping["unmatched_candidate_record_ids"], [])
        self.assertEqual(
            [table["hazard"] for table in catalog["tables"]],
            ["wildfire", "landslide", "tsunami", "flood", "flood", "earthquake", "earthquake", "earthquake", "earthquake"],
        )
        for table in catalog["tables"]:
            self.assertEqual(table["review_readiness"], "ready_for_human_review")
            self.assertIn("risk ranking", table["table_title"].lower())

    def test_table_review_packages_exist_for_every_catalog_entry(self):
        catalog = load_research_json("adopted_priority_a_table_catalog.json")
        for table in catalog["tables"]:
            json_path = RESEARCH_DIR / "table_review_batches" / f"{table['batch_id']}.json"
            html_path = RESEARCH_DIR / "table_review_batches" / f"{table['batch_id']}.html"
            self.assertTrue(json_path.exists(), json_path)
            self.assertTrue(html_path.exists(), html_path)
            batch = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(batch["extracted_rows"]), 9)
            self.assertEqual([row["plan_area"] for row in batch["extracted_rows"]], PLAN_AREAS)
            self.assertTrue((BASE_DIR / batch["page_image_reference"]).exists())

    def test_batch_suggestions_do_not_verify_records(self):
        batch = load_batch()
        self.assertEqual(batch["suggested_review_decision"], "likely_valid_assessment_table")
        self.assertEqual(batch["verification_status_effect"], "none_until_human_batch_decision")
        self.assertTrue(all(record["verification_status"] == "needs_more_review" for record in batch["candidate_records"]))

    def test_production_code_does_not_import_research_assessment(self):
        production_roots = [
            BASE_DIR / "app.py",
            BASE_DIR / "hazard_engine.py",
            BASE_DIR / "hazard_priority.py",
            BASE_DIR / "resident_guidance_engine.py",
            BASE_DIR / "templates",
        ]
        offenders = []
        for root in production_roots:
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in paths:
                if path.suffix not in {".py", ".html"}:
                    continue
                text = path.read_text(encoding="utf-8")
                if "oakland_hazard_assessment" in text:
                    offenders.append(str(path.relative_to(BASE_DIR)))
        self.assertEqual(offenders, [])

    def test_existing_manual_or_generated_values_are_not_trusted(self):
        audit = load_research_json("existing_data_audit.json")["records"]
        scenario_records = [
            record for record in audit
            if record["file"].endswith("oakland_lhmp_area_scenario_ratings.json")
        ]
        self.assertEqual(len(scenario_records), 81)
        self.assertTrue(all(record["supported"] is False for record in scenario_records))
        self.assertTrue(all(record["used_by_production"] is False for record in scenario_records))
        self.assertTrue(all(record["legacy_value_must_not_drive_research"] for record in scenario_records))

    def test_triage_classifies_all_candidates_without_self_verification(self):
        inventory = load_research_json("source_inventory.json")["records"]
        triage = load_research_json("triage_report.json")["counts"]
        self.assertEqual(sum(triage["by_priority"].values()), len(inventory))
        self.assertEqual(triage["by_priority"]["A"], 96)
        self.assertEqual(triage["by_priority"]["B"], 95)
        self.assertEqual(triage["by_priority"]["C"], 19)
        self.assertEqual(triage["by_priority"]["E"], 131)
        self.assertNotIn("visually_verified", triage["by_verification_status"])
        self.assertNotIn("corrected_after_visual_review", triage["by_verification_status"])

    def test_adopted_priority_a_records_have_page_images_but_remain_review_pending(self):
        inventory = load_research_json("source_inventory.json")["records"]
        source_pages = load_research_json("source_page_catalog.json")["pages"]
        adopted_priority_a = [
            record for record in inventory
            if record["source_status"] == "adopted" and record["review_priority"] == "A"
        ]
        self.assertEqual(len(adopted_priority_a), 81)
        self.assertTrue(all(record["verification_status"] == "needs_more_review" for record in adopted_priority_a))
        self.assertTrue(all(record["page_image_reference"] for record in adopted_priority_a))
        self.assertTrue(all(source_page_reference_resolves(record, source_pages) for record in adopted_priority_a))
        self.assertTrue(all((BASE_DIR / record["page_image_reference"]).exists() for record in adopted_priority_a))

    def test_only_explicit_review_actions_can_create_visually_verified_records(self):
        candidate = verified_record(
            verification_status="needs_more_review",
            verified_by="",
            verified_date="",
            review_action_id="",
            permitted_use="requires_visual_verification_before_assessment",
        )
        self.assertFalse(is_record_eligible_for_research_assessment(candidate))
        reviewed = apply_review_action(candidate, {
            "review_action_id": "review-action-verified-001",
            "record_id": candidate["record_id"],
            "review_status": "visually_verified",
            "reviewer": "human-reviewer",
            "review_date": "2026-06-19",
            "page_image_reference": candidate["page_image_reference"],
            "source_row": "Downtown",
            "source_column": "Risk Ranking Score; Hazard Risk Rating",
            "source_table": candidate["source_table"],
            "permitted_use": "research_assessment",
        })
        self.assertTrue(is_record_eligible_for_research_assessment(reviewed))
        self.assertEqual(reviewed["review_action_id"], "review-action-verified-001")

    def test_records_without_complete_source_page_table_row_provenance_are_ineligible(self):
        record = verified_record(source_row="", source_column="")
        self.assertFalse(is_record_eligible_for_research_assessment(record))
        self.assertIn("active_record_missing_complete_visual_provenance", validate_source_record(record))

    def test_missing_review_action_id_prevents_visual_verification(self):
        record = verified_record(review_action_id="")
        self.assertFalse(is_record_eligible_for_research_assessment(record))
        self.assertIn("active_record_missing_complete_visual_provenance", validate_source_record(record))

    def test_adopted_and_draft_sources_are_distinguishable(self):
        inventory = load_research_json("source_inventory.json")["records"]
        statuses = {record["source_status"] for record in inventory}
        self.assertIn("adopted", statuses)
        self.assertIn("draft", statuses)
        matrix_statuses = {
            load_research_json("verified_adopted_matrix.json")["source_status"],
            load_research_json("verified_draft_matrix.json")["source_status"],
        }
        self.assertEqual(matrix_statuses, {"adopted", "draft"})

    def test_rejected_superseded_and_context_only_records_cannot_become_active(self):
        for status in ["rejected", "superseded", "context_only", "extracted_unverified"]:
            self.assertFalse(is_record_eligible_for_research_assessment(verified_record(verification_status=status)))
        context_record = verified_record(metric_type="EPC_context", permitted_use="research_assessment")
        self.assertIn("context_metric_has_non_context_use", validate_source_record(context_record))

    def test_missing_page_image_prevents_visual_verification(self):
        record = verified_record(page_image_reference="")
        self.assertFalse(is_record_eligible_for_research_assessment(record))
        self.assertIn("active_record_missing_complete_visual_provenance", validate_source_record(record))

    def test_corrected_records_preserve_original_extracted_value(self):
        missing_original = verified_record(verification_status="corrected_after_visual_review")
        self.assertIn("corrected_record_missing_original_extracted_value", validate_source_record(missing_original))
        corrected = verified_record(
            verification_status="corrected_after_visual_review",
            original_extracted_value="Medium",
            raw_category="High",
        )
        self.assertNotIn("corrected_record_missing_original_extracted_value", validate_source_record(corrected))

    def test_apply_corrected_review_action_preserves_original_value(self):
        candidate = verified_record(
            verification_status="needs_more_review",
            raw_value="34",
            review_action_id="",
            verified_by="",
            verified_date="",
        )
        reviewed = apply_review_action(candidate, {
            "review_action_id": "review-action-corrected-001",
            "record_id": candidate["record_id"],
            "review_status": "corrected_after_visual_review",
            "reviewer": "human-reviewer",
            "review_date": "2026-06-19",
            "original_extracted_value": "34",
            "corrected_value": "36",
            "correction_reason": "High category was in the adjacent official column.",
            "page_image_reference": candidate["page_image_reference"],
            "source_row": candidate["source_row"],
            "source_column": candidate["source_column"],
            "source_table": candidate["source_table"],
            "permitted_use": "research_assessment",
        })
        self.assertEqual(reviewed["original_extracted_value"], "34")
        self.assertEqual(reviewed["raw_value"], "36")
        self.assertTrue(is_record_eligible_for_research_assessment(reviewed))

    def test_duplicate_and_conflicting_records_are_detected(self):
        first = verified_record(record_id="a", raw_value="36")
        duplicate = verified_record(record_id="b", raw_value="36")
        conflict = verified_record(record_id="c", raw_value="12")
        self.assertEqual(len(detect_duplicate_records([first, duplicate])), 1)
        self.assertEqual(len(detect_conflicting_records([first, conflict])), 1)

    def test_matrix_covers_all_plan_areas_and_hazards_with_explicit_missing_cells(self):
        adopted = load_research_json("verified_adopted_matrix.json")["cells"]
        draft = load_research_json("verified_draft_matrix.json")["cells"]
        for cells in (adopted, draft):
            self.assertEqual(len(cells), len(PLAN_AREAS) * len(HAZARDS))
            self.assertEqual({cell["plan_area"] for cell in cells}, set(PLAN_AREAS))
            self.assertEqual({cell["hazard"] for cell in cells}, set(HAZARDS))
            self.assertTrue(all("missing_evidence" in cell for cell in cells))
            self.assertTrue(all(cell["future_user_facing_assessment_supportable"] is False for cell in cells))

    def test_research_engine_fails_closed_without_verified_records(self):
        inventory = load_research_json("source_inventory.json")["records"]
        result = build_research_assessment(
            jurisdiction="oakland",
            plan_area="Downtown",
            source_status="adopted",
            source_records=inventory,
            gis_results={},
            methodology_config={},
        )
        self.assertFalse(result["production_connected"])
        for hazard in HAZARDS:
            self.assertIn(result["hazards"][hazard]["assessment_status"], {"incomplete", "unsupported"})
            self.assertEqual(result["hazards"][hazard]["proposed_area_assessment"], None)

    def test_research_engine_requires_hazard_specific_methodology(self):
        record = verified_record(hazard="earthquake")
        without_method = build_research_assessment(
            jurisdiction="oakland",
            plan_area="Downtown",
            source_status="adopted",
            source_records=[record],
            methodology_config={},
        )
        self.assertEqual(without_method["hazards"]["earthquake"]["assessment_status"], "source_verified_method_pending")

        with_method = build_research_assessment(
            jurisdiction="oakland",
            plan_area="Downtown",
            source_status="adopted",
            source_records=[record],
            methodology_config={"earthquake": {"methodology_name": "official_area_rating"}},
        )
        self.assertEqual(with_method["hazards"]["earthquake"]["assessment_status"], "verified_official")
        self.assertTrue(with_method["hazards"]["earthquake"]["full_provenance"])

    def test_hazard_and_plan_area_records_do_not_cross_feed(self):
        earthquake_downtown = verified_record(hazard="earthquake", plan_area="Downtown")
        flood_downtown = verified_record(record_id="flood", hazard="flood", plan_area="Downtown")
        earthquake_west = verified_record(record_id="west", hazard="earthquake", plan_area="West Oakland")
        result = build_research_assessment(
            jurisdiction="oakland",
            plan_area="Downtown",
            source_status="adopted",
            source_records=[earthquake_downtown, flood_downtown, earthquake_west],
            methodology_config={
                "earthquake": {"methodology_name": "official_area_rating"},
                "flood": {"methodology_name": "official_area_rating"},
            },
        )
        earthquake_ids = {item["record_id"] for item in result["hazards"]["earthquake"]["full_provenance"]}
        flood_ids = {item["record_id"] for item in result["hazards"]["flood"]["full_provenance"]}
        self.assertEqual(earthquake_ids, {"verified-test-record"})
        self.assertEqual(flood_ids, {"flood"})

    def test_methodology_report_rejects_prohibited_averaging(self):
        report = load_research_json("methodology_report.json")["hazards"]
        for hazard in HAZARDS:
            prohibited = set(report[hazard]["prohibited_methods"])
            self.assertIn("combining_adopted_and_draft_records", prohibited)
            self.assertIn("using_context_only_metrics_as_hazard_categories", prohibited)
            self.assertEqual(
                report[hazard]["recommendation"],
                "mapped_findings_only_until_visual_verification_and_method_approval",
            )

    def test_plan_area_and_gis_validation_rules_fail_closed(self):
        geometry = load_research_json("plan_area_geometry_validation.json")
        gis = load_research_json("gis_source_validation.json")
        self.assertIn("zip_only_input_cannot_assign_area", geometry["fail_closed_rules"])
        self.assertIn("invalid_geometry_disables_area_assignment", geometry["fail_closed_rules"])
        self.assertIn("failed_gis_is_data_unavailable_not_low", gis["fail_closed_rules"])
        self.assertIn("fault_proximity_is_not_fault_zone_intersection", gis["fail_closed_rules"])
        self.assertIn("tsunami_evacuation_and_inundation_terms_remain_separate", gis["fail_closed_rules"])

    def test_golden_fixture_has_raw_response_and_cannot_masquerade_as_live_data(self):
        fixture = json.loads(
            (BASE_DIR / "tests" / "fixtures" / "oakland_hazard_assessment" / "golden_gis_fixture.json")
            .read_text(encoding="utf-8")
        )
        self.assertTrue(fixture["source_version"])
        self.assertTrue(fixture["retrieval_date"])
        self.assertTrue(fixture["raw_response"])
        self.assertTrue(fixture["expected_interpretation"])
        self.assertIn("fixture", fixture["fixture_id"])
        self.assertIn("never masquerade as a live official query", fixture["reason_for_expected_interpretation"])

    def test_adversarial_review_fixtures_are_rejected_or_flagged(self):
        fixtures = json.loads(
            (BASE_DIR / "tests" / "fixtures" / "oakland_hazard_assessment" / "adversarial_review_fixtures.json")
            .read_text(encoding="utf-8")
        )
        expected = {
            "row_shifted_by_one_line",
            "wrong_table_number",
            "wrong_pdf_page",
            "adopted_value_labeled_draft",
            "draft_value_labeled_adopted",
            "epc_percentage_mislabeled_probability",
            "exposed_population_mislabeled_hazard_score",
            "earthquake_scenario_attached_to_flood",
            "west_oakland_record_attached_to_downtown",
            "high_category_from_wrong_column",
            "correct_page_wrong_table",
            "correct_table_wrong_row",
            "correct_row_wrong_scenario_column",
            "table_continuation_omitted",
            "table_title_cropped_out",
            "footnote_changes_category_meaning",
            "plan_area_alias_wrong_area",
            "high_value_from_neighboring_row",
            "final_rating_confused_with_intermediate_score",
            "adopted_page_confused_with_draft_page",
        }
        self.assertEqual({fixture["case_id"] for fixture in fixtures["fixtures"]}, expected)
        for fixture in fixtures["fixtures"]:
            self.assertIn(fixture["expected_validation"], {"reject", "needs_more_review", "normalize_or_reject"})

    def test_batch_approval_requires_reviewer_and_timestamp(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        self.assertEqual(validate_table_batch_decision(batch, decision, BASE_DIR), [])
        no_reviewer = {**decision, "reviewer": ""}
        no_timestamp = {**decision, "reviewed_at": ""}
        self.assertIn("missing_reviewer", validate_table_batch_decision(batch, no_reviewer, BASE_DIR))
        self.assertIn("missing_review_timestamp", validate_table_batch_decision(batch, no_timestamp, BASE_DIR))

    def test_batch_approval_requires_adopted_source_and_identified_table(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        draft_batch = {**batch, "source_status": "draft"}
        cropped_title = {**batch, "table_title_status": "cropped"}
        hidden_headers = {**batch, "headers_status": "cropped"}
        self.assertIn("source_is_not_adopted", validate_table_batch_decision(draft_batch, decision, BASE_DIR))
        self.assertIn("table_title_not_visible", validate_table_batch_decision(cropped_title, decision, BASE_DIR))
        self.assertIn("headers_not_visible", validate_table_batch_decision(hidden_headers, decision, BASE_DIR))

    def test_missing_continuation_or_hidden_row_blocks_batch_approval(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        continuation = {**batch, "another_page_needed": True}
        truncated = {**batch, "hidden_or_truncated_rows": ["West Oakland"]}
        self.assertIn("missing_continuation_page", validate_table_batch_decision(continuation, decision, BASE_DIR))
        self.assertIn("hidden_or_truncated_rows", validate_table_batch_decision(truncated, decision, BASE_DIR))

    def test_value_plan_area_hazard_and_scenario_mismatches_block_batch_approval(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        value_mismatch = deepcopy(batch)
        value_mismatch["extracted_rows"][0]["risk_ranking_score"] = "999"
        plan_area_mismatch = deepcopy(batch)
        plan_area_mismatch["candidate_records"][0]["plan_area"] = "Made Up Area"
        hazard_mismatch = deepcopy(batch)
        hazard_mismatch["candidate_records"][0]["hazard"] = "earthquake"
        scenario_mismatch = deepcopy(batch)
        scenario_mismatch["candidate_records"][0]["scenario"] = "Wrong scenario"
        self.assertIn("candidate_value_mismatch", validate_table_batch_decision(value_mismatch, decision, BASE_DIR))
        self.assertIn("candidate_plan_area_mismatch", validate_table_batch_decision(plan_area_mismatch, decision, BASE_DIR))
        self.assertIn("candidate_hazard_mismatch", validate_table_batch_decision(hazard_mismatch, decision, BASE_DIR))
        self.assertIn("candidate_scenario_mismatch", validate_table_batch_decision(scenario_mismatch, decision, BASE_DIR))

    def test_context_or_exposure_table_cannot_produce_assessment_records(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        context_batch = {**batch, "assessment_eligible": False}
        population_batch = {**batch, "metric_type": "population_exposure"}
        epc_batch = {**batch, "metric_type": "EPC_context"}
        self.assertIn("table_not_assessment_eligible", validate_table_batch_decision(context_batch, decision, BASE_DIR))
        self.assertIn("context_or_exposure_metric_not_assessment_category", validate_table_batch_decision(population_batch, decision, BASE_DIR))
        self.assertIn("context_or_exposure_metric_not_assessment_category", validate_table_batch_decision(epc_batch, decision, BASE_DIR))

    def test_approved_batch_derives_visible_verified_record_updates(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        updates = derive_record_updates_from_batch(batch, decision, BASE_DIR)
        self.assertEqual(len(updates), 9)
        for update in updates:
            self.assertEqual(update["verification_status"], "visually_verified")
            self.assertEqual(update["approved_table_batch_id"], batch["batch_id"])
            self.assertEqual(update["source_column"], "Risk Ranking Score; Hazard Risk Rating")
            self.assertIn(update["source_row"], PLAN_AREAS)

    def test_corrected_batch_preserves_original_values(self):
        batch = load_batch()
        decision = valid_batch_decision(batch)
        first = batch["candidate_records"][0]
        decision["decision"] = "approve_with_corrections"
        decision["corrections"] = {
            first["record_id"]: {
                "original_extracted_value": first["raw_value"],
                "corrected_value": first["raw_value"],
                "correction_reason": "No numeric change; correction fixture proves preservation.",
            }
        }
        updates = derive_record_updates_from_batch(batch, decision, BASE_DIR)
        corrected = next(update for update in updates if update["record_id"] == first["record_id"])
        self.assertEqual(corrected["verification_status"], "corrected_after_visual_review")
        self.assertIn("original_extracted_value", corrected)
        self.assertIn("correction_reason", corrected)

    def test_regenerating_table_batches_does_not_erase_review_decisions(self):
        decision_dir = RESEARCH_DIR / "table_review_decisions"
        decision_dir.mkdir(exist_ok=True)
        sentinel = decision_dir / "sentinel_test_decision.json"
        sentinel.write_text('{"keep": true}\n', encoding="utf-8")
        self.addCleanup(lambda: sentinel.exists() and sentinel.unlink())
        subprocess.run(
            ["venv/bin/python", "scripts/generate_oakland_table_review_batches.py"],
            cwd=BASE_DIR,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), '{"keep": true}\n')

    def test_no_production_feature_flag_is_enabled_for_research(self):
        for path in [BASE_DIR / "app.py", BASE_DIR / "hazard_engine.py", BASE_DIR / "hazard_priority.py"]:
            parsed = ast.parse(path.read_text(encoding="utf-8"))
            names = {
                node.id for node in ast.walk(parsed)
                if isinstance(node, ast.Name)
            }
            self.assertNotIn("ENABLE_OAKLAND_RESEARCH_ASSESSMENT", names)


if __name__ == "__main__":
    unittest.main()
