import json
import tempfile
import unittest
from pathlib import Path

from scripts.lhmp_batch_extract import run_batch
from scripts.lhmp_coverage_report import (
    build_coverage_matrix,
    generate_coverage_report,
    suggest_coverage_tier,
)
from scripts.lhmp_extract import _load_aliases, _suggest_hazard


BASE_DIR = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def registry_record(city, *, enabled=True):
    return {
        "city": city,
        "city_display_name": city.replace("-", " ").title(),
        "plan_title": f"{city.title()} Plan",
        "plan_year": 2024,
        "local_pdf_path": f"inputs/{city}.pdf",
        "notes": "",
        "enabled": enabled,
        "max_render_pages": 2,
        "max_evidence_candidates": 20,
    }


def candidate(
    candidate_id,
    *,
    hazard="earthquake",
    heading="Earthquake Hazard Profile",
    text="Earthquake risk and vulnerability mitigation actions.",
):
    return {
        "id": candidate_id,
        "city": "alpha",
        "source_document": "alpha.pdf",
        "pdf_page": 2,
        "page_label": "4-2",
        "section_heading": heading,
        "suggested_hazard": hazard,
        "extraction_reason": "Configured alias matched.",
        "original_text": text,
        "extracted_snippet": text,
        "review_status": "candidate",
    }


class LHMPBatchExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = self.root / "data" / "lhmp" / "plan_registry.json"
        write_json(
            self.registry,
            [
                registry_record("alpha"),
                registry_record("disabled-city", enabled=False),
                registry_record("broken"),
                registry_record("omega"),
            ],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_skips_disabled_and_continues_after_failure(self):
        calls = []

        def fake_extractor(**kwargs):
            calls.append(kwargs["city"])
            if kwargs["city"] == "broken":
                raise RuntimeError("fixture failure")
            return {
                "evidence_candidates": 2,
                "visual_candidates": 1,
                "table_candidates": 0,
                "data_source_candidates": 1,
                "rendered_pages": [2],
                "warnings": [],
            }

        summary_path = self.root / "data" / "lhmp" / "batch_summary.json"
        summary = run_batch(
            self.registry,
            summary_path=summary_path,
            project_root=self.root,
            extractor=fake_extractor,
        )

        self.assertEqual(calls, ["alpha", "broken", "omega"])
        statuses = {item["city"]: item["status"] for item in summary["results"]}
        self.assertEqual(statuses["disabled-city"], "skipped_disabled")
        self.assertEqual(statuses["broken"], "failed")
        self.assertEqual(statuses["omega"], "succeeded")
        self.assertEqual(summary["totals"]["failed"], 1)
        self.assertEqual(summary["totals"]["succeeded"], 2)
        self.assertTrue(summary_path.is_file())

    def test_batch_does_not_touch_reviewed_or_static_directories(self):
        reviewed = self.root / "data" / "lhmp" / "reviewed" / "alpha" / "facts.json"
        public = self.root / "static" / "lhmp" / "alpha" / "figures" / "approved.txt"
        reviewed.parent.mkdir(parents=True)
        public.parent.mkdir(parents=True)
        reviewed.write_text("reviewed sentinel\n", encoding="utf-8")
        public.write_text("public sentinel\n", encoding="utf-8")

        run_batch(
            self.registry,
            summary_path=self.root / "batch.json",
            project_root=self.root,
            extractor=lambda **kwargs: {
                "evidence_candidates": 0,
                "visual_candidates": 0,
                "table_candidates": 0,
                "data_source_candidates": 0,
                "rendered_pages": [],
                "warnings": [],
            },
        )

        self.assertEqual(reviewed.read_text(), "reviewed sentinel\n")
        self.assertEqual(public.read_text(), "public sentinel\n")


class LHMPCoverageReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.registry = self.root / "data" / "lhmp" / "plan_registry.json"
        write_json(
            self.registry,
            [registry_record("alpha"), registry_record("missing-city")],
        )
        self.extracted = self.root / "data" / "lhmp" / "extracted"
        alpha = self.extracted / "alpha"
        write_json(
            alpha / "plan_manifest.json",
            {
                "review_status": "candidate",
                "source_document": "alpha.pdf",
                "pdf_page_count": 10,
                "source_pdf_sha256": "a" * 64,
                "extracted_at": "2026-06-12T00:00:00+00:00",
                "warnings": ["fixture warning"],
            },
        )
        write_json(
            alpha / "evidence_candidates.json",
            [
                candidate("eq-1"),
                candidate("eq-2"),
                candidate(
                    "unknown-1",
                    hazard=None,
                    heading="Volcanic Activity",
                    text="Volcanic activity appears in a source list.",
                ),
            ],
        )
        write_json(
            alpha / "visuals_candidates.json",
            [
                {
                    **candidate("visual-1"),
                    "visual_type": "map",
                    "caption": "Earthquake risk map",
                    "page_image_path": "data/lhmp/extracted/alpha/page-images/page-0002.png",
                    "address_specific": False,
                }
            ],
        )
        write_json(
            alpha / "tables_candidates.json",
            [
                {
                    **candidate("table-1"),
                    "caption": "Earthquake losses",
                    "extracted_text": "Risk, vulnerability, and mitigation actions",
                    "page_image_path": "",
                }
            ],
        )
        write_json(
            alpha / "data_sources_candidates.json",
            [
                {
                    **candidate("source-1"),
                    "source_text": "Source: California Geological Survey",
                    "suggested_name": "California Geological Survey",
                    "suggested_agency": "",
                    "suggested_url": "",
                }
            ],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_report_handles_missing_candidates_and_lists_unresolved(self):
        matrix = build_coverage_matrix(
            self.registry,
            extracted_root=self.extracted,
            batch_summary_path=self.root / "missing-batch.json",
        )
        cities = {city["city"]: city for city in matrix["cities"]}

        self.assertEqual(cities["missing-city"]["status"], "missing")
        self.assertTrue(cities["missing-city"]["warnings"])
        self.assertEqual(cities["alpha"]["unresolved"]["total"], 1)
        self.assertNotIn("volcanic", cities["alpha"]["hazards"])
        self.assertEqual(cities["alpha"]["hazards_detected"], ["earthquake"])

    def test_report_never_reads_reviewed_files_or_writes_public_assets(self):
        reviewed = (
            self.root / "data" / "lhmp" / "reviewed" / "alpha" / "facts.json"
        )
        static = self.root / "static" / "lhmp" / "alpha" / "figures" / "approved.txt"
        reviewed.parent.mkdir(parents=True)
        static.parent.mkdir(parents=True)
        reviewed.write_text(
            '[{"user_facing_text":"REVIEWED SECRET MUST NOT APPEAR"}]\n',
            encoding="utf-8",
        )
        static.write_text("public sentinel\n", encoding="utf-8")
        reviewed_before = reviewed.read_bytes()
        static_before = static.read_bytes()

        matrix_path = self.root / "data" / "lhmp" / "coverage_matrix.json"
        report_path = self.root / "data" / "lhmp" / "coverage_report.html"
        matrix = generate_coverage_report(
            registry_path=self.registry,
            extracted_root=self.extracted,
            batch_summary_path=self.root / "missing-batch.json",
            matrix_path=matrix_path,
            report_path=report_path,
        )

        self.assertNotIn("REVIEWED SECRET", json.dumps(matrix))
        self.assertNotIn("REVIEWED SECRET", report_path.read_text())
        self.assertEqual(reviewed.read_bytes(), reviewed_before)
        self.assertEqual(static.read_bytes(), static_before)

    def test_tier_logic_requires_multiple_candidate_signals(self):
        self.assertEqual(suggest_coverage_tier({}), "none")
        self.assertEqual(suggest_coverage_tier({"evidence": 1}), "basic")
        self.assertEqual(
            suggest_coverage_tier({"evidence": 3, "section_headings": 1}),
            "standard",
        )
        self.assertEqual(
            suggest_coverage_tier(
                {
                    "evidence": 4,
                    "section_headings": 1,
                    "mitigation_candidates": 1,
                    "visuals": 1,
                }
            ),
            "strong",
        )
        self.assertEqual(
            suggest_coverage_tier(
                {
                    "evidence": 8,
                    "section_headings": 1,
                    "mitigation_candidates": 1,
                    "risk_profile_candidates": 1,
                    "visuals": 1,
                    "tables": 1,
                    "data_sources": 1,
                }
            ),
            "full",
        )

    def test_hazard_aliases_are_applied_and_unknown_terms_remain_unresolved(self):
        aliases, overrides = _load_aliases(BASE_DIR, "hayward")
        self.assertEqual(
            _suggest_hazard("Seismic hazards", aliases, overrides)[0],
            "earthquake",
        )
        self.assertEqual(
            _suggest_hazard("Geologic hazards", aliases, overrides)[0],
            "earthquake",
        )
        self.assertIsNone(
            _suggest_hazard("Volcanic activity", aliases, overrides)[0]
        )


if __name__ == "__main__":
    unittest.main()
