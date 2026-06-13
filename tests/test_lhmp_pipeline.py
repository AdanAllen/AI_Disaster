import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import fitz
from pydantic import ValidationError

import lhmp_repository
from lhmp_models import (
    EvidenceCandidate,
    ReviewedFact,
    ReviewedVisual,
)
from scripts.lhmp_extract import extract_lhmp


BASE_DIR = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def reviewed_fields(**overrides):
    payload = {
        "id": "hayward-earthquake-001",
        "city": "hayward",
        "hazard": "earthquake",
        "source_document": "hayward-plan.pdf",
        "pdf_page": 2,
        "page_label": "9-4",
        "section_heading": "Earthquake Hazard",
        "reviewer_id": "reviewer-1",
        "reviewed_at": datetime(2026, 6, 12, tzinfo=timezone.utc).isoformat(),
        "review_notes": "Verified against the source PDF.",
        "review_status": "reviewed",
    }
    payload.update(overrides)
    return payload


class LHMPExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        aliases = json.loads(
            (BASE_DIR / "data" / "lhmp" / "hazard_aliases.json").read_text(
                encoding="utf-8"
            )
        )
        write_json(self.root / "data" / "lhmp" / "hazard_aliases.json", aliases)
        self.pdf_path = self.root / "inputs" / "hayward-plan.pdf"
        self.pdf_path.parent.mkdir(parents=True)
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "PLAN OVERVIEW\nThis plan describes community mitigation planning and local hazards.",
        )
        page = document.new_page()
        page.insert_text(
            (72, 72),
            (
                "EARTHQUAKE HAZARD\n"
                "Strong earthquake shaking can damage buildings and interrupt roads, utilities, "
                "communications, and household recovery across Hayward.\n\n"
                "Figure 9-1 Earthquake Fault Context Map\n"
                "Source: California Geological Survey https://example.gov/seismic"
            ),
        )
        page = document.new_page()
        page.insert_text(
            (72, 72),
            (
                "EARTHQUAKE TABLES\n"
                "Table 9-2 Earthquake scenario summary\n"
                "Scenario information shown for mitigation planning and source review."
            ),
        )
        document.save(self.pdf_path)
        document.close()

    def tearDown(self):
        self.temp.cleanup()

    def test_extractor_creates_bounded_non_public_candidates(self):
        reviewed = self.root / "data" / "lhmp" / "reviewed" / "hayward"
        reviewed.mkdir(parents=True)
        sentinel = reviewed / "facts.json"
        sentinel.write_text('[{"sentinel": true}]\n', encoding="utf-8")
        sentinel_before = sentinel.read_bytes()

        result = extract_lhmp(
            city="hayward",
            pdf_path=self.pdf_path,
            hazard_filter="earthquake",
            max_render_pages=1,
            replace_candidates=False,
            project_root=self.root,
        )

        output = self.root / "data" / "lhmp" / "extracted" / "hayward"
        for filename in (
            "plan_manifest.json",
            "evidence_candidates.json",
            "visuals_candidates.json",
            "tables_candidates.json",
            "data_sources_candidates.json",
        ):
            self.assertTrue((output / filename).is_file())

        manifest = json.loads((output / "plan_manifest.json").read_text())
        evidence = json.loads((output / "evidence_candidates.json").read_text())
        visuals = json.loads((output / "visuals_candidates.json").read_text())
        tables = json.loads((output / "tables_candidates.json").read_text())

        self.assertEqual(manifest["review_status"], "candidate")
        self.assertEqual(manifest["pdf_page_count"], 3)
        self.assertEqual(manifest["rendered_pdf_pages"], [2])
        self.assertTrue(evidence)
        self.assertTrue(all(item["pdf_page"] > 0 for item in evidence))
        self.assertTrue(
            all(item["suggested_hazard"] == "earthquake" for item in evidence)
        )
        self.assertEqual(len(visuals), 1)
        self.assertEqual(visuals[0]["pdf_page"], 2)
        self.assertFalse(visuals[0]["address_specific"])
        self.assertTrue(tables)
        self.assertEqual(result["rendered_pages"], [2])

        rendered = list((output / "page-images").glob("*.png"))
        self.assertEqual(len(rendered), 1)
        self.assertTrue(
            rendered[0].resolve().is_relative_to(
                (self.root / "data" / "lhmp" / "extracted").resolve()
            )
        )
        self.assertFalse((self.root / "static" / "lhmp").exists())
        self.assertEqual(sentinel.read_bytes(), sentinel_before)

    def test_existing_candidates_require_explicit_replacement(self):
        extract_lhmp(
            city="hayward",
            pdf_path=self.pdf_path,
            max_render_pages=0,
            project_root=self.root,
        )
        with self.assertRaises(FileExistsError):
            extract_lhmp(
                city="hayward",
                pdf_path=self.pdf_path,
                max_render_pages=0,
                project_root=self.root,
            )
        keep_file = (
            self.root
            / "data"
            / "lhmp"
            / "extracted"
            / "hayward"
            / "page-images"
            / ".gitkeep"
        )
        keep_file.touch()
        extract_lhmp(
            city="hayward",
            pdf_path=self.pdf_path,
            max_render_pages=0,
            replace_candidates=True,
            project_root=self.root,
        )
        self.assertTrue(keep_file.exists())

    def test_markdown_page_and_printed_label_are_separate(self):
        markdown = self.root / "inputs" / "hayward.md"
        markdown.write_text(
            (
                "# Earthquake Hazard\n"
                "<!-- pdf_page: 2; page_label: 9-4 -->\n\n"
                "Strong earthquake shaking can damage buildings and interrupt roads, utilities, "
                "communications, and household recovery across Hayward.\n\n"
                "# Unresolved material\n"
                "This sentence does not occur anywhere in the source document and must be skipped.\n"
            ),
            encoding="utf-8",
        )
        result = extract_lhmp(
            city="hayward",
            pdf_path=self.pdf_path,
            markdown_path=markdown,
            max_render_pages=0,
            project_root=self.root,
        )
        evidence = json.loads(
            (
                self.root
                / "data"
                / "lhmp"
                / "extracted"
                / "hayward"
                / "evidence_candidates.json"
            ).read_text()
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["pdf_page"], 2)
        self.assertEqual(evidence[0]["page_label"], "9-4")
        self.assertEqual(evidence[0]["section_heading"], "Earthquake Hazard")
        self.assertTrue(any("Skipped unresolved Markdown" in item for item in result["warnings"]))

    def test_candidate_is_not_a_reviewed_fact(self):
        candidate = EvidenceCandidate(
            id="candidate-1",
            city="hayward",
            source_document="plan.pdf",
            pdf_page=2,
            page_label="9-4",
            section_heading="Earthquake",
            suggested_hazard="earthquake",
            extraction_reason="Matched configured alias.",
            extracted_snippet="Source text only.",
            review_status="candidate",
        )
        with self.assertRaises(ValidationError):
            ReviewedFact.model_validate(candidate.model_dump())


class LHMPReviewedRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.reviewed_root = Path(self.temp.name) / "reviewed"
        self.city_root = self.reviewed_root / "hayward"
        self.city_root.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_repository_reads_only_valid_reviewed_records(self):
        write_json(
            self.city_root / "plan_manifest.json",
            {
                "schema_version": 1,
                "city": "hayward",
                "jurisdiction_name": "City of Hayward",
                "plan_title": "Hayward Local Resilience Plan",
                "source_document": "hayward-plan.pdf",
                "source_url": "https://example.gov/hayward-plan",
                "source_pdf_sha256": "a" * 64,
                "edition": "2021",
                "effective_date": "2021",
                "document_status": "Adopted",
                "reviewer_id": "reviewer-1",
                "reviewed_at": "2026-06-12T12:00:00Z",
                "review_notes": "Verified against the original PDF.",
                "review_status": "reviewed",
            },
        )
        write_json(
            self.city_root / "facts.json",
            [
                reviewed_fields(
                    user_facing_text="Hayward plans for strong earthquake shaking.",
                    plain_english_summary="Earthquake shaking is citywide planning context.",
                    source_snippet="The plan discusses strong earthquake shaking.",
                    display_locations=["risk_summary", "city_hazard_page"],
                    geographic_scope="citywide",
                    address_specific=False,
                )
            ],
        )
        write_json(
            self.city_root / "visuals.json",
            [
                reviewed_fields(
                    id="hayward-earthquake-map",
                    title="Earthquake planning map",
                    caption="A contextual figure from the local plan.",
                    visual_type="map",
                    asset_path="lhmp/hayward/figures/earthquake-map.png",
                    display_locations=["city_hazard_page"],
                    address_specific=False,
                )
            ],
        )
        write_json(
            self.city_root / "data_sources.json",
            [
                reviewed_fields(
                    id="hayward-cgs-source",
                    name="California Geological Survey",
                    agency="California Geological Survey",
                    official_url="https://example.gov/cgs",
                    supported_claim="Supports the plan's earthquake map.",
                    dataset_id="cgs-example",
                )
            ],
        )

        with patch.object(lhmp_repository, "REVIEWED_ROOT", self.reviewed_root):
            self.assertEqual(
                lhmp_repository.get_reviewed_plan("hayward")["review_status"],
                "reviewed",
            )
            facts = lhmp_repository.get_reviewed_facts(
                "hayward",
                hazard="earthquake",
                display_location="risk_summary",
            )
            self.assertEqual(len(facts), 1)
            self.assertEqual(
                lhmp_repository.get_reviewed_facts(
                    "hayward", display_location="map_explanation"
                ),
                [],
            )
            self.assertEqual(len(lhmp_repository.get_reviewed_visuals("hayward")), 1)
            self.assertEqual(
                len(lhmp_repository.get_reviewed_data_sources("hayward")), 1
            )

    def test_missing_and_malformed_reviewed_data_fail_closed(self):
        write_json(self.city_root / "facts.json", [{"review_status": "candidate"}])
        extracted = self.reviewed_root.parent / "extracted" / "hayward"
        write_json(
            extracted / "evidence_candidates.json",
            [{"id": "must-never-load", "review_status": "candidate"}],
        )
        with patch.object(lhmp_repository, "REVIEWED_ROOT", self.reviewed_root):
            self.assertEqual(lhmp_repository.get_reviewed_facts("hayward"), [])
            self.assertEqual(lhmp_repository.get_reviewed_plan("missing-city"), {})
            self.assertEqual(lhmp_repository.get_reviewed_facts("../../etc"), [])

    def test_address_specific_visual_requires_reviewed_gis_linkage(self):
        base = reviewed_fields(
            id="hayward-address-map",
            title="Address map",
            caption="Contextual map.",
            visual_type="map",
            asset_path="lhmp/hayward/figures/address-map.png",
            display_locations=["map_explanation"],
            address_specific=True,
        )
        with self.assertRaises(ValidationError):
            ReviewedVisual.model_validate(base)
        valid = ReviewedVisual.model_validate({
            **base,
            "gis_dataset_id": "official-layer-id",
            "official_gis_url": "https://example.gov/official-gis",
            "gis_linkage_reviewed": True,
        })
        self.assertTrue(valid.address_specific)

    def test_new_repository_is_not_wired_into_hazard_ranking(self):
        for filename in ("app.py", "hazard_engine.py", "resident_guidance_engine.py"):
            source = (BASE_DIR / filename).read_text(encoding="utf-8")
            self.assertNotIn("lhmp_repository", source)

    def test_pdf_dependency_is_not_in_production_requirements(self):
        production = (BASE_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
        offline = (BASE_DIR / "requirements-lhmp.txt").read_text(encoding="utf-8").lower()
        self.assertNotIn("pymupdf", production)
        self.assertIn("pymupdf", offline)


if __name__ == "__main__":
    unittest.main()
