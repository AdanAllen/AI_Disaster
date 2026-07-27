import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "data/geospatial/evidence_gold_set_fixtures.json"
REPORT_PATH = ROOT / "docs/evidence-verification-report.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_gold_set_has_five_required_cases_for_every_dataset():
    fixtures = load_json(FIXTURE_PATH)
    datasets = load_json(ROOT / "data/geospatial/datasets.json")
    required = {
        "positive",
        "negative",
        "boundary",
        "not_evaluated",
        "unavailable",
    }

    assert fixtures["policy"]["automated_review_is_human_approval"] is False
    assert len(fixtures["cases"]) == 40
    for dataset in datasets:
        case_types = {
            case["test_case_type"]
            for case in fixtures["cases"]
            if case["dataset_id"] == dataset["dataset_id"]
        }
        assert case_types == required


def test_deterministic_report_has_no_unexpected_failures():
    report = load_json(REPORT_PATH)

    assert report["mode"] == "deterministic_snapshot"
    assert report["summary"] == {
        "passed": 34,
        "known_mismatch": 0,
        "failed": 0,
        "skipped_external_review": 6,
        "total": 40,
    }
    assert all(case["external_review_required"] for case in report["cases"])


def test_gold_set_verifier_passes_without_network_or_writes():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_evidence_gold_set.py",
            "--no-write",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 failed" in result.stdout
