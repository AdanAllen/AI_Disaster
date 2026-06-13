# LHMP extraction data

This directory separates offline extraction output from manually reviewed
content.

- `raw/`: local source PDFs. PDFs are ignored by git.
- `markdown/`: optional text conversions used to locate source passages.
- `extracted/`: machine-generated candidates and non-public page renders.
- `reviewed/`: human-approved records. This is the only directory that
  `lhmp_repository.py` reads.
- `plan_registry.json`: countywide inventory of local plan inputs and
  per-document extraction caps.
- `hazard_aliases.json`: configurable mapping from plan terminology to
  canonical StayReady hazard names.
- `batch_summary.json`: latest per-plan extraction outcome.
- `coverage_matrix.json` and `coverage_report.html`: generated candidate
  coverage audits. They are review workflow aids, not website inputs.

Candidate evidence is not publishable copy. Reviewers must verify the original
PDF, physical PDF page, printed page label, scope, wording, and hazard before
manually creating or updating a reviewed record.

Approved public visuals must be manually cropped and copied to
`static/lhmp/{city}/figures/`. Full candidate page renders must remain under
`data/lhmp/extracted/{city}/page-images/`.

Install the offline dependency separately, then run the countywide audit:

```bash
pip install -r requirements-lhmp.txt
python scripts/lhmp_batch_extract.py \
  --registry data/lhmp/plan_registry.json \
  --replace-candidates
python scripts/lhmp_coverage_report.py
```

The coverage tiers (`none`, `basic`, `standard`, `strong`, and `full`) describe
the breadth of deterministic candidates found for human review. They do not
mean that a plan fact, hazard claim, map, table, or source has been approved.
Unknown terminology is reported as unresolved unless it maps through
`hazard_aliases.json`.
