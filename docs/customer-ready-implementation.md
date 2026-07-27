# Customer-ready implementation status

StayReady is being prepared as an independent Alameda County preparedness public
beta. The repository now has stronger technical contracts, but the product is
**not approved to launch**. It is not an alert, evacuation-order, engineering,
insurance, or emergency-response service.

## Implemented

- Alameda County Address Points are the primary locator. An authenticated ArcGIS World Geocoder is only a fallback.
- Official city and unincorporated-community polygons determine jurisdiction. ZIPs and parsed city text do not.
- Candidate and result APIs keep coordinates out of responses, URLs, cookies, and analytics.
- Ambiguous input requires selection. PO boxes, unmatched input, out-of-county points, ZIP fallback, and map pins have explicit states.
- Resident result state expires after 30 minutes and uses Redis-compatible shared state in staging/production.
- Mapped datasets have separate source-check, effective, retrieval, and human-review fields. Production withholds every unverified edition.
- A 40-case mapped-evidence gold set covers positive, negative, boundary,
  not-evaluated, and unavailable behavior for all eight datasets. The deterministic
  suite currently reports 34 passes, zero known mismatches, six truthful
  scope-review skips, and zero unexpected failures.
- FEMA address evidence now uses a reproducible current layer-28 snapshot and a
  separate effective-panel coverage snapshot. Display-simplified ZIP files no
  longer participate in evidence decisions.
- CAL FIRE evidence now resolves SRA/LRA/FRA applicability first, keeps effective
  SRA and 2025 LRA recommendation geometry separate, and withholds every LRA
  adoption claim pending ordinance evidence.
- A 75-item local-plan citation queue covers all 15 jurisdictions. It identifies
  superseded or mismatched local documents and withholds every machine-selected
  claim pending named review.
- Source schemas and local checksums are monitored without automatically promoting claims.
- Native map controls, coordinate-free text alternatives, self-hosted core UI/map dependencies, report-only CSP, CSRF, readiness checks, and privacy-safe aggregate analytics are in place.
- Paid production and staging Render blueprints are defined, including shared state and nightly source monitoring.

## Gates that require people or credentials

- A named emergency-management/GIS reviewer must complete and sign the evidence gold set.
- A professional translator must complete and approve every Spanish critical-journey key in context.
- Thirty resident sessions must be completed across all six county regions and required access/language cohorts.
- An independent WCAG 2.2 AA audit must close all critical and serious findings.
- Render, ArcGIS, monitoring, domain, and rollback credentials must be configured by an authorized operator.
- Production rollout must use 5%, 25%, and 100% stages with a 48-hour review at each stage.

External approval state is recorded in `data/release_approvals.json`. Its initial
values are deliberately pending; only evidence from a real named reviewer,
auditor, participant-research owner, or authorized operator may close a gate.

Run:

```text
python scripts/build_local_plan_claim_queue.py --check
python scripts/verify_evidence_gold_set.py --no-write
python scripts/validate_release.py
python scripts/validate_release.py --public-beta
```

The first three are technical checks. The last command is expected to fail until
the external-review queue is complete. See `docs/external-review-queue.md`.
