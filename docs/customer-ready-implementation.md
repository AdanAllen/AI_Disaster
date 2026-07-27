# Customer-ready implementation status

StayReady is being prepared as an independent, English-only Alameda County
preparedness public beta. The repository now has stronger technical contracts,
but the product is **not approved to launch**. It is not an alert,
evacuation-order, engineering, insurance, legal-disclosure, property-risk, or
emergency-response service.

The initial beta does not claim bilingual support. Professionally reviewed
Spanish and Spanish usability work belong to a later language-access release.

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
- A 75-item local-plan citation queue covers all 15 jurisdictions. The selected
  source is each jurisdiction's latest official publication. Official drafts
  are eligible as current planning context only with visible draft status,
  exact citations, and no implication of adoption or final authority.
- The July 1, 2026 City of San Leandro LHMP version 0.8 public review draft is
  the selected San Leandro source. Its adoption section remains `TBD`, and the
  separate 2025 Environmental Hazards Element is retained only as provenance.
- Source schemas and local checksums are monitored without automatically promoting claims.
- Native map controls, coordinate-free text alternatives, self-hosted core UI/map dependencies, report-only CSP, CSRF, readiness checks, and privacy-safe aggregate analytics are in place.
- Paid production and staging Render blueprints are defined, including shared state and nightly source monitoring.

## Initial English public-beta gates

- A named emergency-management/GIS reviewer must complete and sign the evidence gold set.
- A named reviewer must approve the complete English safety-critical journey.
- At least eight English usability sessions must cover four county regions,
  including three access-needs participants, with a findings report and no open
  critical comprehension or safety issue.
- An independent WCAG 2.2 AA audit must close all critical and serious findings.
- Render, ArcGIS, monitoring, domain, and rollback credentials must be configured by an authorized operator.
- Production rollout must use 5%, 25%, and 100% stages with a 48-hour review at each stage.

Unreviewed local-plan candidates may remain withheld without blocking the beta.
Any displayed plan claim must have exact source status and citation, faithful
wording, preserved scope and limitations, and the required beta review evidence.
Formal adoption is not required when the newest official publication is a
clearly labeled draft.

## Later language-access milestone

- Complete and professionally review the Spanish critical journey.
- Conduct Spanish usability sessions.
- Enable Spanish resident-facing controls only after those gates pass.
- Treat Arabic, Chinese, Farsi, Tagalog, Vietnamese, and other county languages
  as separately reviewed future milestones.

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
