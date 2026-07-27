# StayReady external review queue

Last automated source check: 2026-07-27

This file is the handoff boundary between repository automation and approvals that
must be completed by qualified people. No item below is approved merely because a
test, extraction, or service comparison passed.

## 1. Mapped evidence: GIS/emergency-management review

All eight datasets remain `provisional`. The deterministic suite has 40 cases:
34 pass, zero are source/snapshot mismatches, six are deliberately skipped
because a separate authoritative machine-readable coverage control is not yet
registered, and zero fail unexpectedly.

| Dataset | Automated cases | Current issue | Reviewer action |
| --- | ---: | --- | --- |
| FEMA NFHL layer 28 + panel layer 3 | 5 pass, 0 mismatch, 0 skip | Exact current service geometry, 2,451-feature county snapshot, Zone X and Zone D semantics, and effective-panel control are automated; public methodology remains unsigned | Inspect the AE boundary, Sunol Zone X, and Zone D controls; approve point/category/SFHA wording and panel-specific limitations |
| CAL FIRE effective SRA + LRA recommendation + responsibility 26_1 | 5 pass, 0 mismatch, 0 skip | Technical SRA/LRA/FRA selection is complete; all 15 LRA ordinance statuses remain unverified | Inspect Sunol/boundary High and FRA controls; verify ordinance evidence jurisdiction by jurisdiction and approve hazard-not-risk wording |
| Local mapped faults | 4 pass, 1 skip | File lineage/contributors unknown; 2 km threshold is product context, not an agency zone | Confirm official trace edition and three distance calculations; approve wording and rounding behavior |
| CGS Alquist-Priolo | 4 pass, 1 skip | Live service behavior is tested but official-viewer comparison is unsigned | Compare five cases and verify regulatory-zone wording |
| CGS liquefaction | 4 pass, 1 skip | Same unsigned viewer comparison | Compare five cases and verify that non-matches remain limited claims |
| CGS earthquake-induced landslide | 4 pass, 1 skip | Not a general landslide forecast | Compare five cases and approve the narrower claim |
| CGS tsunami hazard area | 4 pass, 1 skip | Evacuation-planning layer, not live warning or disclosure | Compare five shoreline/control cases and approve emergency-language boundary |
| DWR/DSOD dam scenarios | 4 pass, 1 skip | Hypothetical scenarios; federal/withheld dams and live evacuation decisions are outside the layer | Compare five cases, scenario labels, edition, and limitations |

Required signoff fields are in `docs/evidence-review-worksheet.csv`. A reviewer
must provide a real name, date, case result, limitation review, and approval
decision. The script must never fill those fields.

## 2. Local-plan claims: jurisdiction reviewer queue

Automation located 75 citation candidates—five for each jurisdiction—with a PDF
page, short excerpt, hazard tag, source-status badge, scope, and prohibited
claims. Every candidate is withheld in `data/lhmp/claim_candidates.json`;
`local_plans.json` still contains zero approved claim records.

Source selection follows the latest-official-publication policy: the newest
official jurisdiction publication is selected even when it is a draft. Drafts
must remain visibly draft-labeled in every source badge, citation, limitation,
and claim record. Adoption is not itself a beta publication requirement.

The repository also contains 63 Oakland sub-area table records and five Oakland
citywide hazard-ranking records from the draft 2026–2031 plan. Earlier automated
source-page checks are now explicitly `pending_external_review`; they are
fail-closed from the resident summary until a named reviewer and date are
recorded.

| Jurisdiction | Current status found on official source | Local source status | Candidate count | Human action |
| --- | --- | --- | ---: | --- |
| Alameda County / unincorporated | 2021 plan adopted 2022-03-01; FEMA approval expires 2027-03-10 | Matching plan, adoption-year filename | 5 | Verify plan identity, pages, county vs community scope |
| Alameda | Official URL now publishes a June 2026 Final Review Draft; filename reports Cal OES approval | Current 488-page PDF imported; 2022 PDF superseded | 5 | Confirm local adoption/FEMA status and review the five regenerated citations |
| Albany | 2023 draft PDF; current adoption/edition not established | Draft | 5 | Confirm official current plan and adoption record |
| Berkeley | 2024 plan adopted 2025-03-18 | Matching 2024 plan | 5 | Verify citations and approve claim-level wording |
| Dublin | 2024 Tri-Valley annex present; city adoption record not established | Pages 14–48 only | 5 | Confirm Dublin adoption; verify annex page range and claims |
| Emeryville | 2025–2030 final-draft source found; adoption record not established | Final draft | 5 | Confirm whether adopted and import final adopted copy if different |
| Fremont | 2024 Tri-City plan adopted 2024-11-12 and FEMA-approved | City annex | 5 | Verify citations against Annex A |
| Hayward | Official page describes the 2021 plan as adopted | 2021 plan | 5 | Locate exact resolution/date and verify citations |
| Livermore | Tri-Valley plan adopted 2024-03-11; FEMA approved 2023-12-15 | Pages 49–83 only | 5 | Verify annex range and citations |
| Newark | Tri-City plan is FEMA-approved; Newark adoption record not established | City annex | 5 | Locate Newark resolution and verify Annex B citations |
| Oakland | 2021–2026 plan adopted; 2026–2031 plan remains draft | Draft 2026 plan | 5 | Keep adopted and draft claims distinct; verify current Council status |
| Piedmont | Updated plan adopted 2026-03-02 after reported Cal OES/FEMA review | Current linked 721-page text imported, but it remains draft-labeled | 5 | Confirm the published text is identical to the adopted plan; then review five regenerated citations |
| Pleasanton | Tri-Valley plan adopted 2024-02-20; FEMA approved 2023-12-15 | Pages 84–113 only | 5 | Verify annex range and citations |
| San Leandro | City LHMP version 0.8 public review draft dated 2026-07-01; adoption section is TBD | Project-owner supplied, City-branded 253-page LHMP imported; 2025 Environmental Hazards Element remains separate provenance | 5 | Verify the five exact draft citations and, when available, record an official direct-PDF URL; do not imply adoption or FEMA approval |
| Union City | Tri-City plan is FEMA-approved; Union City adoption record not established | City annex | 5 | Locate Union City resolution and verify Annex C citations |

For a beta-reviewed batch, targeted human sampling must verify jurisdiction,
geographic scope, exact PDF page and table/section, plan edition, draft/adopted
status, faithful paraphrase, limitation, and whether sampled evidence supports
the documented publication decision. Ambiguous or high-impact claims require
individual review. Unsampled or unresolved claims remain withheld.

## 3. English public-beta language review

- A named reviewer must approve the complete English critical journey, including
  errors, privacy copy, limitations, and emergency warnings.
- Spanish is disabled and is not a blocker for the English-only initial beta.
- Professional Spanish translation, in-context review, and Spanish usability
  testing remain required for the later Spanish release.
- Arabic, Chinese, Farsi, Tagalog, and Vietnamese remain later milestones.

## 4. Accessibility and resident usability

- Complete an independent WCAG 2.2 AA audit with keyboard, VoiceOver/NVDA,
  200% zoom, high contrast, reduced motion, error recovery, print, and slow
  network. Close every critical and serious finding.
- Complete the eight-session English beta round in
  `docs/resident-research-plan.md`: four county regions, three access-needs
  participants, a findings report, and no unresolved critical comprehension or
  safety issue.
- Retain the 30-session, Spanish-inclusive study as a later language-access and
  general-availability milestone.
- Record responsive evidence at 320, 375, 430, 768, 1024, and 1440 CSS pixels.

## 5. Operations, privacy, and rollout

- An authorized operator must configure paid always-on staging/production,
  shared Redis-compatible state, ArcGIS credentials, monitoring, domain/TLS,
  backups, incident contacts, and rollback access.
- Complete an external privacy review of logs, analytics allowlists, feedback,
  retention, and data-subject request handling.
- Rehearse rollback and a partial-GIS-outage incident before traffic.
- After all other gates pass, use 5%, 25%, and 100% rollout stages with a
  48-hour review at each stage.

## Gate interpretation

`python scripts/validate_release.py` validates technical contracts only.
`python scripts/validate_release.py --public-beta` is intentionally expected to
fail until named approvals and operational configuration are recorded. Neither
command itself grants approval. Truthful pending/completed gate evidence is
recorded in `data/release_approvals.json`; automated tooling must not add a
reviewer, auditor, operator, or rollout authorization.
