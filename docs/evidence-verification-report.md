# StayReady evidence verification report

Generated: 2026-07-27T20:33:38+00:00

Automated comparison is technical evidence only. It does not constitute human GIS, emergency-management, legal, accessibility, or publication approval.

## Summary

- Deterministic cases: 40
- Passed: 34
- Known source/snapshot mismatches: 0
- Failed unexpectedly: 0
- Skipped pending authoritative scope review: 6

## Dataset results

| Dataset | Edition/version | Passed | Known mismatch | Failed | Scope-review skip | Publication |
|---|---|---:|---:|---:|---:|---|
| `fema_nfhl_local` | Alameda County NFHL 06001C_20260525 / `nfhl-mapserver28-20260727-sha256-147c9049` | 5 | 0 | 0 | 0 | provisional |
| `calfire_fhsz_local` | Effective SRA 23_3 + 2025 LRA recommendations + responsibility areas 26_1 / `calfire-sra23_3-lra25-responsibility26_1` | 5 | 0 | 0 | 0 | provisional |
| `usgs_cgs_faults_local` | Lineage unverified — withheld / `local-snapshot-sha256-5c1b7015` | 4 | 0 | 0 | 1 | provisional |
| `cgs_alquist_priolo_remote` | CGS published service, 2025 metadata / `official-feature-service-accessed-2026-06-06` | 4 | 0 | 0 | 1 | provisional |
| `cgs_liquefaction_remote` | CGS published service, 2025 metadata / `official-feature-service-accessed-2026-06-06` | 4 | 0 | 0 | 1 | provisional |
| `cgs_earthquake_landslide_remote` | CGS published service, 2025 metadata / `official-feature-service-accessed-2026-06-06` | 4 | 0 | 0 | 1 | provisional |
| `cgs_tsunami_hazard_area_remote` | CGS current published tsunami hazard-area service / `official-feature-service-accessed-2026-06-06` | 4 | 0 | 0 | 1 | provisional |
| `dwr_dsod_dam_inundation_remote` | Approved inundation boundaries as of October 1, 2025 / `approved-boundaries-2025-10-01-accessed-2026-06-15` | 4 | 0 | 0 | 1 | provisional |

## Case results

### fema-positive-ae

- Outcome: **passed**
- Fixture: Official FEMA polygon 06001C_1016 internal sample
- Official: `matched`; categories: AE
- StayReady: `matched`; categories: AE
- Scope: FEMA flood-hazard polygon at the selected point
- Uncertainty: The repository snapshot lineage remains unverified even where this point agrees with the current official service.
- External review: required

### fema-negative-sunol-school

- Outcome: **passed**
- Fixture: Sunol Glen Unified School District
- Official: `not_matched`; categories: X
- StayReady: `not_matched`; categories: X
- Scope: No FEMA Special Flood Hazard Area match at the selected point
- Uncertainty: Point evidence remains distinct from parcel, elevation-certificate, insurance, and building determinations.
- External review: required

### fema-boundary-ae

- Outcome: **passed**
- Fixture: Official FEMA 06001C_1016 polygon boundary sample
- Official: `matched`; categories: AE
- StayReady: `matched`; categories: AE
- Scope: FEMA polygon boundary at address-point precision
- Uncertainty: Boundary inclusion is checked against the unsimplified county snapshot generated from the official package.
- External review: required

### fema-not-evaluated

- Outcome: **passed**
- Fixture: NFHL evaluation-scope control
- Official: `not_evaluated`; categories: D
- StayReady: `not_evaluated`; categories: D
- Scope: Official Zone D and effective-panel applicability, not polygon absence
- Uncertainty: The control automates the FEMA category and panel record; public interpretation still requires named methodology review.
- External review: required

### fema-unavailable

- Outcome: **passed**
- Fixture: Deterministic missing-snapshot simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the FEMA source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### calfire-positive-sunol-school

- Outcome: **passed**
- Fixture: Sunol Glen Unified School District
- Official: `matched`; categories: High
- StayReady: `matched`; categories: High
- Scope: Current SRA FHSZ and 2025 recommended LRA comparison
- Uncertainty: The point is selected through the current responsibility-area snapshot before the effective SRA severity snapshot.
- External review: required

### calfire-negative-alameda-city-hall

- Outcome: **passed**
- Fixture: Alameda City Hall
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: No Moderate, High, or Very High FHSZ match at the point
- Uncertainty: Local adoption status still requires jurisdiction review.
- External review: required

### calfire-boundary-high

- Outcome: **passed**
- Fixture: Unverified local High-zone boundary sample
- Official: `matched`; categories: High
- StayReady: `matched`; categories: High
- Scope: FHSZ polygon boundary at point precision
- Uncertainty: The exact recorded point intersects the effective SRA High polygon in both official and deterministic geometry.
- External review: required

### calfire-not-evaluated

- Outcome: **passed**
- Fixture: FRA/SRA/LRA applicability control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: Responsibility-area and local-adoption applicability
- Uncertainty: Federal-land fire mapping is outside the CAL FIRE SRA/LRA evidence contract and remains a separate preparedness limitation.
- External review: required

### calfire-unavailable

- Outcome: **passed**
- Fixture: Deterministic missing-snapshot simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the FHSZ source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### fault-positive-hayward-city-hall

- Outcome: **passed**
- Fixture: Hayward City Hall
- Official: `matched`; categories: Hayward fault zone
- StayReady: `matched`; categories: Hayward fault zone
- Scope: Proximity within 2 km; never polygon membership
- Uncertainty: Local snapshot lineage still requires approval.
- External review: required

### fault-negative-alameda-city-hall

- Outcome: **passed**
- Fixture: Alameda City Hall
- Official: `not_matched`; categories: Hayward fault zone
- StayReady: `not_matched`; categories: Hayward fault zone
- Scope: Nearest mapped trace is outside the 2 km proximity threshold
- Uncertainty: Distance should be sampled by a qualified reviewer.
- External review: required

### fault-boundary-two-km

- Outcome: **passed**
- Fixture: Synthetic 2 km proximity-threshold sample
- Official: `not_matched`; categories: Hayward fault zone
- StayReady: `not_matched`; categories: Hayward fault zone
- Scope: Approximately 2 km from the nearest loaded mapped trace
- Uncertainty: Official distance is approximately 2.002 km; manual sampling is required.
- External review: required

### fault-not-evaluated

- Outcome: **skipped_external_review**
- Fixture: Fault-map completeness control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: Completeness and applicable fault-map age/class
- Uncertainty: The local file mixes USGS and CGS records and has no verified download lineage.
- External review: required

### fault-unavailable

- Outcome: **passed**
- Fixture: Deterministic unreadable-layer simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only fault proximity is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### ap-positive-oakland-east

- Outcome: **passed**
- Fixture: CGS Oakland East zone OBJECTID 883 internal sample
- Official: `matched`; categories: Oakland East
- StayReady: `matched`; categories: Oakland East
- Scope: Alquist-Priolo regulatory fault zone polygon
- Uncertainty: None recorded by automation.
- External review: required

### ap-negative-alameda-city-hall

- Outcome: **passed**
- Fixture: Alameda City Hall
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: No returned Alquist-Priolo polygon at the point
- Uncertainty: Evaluation coverage still needs external confirmation.
- External review: required

### ap-boundary-oakland-east

- Outcome: **passed**
- Fixture: CGS Oakland East polygon boundary sample
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: Service boundary behavior at recorded coordinate precision
- Uncertainty: Boundary result is precision-sensitive.
- External review: required

### ap-not-evaluated

- Outcome: **skipped_external_review**
- Fixture: CGS regulatory-map coverage control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: CGS regulatory-map evaluation coverage
- Uncertainty: A CGS map-index or evaluation-coverage authority is required.
- External review: required

### ap-unavailable

- Outcome: **passed**
- Fixture: Deterministic ArcGIS timeout simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the Alquist-Priolo source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### liquefaction-positive-newark

- Outcome: **passed**
- Fixture: CGS Newark quadrangle OBJECTID 84 internal sample
- Official: `matched`; categories: Newark
- StayReady: `matched`; categories: Newark
- Scope: CGS liquefaction regulatory-zone polygon
- Uncertainty: None recorded by automation.
- External review: required

### liquefaction-negative-piedmont-city-hall

- Outcome: **passed**
- Fixture: Piedmont City Hall
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: No returned liquefaction regulatory polygon at the point
- Uncertainty: Evaluation coverage still requires confirmation.
- External review: required

### liquefaction-boundary-newark

- Outcome: **passed**
- Fixture: CGS Newark polygon boundary sample
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: Service boundary behavior at recorded coordinate precision
- Uncertainty: Boundary result is precision-sensitive.
- External review: required

### liquefaction-not-evaluated

- Outcome: **skipped_external_review**
- Fixture: CGS seismic-hazard-map coverage control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: CGS seismic-hazard-map evaluation coverage
- Uncertainty: A separate map-index comparison is required.
- External review: required

### liquefaction-unavailable

- Outcome: **passed**
- Fixture: Deterministic ArcGIS invalid-response simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the liquefaction source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### landslide-positive-dublin

- Outcome: **passed**
- Fixture: CGS Dublin quadrangle OBJECTID 2697 internal sample
- Official: `matched`; categories: Dublin
- StayReady: `matched`; categories: Dublin
- Scope: CGS earthquake-induced landslide regulatory-zone polygon
- Uncertainty: None recorded by automation.
- External review: required

### landslide-negative-alameda-city-hall

- Outcome: **passed**
- Fixture: Alameda City Hall
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: No returned earthquake-induced landslide polygon at the point
- Uncertainty: Evaluation coverage still requires confirmation.
- External review: required

### landslide-boundary-dublin

- Outcome: **passed**
- Fixture: CGS Dublin polygon boundary sample
- Official: `matched`; categories: Dublin
- StayReady: `matched`; categories: Dublin
- Scope: Earthquake-induced landslide polygon boundary
- Uncertainty: Automated service intersection returns a match at the recorded precision.
- External review: required

### landslide-not-evaluated

- Outcome: **skipped_external_review**
- Fixture: CGS seismic-hazard-map coverage control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: CGS seismic-hazard-map evaluation coverage
- Uncertainty: A separate official map-index comparison is required.
- External review: required

### landslide-unavailable

- Outcome: **passed**
- Fixture: Deterministic ArcGIS timeout simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the earthquake-induced landslide source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### tsunami-positive-alameda-area

- Outcome: **passed**
- Fixture: CGS Alameda tsunami area OBJECTID 12 internal sample
- Official: `matched`; categories: Yes, Tsunami Hazard Area
- StayReady: `matched`; categories: Yes, Tsunami Hazard Area
- Scope: Tsunami evacuation and response planning area
- Uncertainty: None recorded by automation.
- External review: required

### tsunami-negative-oakland-city-hall

- Outcome: **passed**
- Fixture: Oakland City Hall
- Official: `not_matched`; categories: No, Outside Hazard Area
- StayReady: `not_matched`; categories: none
- Scope: Outside the returned tsunami evacuation-planning area
- Uncertainty: None recorded by automation.
- External review: required

### tsunami-boundary-alameda

- Outcome: **passed**
- Fixture: CGS Alameda tsunami-area boundary sample
- Official: `matched`; categories: No, Outside Hazard Area, Yes, Tsunami Hazard Area
- StayReady: `matched`; categories: Yes, Tsunami Hazard Area
- Scope: Shared boundary between outside and evacuation-planning polygons
- Uncertainty: Shared-boundary behavior requires manual inspection.
- External review: required

### tsunami-not-evaluated

- Outcome: **skipped_external_review**
- Fixture: Tsunami populated-coast evaluation-scope control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: Populated coastal evaluation coverage
- Uncertainty: Official scope wording needs reviewer approval.
- External review: required

### tsunami-unavailable

- Outcome: **passed**
- Fixture: Deterministic ArcGIS invalid-response simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the tsunami source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

### dam-positive-san-leandro-scenarios

- Outcome: **passed**
- Fixture: Official DSOD internal scenario sample near San Leandro
- Official: `matched`; categories: Chabot · Scenario1 · Sunny Day, New Upper San Leandro · Scenario1 · Sunny Day
- StayReady: `matched`; categories: Chabot · Scenario1 · Sunny Day, New Upper San Leandro · Scenario1 · Sunny Day
- Scope: Published hypothetical dam-failure inundation scenarios
- Uncertainty: None recorded by automation.
- External review: required

### dam-negative-alameda-city-hall

- Outcome: **passed**
- Fixture: Alameda City Hall
- Official: `not_matched`; categories: none
- StayReady: `not_matched`; categories: none
- Scope: No returned approved dam-failure scenario polygon
- Uncertainty: None recorded by automation.
- External review: required

### dam-boundary-new-upper-san-leandro

- Outcome: **passed**
- Fixture: DSOD New Upper San Leandro Scenario1 boundary sample
- Official: `matched`; categories: New Upper San Leandro · Scenario1 · Sunny Day
- StayReady: `matched`; categories: New Upper San Leandro · Scenario1 · Sunny Day
- Scope: Published hypothetical scenario boundary
- Uncertainty: Manual viewer inspection is required.
- External review: required

### dam-not-evaluated

- Outcome: **skipped_external_review**
- Fixture: DSOD approved-scenario applicability control
- Official: `not_evaluated`; categories: none
- StayReady: `not_evaluated`; categories: none
- Scope: Completeness of approved scenario submissions
- Uncertainty: Scenario completeness requires DSOD subject-matter review.
- External review: required

### dam-unavailable

- Outcome: **passed**
- Fixture: Deterministic ArcGIS timeout simulation
- Official: `unavailable`; categories: none
- StayReady: `data_unavailable`; categories: none
- Scope: Only the dam-scenario source is unavailable
- Uncertainty: None recorded by automation.
- External review: required

## Optional live-source check

- Passed: 26
- Failed: 0
- Unavailable: 0
- Total queried: 26

## Commands

```bash
python scripts/verify_evidence_gold_set.py
python scripts/verify_evidence_gold_set.py --live
python scripts/verify_evidence_gold_set.py --refresh-official --live
```

`--refresh-official` updates observations only. Expected results and human approval fields are never updated automatically.
