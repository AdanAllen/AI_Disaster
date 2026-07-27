# FEMA, CAL FIRE, and coverage-control remediation

Date: 2026-07-27

This is a technical evidence record, not human GIS or publication approval. All
eight mapped datasets remain `provisional` and resident-facing publication stays
fail-closed.

## FEMA NFHL

- Authority: FEMA NFHL MapServer layer 28, `Flood Hazard Zones`, service version
  11.1.
- Deterministic geometry: all 2,451 Alameda County `DFIRM_ID=06001C` features
  queried from current layer 28 and stored without display simplification.
- County package: `NFHL_06001C` resolved to
  `06001C_20260525.zip`, archive SHA-256
  `5f91caa649744793a08b4cc459e3b27930ca9e8098cbd3ad3b34c5a47ecbeb2e`.
- Coverage authority: the 164 `S_FIRM_PAN` records from that package, compared
  through NFHL MapServer layer 3 (`FIRM Panels`).
- Boundary policy: `contains` or `touches` at the recorded coordinate. ZIP-clipped
  simplified files are display-only and cannot drive address evidence.
- Zone X is retained as an official category while `SFHA_TF=F` remains a checked
  non-SFHA match. It is never worded as zero flood risk.
- Zone D is returned as `not_evaluated`/undetermined, not as a checked non-match.

Changed controls:

| Case | Previous deterministic result | Current official result | New deterministic result | Reason |
| --- | --- | --- | --- | --- |
| Sunol school | no category | Zone X, non-SFHA | Zone X, non-SFHA | Replaced the incomplete snapshot with the current official service geometry |
| Boundary | VE | AE | AE | Production evidence now uses current layer 28 geometry, not the older package boundary or simplified ZIP display geometry |
| Scope | skipped | Zone D + effective panel | Zone D / `not_evaluated` | Added layer 3 and `S_FIRM_PAN` applicability control |

Limitations: this is point-in-polygon evidence, not a parcel, elevation
certificate, insurance, building, or safety determination. Effective dates are
panel-specific.

## CAL FIRE FHSZ

The previous comparator combined a Cal OES-hosted SRA layer with a
Stanford-owned LRA project layer. That was not an authoritative SRA/LRA
selection contract.

The replacement import uses CAL FIRE owner `prefire.calfire` services:

| Authority | Service/layer | Edition | Local snapshot |
| --- | --- | --- | --- |
| Effective SRA severity | `FHSZSRA_23_3/FeatureServer/0` | Adopted 2024-01-31; effective 2024-04-01 | `static/FireHazSRA.geojson` |
| LRA recommendation | `FHSALRA25_v1_All/FeatureServer/0` | 2025 statewide recommendations | `static/FireHazLRARecommended.geojson` |
| Responsibility applicability | `State_Responsibility_Area/FeatureServer/0` | SRA 26_1 as of 2026-04-09 | `static/FireResponsibilityAreas.geojson` |

Production selection is now:

1. Resolve the official responsibility polygon.
2. For SRA, check only the effective SRA severity snapshot.
3. For LRA, retain 2025 recommendation evidence separately and make no adopted
   local-zone claim without ordinance evidence.
4. For FRA, return `not_evaluated` because CAL FIRE SRA/LRA severity layers are
   not applicable.
5. For a responsibility boundary or outage, do not choose a favorable authority
   or convert the condition into a non-match.

The 15-jurisdiction adoption matrix is
`data/geospatial/calfire_lra_adoption_matrix.json`. Every jurisdiction is
currently `ordinance_not_verified`; this is deliberately fail-closed.

Changed controls:

| Case | Previous deterministic result | Current official result | New deterministic result | Reason |
| --- | --- | --- | --- | --- |
| Sunol school | Very High | SRA High | SRA High | Replaced unverified mixed snapshot and select after responsibility check |
| Boundary | Moderate | SRA High | SRA High | Current official geometry and exact recorded coordinate |
| Applicability | skipped | FRA | `not_evaluated` | Added official 26_1 responsibility-area control |

Fire Hazard Severity Zone is hazard severity, not complete wildfire risk,
probability, smoke, ember exposure, evacuation conditions, or building
vulnerability.

## Remaining six scope controls

| Dataset | Authority checked | Why automation still cannot infer coverage from absence |
| --- | --- | --- |
| Mapped fault traces | USGS Quaternary Faults MapServer layer 4 | It is a fault inventory, not a polygon of evaluated completeness; the two-kilometer threshold is StayReady-created proximity context |
| CGS Alquist-Priolo | Official CGS zone service and Regulatory Maps/Information Warehouse | Zone polygons do not by themselves prove that an absent point lies in an officially evaluated quadrangle and edition |
| CGS liquefaction | Official CGS zone service and Seismic Hazard Zone map/evaluation-report catalog | CGS is still mapping principal urban and growth areas; a machine-readable evaluated-quadrangle control has not been registered |
| CGS earthquake-induced landslide | Same official seismic-hazard map catalog | Polygon absence cannot establish official quadrangle evaluation for this hazard |
| CGS tsunami hazard area | Official 2021 hazard-area service and county map catalog | The layer is for coastal evacuation planning; there is no separate machine-readable statewide “evaluated but no hazard” polygon registered |
| DWR/DSOD dam scenarios | Official approved-inundation scenario service | It is a scenario inventory and can omit/exclude/withhold dams; polygon absence is not proof of evaluated safety |

These six cases remain explicit scope-review skips. They were not changed to
passes by treating polygon absence as coverage.
