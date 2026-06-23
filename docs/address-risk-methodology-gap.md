# StayReady address-risk methodology gap

## Current rating-label inventory

| Label family | Current source/code | Scope and meaning | Address-valid? | Classification |
|---|---|---|---|---|
| `High`, `Medium`, `Low` probability and impact | `data/hazard_priority/jurisdiction_hazard_rankings.json`; `hazard_priority.calculate_citywide_priority()` | Jurisdiction planning categories. Oakland records currently cite a draft LHMP table. | No. Display only with visible jurisdiction, source status, and exact citation. | Valid jurisdiction category when eligible; otherwise withheld |
| Official LHMP priority (`High`, `Medium`, `Low`) | Same ranking records and exact source table | Official or draft jurisdiction-wide planning priority, not property risk. | No. | Valid jurisdiction category with scope/status disclosure |
| Oakland combined scenario rating | `official_area_rating_for_hazard()` | Plan-area scenario summary using documented categorical combination. | No. | Valid plan-area context only after evidence/review eligibility |
| FEMA zone/category including SFHA and Zone AE/X | FEMA GIS result in `hazard_engine` | Address-point mapped flood category. | Yes, as the exact mapped category only. | Valid mapped-zone category |
| CAL FIRE Moderate/High/Very High FHSZ | CAL FIRE GIS result in `hazard_engine` | Address-point mapped fire-hazard severity category. | Yes, as mapped severity—not annual probability or expected loss. | Valid mapped-zone category |
| CGS Alquist-Priolo, liquefaction, earthquake-landslide matches | CGS structured records | Positive address-point regulatory/hazard-zone intersection when `matched=True`. | Yes, as an intersection only. | Valid address-level category |
| Fault distance/proximity | USGS/CGS fault check | Distance to a mapped feature; not polygon membership. | Yes, as proximity context. | Valid address context |
| `Start here`, `Important preparedness priority`, `Keep in your plan` | `_classify_preparedness_priority()` / `merge_structured_result()` | Action ordering based on evidence and regional importance. | Yes, if clearly described as preparedness attention, not risk. | Preparedness context |
| `High/Moderate/Low` from `zip_risk_scores.csv` | Legacy compatibility code in `app.py` | Broad ZIP heuristic used only for explicit ZIP fallback. | No. | Legacy heuristic |
| `risk_level` derived from `exposure_level.title()` | `hazard_engine.merge_structured_result()` | Compatibility label that can resemble a risk rating while representing exposure state. | No for a universal address-risk claim. | Unsupported/misleading outside compatibility use |
| Confidence `High/Medium/Low` | source/evidence confidence fields | Confidence in evidence handling, not hazard likelihood or damage. | Only when visibly labeled confidence. | Evidence-quality category |

## Hazard-by-hazard gap

| Hazard | Jurisdiction probability / impact | Address exposure | Building vulnerability | Combining formula | Appropriate agency address category | Honest address H/M/L today? | Required work |
|---|---|---|---|---|---|---|---|
| Earthquake | Oakland draft values exist; Berkeley eligible values are incomplete | Fault proximity and CGS regulatory/hazard zones | Missing | None | CGS zone membership is available but is not overall risk | No | Adopted/reviewed jurisdiction evidence; shaking model; site/building vulnerability; validated loss or risk methodology; expert/legal review |
| Flood | Some jurisdiction planning context exists | FEMA mapped flood categories | Elevation, first-floor height, drainage, construction, insurance missing | None | FEMA flood-zone/SFHA category | No universal H/M/L | Preserve exact FEMA category; obtain validated depth/frequency and building-vulnerability methodology before any combined rating |
| Wildfire | Some jurisdiction planning context exists | CAL FIRE FHSZ severity category | Structure ignition, defensible space, roof/vents, access missing | None | CAL FIRE FHSZ category | No universal H/M/L | Preserve severity; add authoritative structure/community vulnerability and validated probability/loss method with expert review |
| Liquefaction | Regional earthquake context may exist | CGS liquefaction-zone membership | Foundation/soil/building response missing | None | CGS zone membership | No | Geotechnical/site data and a reviewed vulnerability/loss model; do not infer from zone membership alone |
| Earthquake-induced landslide | Regional earthquake context may exist | CGS earthquake-landslide zone membership | Slope, drainage, retaining structures missing | None | CGS zone membership | No | Site/slope and building vulnerability plus a reviewed consequence model |
| Tsunami | Draft Oakland planning categories may exist | CGS tsunami hazard-area membership | Elevation, evacuation ability, building vulnerability missing | None | Official tsunami hazard area | No | Preserve evacuation-area finding; authoritative intensity/depth plus vulnerability methodology would be required |
| Dam inundation | General/local-plan context may exist | Official scenario polygon match where available | Building and warning/evacuation vulnerability missing | None | Scenario inundation membership | No | Scenario provenance, applicability review, depth/timing where authoritative, and a validated consequence model |

## Minimum standard before an individual-address rating

StayReady would need all of the following—not simply more UI or code:

1. A hazard-specific scientific definition of the rating being claimed.
2. Authoritative exposure inputs with known dates, resolution, and uncertainty.
3. Relevant building/site/household vulnerability inputs collected lawfully and with consent.
4. A documented and externally reviewed method combining probability, exposure, vulnerability, and consequence without mixing incompatible categories.
5. Calibration and validation against suitable outcomes or authoritative benchmarks.
6. Claim-level citations, versioning, uncertainty disclosure, and review governance.
7. Legal, insurance, accessibility, privacy, and emergency-management review.

Until then, the product should present exact regional planning categories, mapped findings, proximity, non-matches, unavailable states, unknown vulnerability, and preparedness attention as separate concepts.
