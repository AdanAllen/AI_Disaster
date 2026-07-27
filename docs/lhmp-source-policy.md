# Local-plan source selection policy

For each jurisdiction, StayReady selects the newest official LHMP or
hazard-mitigation planning document published by the jurisdiction. An official
draft may be selected when it is the newest publication, provided every
resident-facing use identifies it as a draft and does not imply adoption,
finality, FEMA approval, or legal authority.

Each claim remains tied to one source document. Older adopted plans may remain
in provenance, but their statements cannot be silently combined with a newer
draft. Local-plan context is jurisdiction-level and never becomes an
address-level mapped finding or property-risk rating.

## Controlled selection reasons

- `latest_official_publication`
- `latest_official_draft`
- `current_adopted_plan`
- `official_source_unresolved`

## Selected publications

| Jurisdiction | Selected publication | Publication date | Status | Selection reason |
| --- | --- | --- | --- | --- |
| Alameda County | Alameda County Hazard Mitigation Plan | 2022 | Adopted | `current_adopted_plan` |
| Alameda | Climate Adaptation and Hazard Mitigation Plan | June 2026 | Final review draft | `latest_official_draft` |
| Albany | Draft Local Hazard Mitigation Plan Update | 2023 | Draft | `latest_official_draft` |
| Berkeley | Local Hazard Mitigation Plan | 2024 | Adopted | `current_adopted_plan` |
| Dublin | Tri-Valley LHMP, Volume 2: City Annexes | 2024 | Adoption unresolved | `latest_official_publication` |
| Emeryville | Local Hazard Mitigation Plan 2025-2030 | 2025 | Final draft | `latest_official_draft` |
| Fremont | Tri-City LHMP: Fremont Annex | 2024 | Adopted | `current_adopted_plan` |
| Hayward | Local Resilience Plan Update | 2021 | Adopted | `current_adopted_plan` |
| Livermore | Tri-Valley LHMP, Volume 2: City Annexes | 2024 | Adopted | `current_adopted_plan` |
| Newark | Tri-City LHMP: Newark Annex | 2024 | Adoption unresolved | `latest_official_publication` |
| Oakland | Draft Local Hazard Mitigation Plan 2026-2031 | 2026 | Draft | `latest_official_draft` |
| Piedmont | Published Local Hazard Mitigation Plan Update text | 2025 | Public-review text with separate 2026 adoption evidence | `latest_official_publication` |
| Pleasanton | Tri-Valley LHMP, Volume 2: City Annexes | 2024 | Adopted | `current_adopted_plan` |
| San Leandro | City Local Hazard Mitigation Plan, version 0.8 | July 1, 2026 | Public review draft | `latest_official_draft` |
| Union City | Tri-City LHMP: Union City Annex | 2024 | Adoption unresolved | `latest_official_publication` |

The machine-readable registry and checksums are in
`data/lhmp/plan_registry.json` and `data/lhmp/source_manifest.json`.
