# Oakland Hazard-Priority Audit

Audit date: 2026-06-17

Purpose: document what the repository currently supports before publishing the generalized StayReady hazard-priority methodology. This audit does not invent missing values.

## Methodology Status

StayReady previously used a simplified LHMP-based hazard-priority methodology using only Probability and Impact. For Oakland address results, the production baseline now uses official adopted Oakland LHMP sub-area scenario Hazard Risk Ratings when a point can be matched to one official Oakland plan-area polygon.

The Oakland draft LHMP includes Table 17-1, "Natural Hazard Risk Ranking," and Table 17-2, "Priority Risk Index Scoring Criteria," on PDF pages 316-318 of `data/raw/lhmps/oakland/oakland-draft-lhmp-2026-2031.pdf`.

The adopted Oakland 2021-2026 LHMP includes Chapter 18 and Appendix E detailed risk-ranking tables. Chapter 18 states that risk was ranked for each hazard of concern by sub-area and that Appendix E contains the ranking of risk by sub-area. The production Oakland area-rating dataset is stored in `data/hazard_priority/oakland_lhmp_area_scenario_ratings.json`.

## Oakland Probability And Impact Support

| Hazard | Probability documented? | Impact documented? | Official PRI ranking documented? | Source status | Source page/table |
| --- | --- | --- | --- | --- | --- |
| Earthquake | Yes. Table value 1.2 maps to High probability under StayReady's simplified categories. | Yes. Table value .60 maps to Medium impact. | High | Draft | Draft Oakland 2026-2031 LHMP, PDF page 316, Table 17-1 |
| Wildfire | Yes. Table value .90 maps to High probability. | Yes. Table value .60 maps to Medium impact. | High | Draft | Draft Oakland 2026-2031 LHMP, PDF page 316, Table 17-1 |
| Flood | Yes. Table value 1.2 maps to High probability. | Yes. Table value .60 maps to Medium impact. | Medium | Draft | Draft Oakland 2026-2031 LHMP, PDF page 316, Table 17-1 |
| Landslide | Yes. Table value .60 maps to Medium probability. | Yes. Table value .60 maps to Medium impact. | Medium | Draft | Draft Oakland 2026-2031 LHMP, PDF page 316, Table 17-1 |
| Tsunami | Yes. Table value .60 maps to Medium probability. | Yes. Table value .30 maps to Low impact. | Low | Draft | Draft Oakland 2026-2031 LHMP, PDF page 316, Table 17-1 |

## Adopted Versus Draft Sources

Adopted Oakland source support currently present in the repository is limited to older plan/context references such as `city_hazard_chunks.json` entries labeled "City of Oakland 2021-2026 Local Hazard Mitigation Plan." Those reviewed chunks support general hazard relevance for earthquake, flood, and wildfire, but they do not provide the Probability and Impact values needed for the new simplified matrix.

The Probability, Impact, and official PRI ranking values above come from the draft Oakland 2026-2031 LHMP. The UI and API must label these as Draft.

## Unsupported Existing Values

The first-pass `lhmp_rating` values in `data/hazard_rankings/city_hazard_rankings.json` are superseded by the generalized Probability/Impact data. Any Low/Medium/High hazard priority not backed by explicit Probability and Impact, or an explicitly documented official fallback, should be treated as Unknown.

Legacy ZIP CSV numeric values, `priority_score`, `local_risk_score`, and any 1-10 values are unsupported for the new hazard-priority methodology.

## Official Oakland Area Scenario Ratings

Adopted source: `data/raw/lhmps/oakland/oakland-lhmp-2021-2026-adopted.pdf`.

Structured source file: `data/hazard_priority/oakland_lhmp_area_scenario_ratings.json`.

Complete matrix output: `docs/oakland_lhmp_area_rating_matrix.md`.

| Hazard | Official scenarios extracted | Source pages/tables | Notes |
| --- | --- | --- | --- |
| Earthquake | EQ Calaveras M6.86; EQ Haywired M7.05; EQ San Andreas M7.38; EQ 100-yr Prob | Appendix E, pages 474, 476, 478, 480; RISK RANKING-Earthquake tables | Multiple adopted scenario ratings are combined by unweighted category average for each plan area. |
| Flood | 100-year / 1 percent annual chance flood; 500-year / 0.2 percent annual chance flood | Appendix E, pages 460 and 462; RISK RANKING-100-yr Flood and RISK RANKING-500-yr Flood | Multiple adopted scenario ratings are combined by unweighted category average for each plan area. |
| Wildfire | Wildfire Very High and High severity | Appendix E, page 472; RISK RANKING-Wildfire table | One adopted scenario rating per plan area; no averaging required. |
| Landslide | High and Very High landslide susceptibility | Appendix E, page 464; RISK RANKING-Landslide Susceptibility table | One adopted scenario rating per plan area; no averaging required. |
| Tsunami | Draft Tsunami Hazard Area | Appendix E, page 466; RISK RANKING-Draft Tsunami Hazard Area table | The table title says Draft Tsunami Hazard Area, but it appears inside the adopted 2021-2026 LHMP Appendix E. It is stored with adopted document status and the scenario name preserves the draft-hazard-area wording. |

Combination rule: Low = 1, Medium = 2, High = 3. Unknown scenarios are excluded. Averages of 1.00-1.49 display Low, 1.50-2.49 display Medium, and 2.50-3.00 display High. The internal numeric score is not displayed as exact risk or probability.

Current production behavior: for a matched Oakland plan area, `displayed_hazard_level` uses the combined official area scenario rating. `stayready_fallback_rating` is preserved but not displayed as the primary Oakland baseline when official area scenario ratings exist.

Missing area behavior: if Oakland coordinates cannot be matched to exactly one configured plan area, `plan_area: Unknown`, `official_lhmp_area_rating: Unknown`, and `displayed_hazard_level: Unknown`. ZIP code, neighborhood text, and nearest-label matching are not used.

## Official GIS Layer Status

Working or registered official/source-backed GIS checks in this repository:

| Hazard | Layer | Status in repo | Notes |
| --- | --- | --- | --- |
| Flood | FEMA NFHL local snapshot | Provisional local snapshot | Checks address-point polygon intersections. Exact download lineage and official viewer comparison remain unverified. |
| Wildfire | CAL FIRE Fire Hazard Severity Zones local snapshot | Provisional local snapshot | Preserves hazard severity terminology when present. Exact edition and viewer comparison remain unverified. |
| Earthquake | USGS/CGS fault traces local snapshot | Provisional local snapshot | Used only for fault proximity context, not fault-zone membership. |
| Earthquake | CGS Alquist-Priolo Earthquake Fault Zones remote service | Provisional official remote service | Official service registered; non-match does not lower citywide earthquake priority. |
| Earthquake | CGS Liquefaction Zones remote service | Provisional official remote service | Adds local ground-failure concern when matched. |
| Landslide | CGS Earthquake-Induced Landslide Zones remote service | Provisional official remote service | Adds local landslide/ground-failure concern when matched. |
| Tsunami | CGS Tsunami Hazard Area remote service | Provisional official remote service | Tsunami local status must come from this official hazard-area intersection. |

Unstable/unavailable conditions: any remote service request can fail. A failed request must return Data unavailable and must not lower or raise a citywide priority.

## Oakland Sub-Area Boundaries

The Oakland draft LHMP uses sub-area names including Central East Oakland, Coliseum/Airport, Downtown, East Oakland Hills, Eastlake/Fruitvale, Glenview/Redwood Heights, North Oakland Hills, North Oakland/Adams Point, and West Oakland.

An official Oakland ArcGIS REST layer was identified and fetched on 2026-06-17:

- Source: City of Oakland ArcGIS REST Services, Accela/Citywide_202410 MapServer layer 9, "Plan Area CW."
- URL: https://gismaps.oaklandca.gov/server/rest/services/Accela/Citywide_202410/MapServer/9
- Local dataset: `data/hazard_priority/oakland_plan_areas.geojson`
- Match field: `OAKLAND_PE`
- Source note: the layer describes nine pedestrian planning areas developed from Oakland census tracts and adopted in 2017.

Current required behavior: assign an Oakland sub-area only by point-in-polygon against this official layer. If coordinates are unavailable, if the point falls outside all assignable polygons, or if the match is ambiguous, return `sub_area: Unknown`. Do not infer sub-area from ZIP code, neighborhood name, street name, or nearest-label matching.

Layer caveat: the official layer contains an additional polygon named "West Oakland - Gateway Industrial District (CFD)." It is not one of the nine configured LHMP sub-areas, so the code does not silently merge it into West Oakland.

## Source Pages And Tables

The current publishable Oakland citywide hazard-priority values are supported by:

- Draft Oakland 2026-2031 LHMP, PDF page 316, Table 17-1, Natural Hazard Risk Ranking.
- Draft Oakland 2026-2031 LHMP, PDF pages 317-318, Table 17-2, Priority Risk Index Scoring Criteria.

Sub-area context currently extracted for display after an official sub-area match:

- Draft Oakland 2026-2031 LHMP, PDF page 102, Table 4-7, Distribution of Buildings in the Planning Area by Sub-Area.
- Draft Oakland 2026-2031 LHMP, PDF page 104, Table 4-9, Community Lifelines in the City of Oakland.
- Draft Oakland 2026-2031 LHMP, PDF page 186, Table 9-6, Estimated Earthquake Impact on Persons and Households.
- Draft Oakland 2026-2031 LHMP, PDF page 187, Table 9-7, Estimated Number of Equity Priority Community Members Exposed to Earthquake.
- Draft Oakland 2026-2031 LHMP, PDF page 212, Table 10-7, Estimated Flood Impact on Persons and Households.
- Draft Oakland 2026-2031 LHMP, PDF pages 213-214, Tables 10-8 and 10-9, Estimated Number of Equity Priority Community Members Exposed to Flood.
- Draft Oakland 2026-2031 LHMP, PDF pages 232-233, Table 11-4, Estimated Number of Persons Located in the High and Very High Landslide Susceptibility Areas.
- Draft Oakland 2026-2031 LHMP, PDF pages 233-234, Table 11-5, Estimated Number of Equity Priority Community Members Exposed to Landslide.
- Draft Oakland 2026-2031 LHMP, PDF pages 286-287, Tables 14-2 through 14-4, Tsunami population, household, and EPC exposure.
- Draft Oakland 2026-2031 LHMP, PDF pages 304-306, Tables 15-5 and 15-6, Wildfire Very High FHSZ population and EPC buffer exposure.

Sub-area table values are stored with `permitted_use: community_context_only`. They must not alter LHMP Probability, Impact, or local hazard priority. A zero EPC value is not a zero-hazard conclusion.

## Physical Versus Community Context

StayReady keeps three layers distinct:

1. Citywide Probability/Impact from LHMP PRI tables.
2. Address-specific physical exposure from official GIS outputs already computed by the app.
3. Sub-area community impact/vulnerability context from LHMP sub-area tables.

Physical exposure examples: FEMA flood-zone annual-chance categories, CAL FIRE/official FHSZ severity categories, CGS tsunami hazard areas, and CGS landslide/liquefaction/fault-related layers. These may change local mapped concern when they directly apply, but they do not rewrite the citywide LHMP Probability and Impact values.

Community context examples: EPC exposure counts, displaced-household estimates, sub-area population in susceptibility areas, and shelter estimates. These may explain community planning context, but they do not assign individual risk or change the local hazard priority.

## Data Still Required

- Human review or adopted-source confirmation for Oakland Probability and Impact if draft values should not be used in production labels.
- Human verification of local FEMA, CAL FIRE, and fault local snapshot lineage against official viewers.
- Additional sub-area table extraction if the UI needs a complete row for every sub-area and every metric, beyond the current display-oriented records.
