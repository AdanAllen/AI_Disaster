# Oakland Draft 2026–2031 LHMP sub-area audit

**Audit date:** 2026-06-22  
**Source:** Draft Oakland 2026–2031 Local Hazard Mitigation Plan, 388-page PDF  
**Production claim:** sub-area planning context only; never property risk

## Conclusions

The draft plan uses nine sub-areas: Central East Oakland; Coliseum/Airport; Downtown; East Oakland Hills; Eastlake/Fruitvale; Glenview/Redwood Heights; North Oakland Hills; North Oakland/Adams Point; and West Oakland.

No draft-plan table assigns a separate High, Medium, Low, PRI, or comparable overall risk rating to each of the nine sub-areas. Table 17-1 (PDF page 316) assigns one **Oakland-wide** PRI value and risk ranking to each hazard. Its formula combines probability (30%), impact (30%), spatial extent (20%), warning time (10%), climate change (10%), and a qualitative adjustment factor for earthquake and wildfire. Those citywide values are not sub-area ratings.

The sub-area tables instead report exposure, modeled scenario consequences, building values/damage, displacement, shelter need, community lifelines, and Equity Priority Community counts. Hazard-map classes such as Very High FHSZ and high/very-high landslide susceptibility remain hazard/exposure classes, not overall sub-area risk.

## Boundary source and assignment rule

The automation gate passed because an official digital source exists:

| Field | Finding |
|---|---|
| Official source | City of Oakland ArcGIS REST, `Accela/Citywide_202410/MapServer/9`, **Plan Area CW** |
| URL | <https://gismaps.oaklandca.gov/server/rest/services/Accela/Citywide_202410/MapServer/9> |
| Edition | Service collection `Citywide_202410`; layer metadata says last updated 2018-06-27; fetched snapshot recorded 2026-06-17 |
| Geometry | Official polygon feature layer; repository GeoJSON snapshot has ten features |
| Source CRS | EPSG:2227, NAD83 / California zone 3 (US survey feet) |
| Stored geometry | Longitude/latitude GeoJSON |
| Coverage | Nine named city plan areas plus a separate `West Oakland - Gateway Industrial District (CFD)` polygon |
| LHMP relationship | Draft page 84 says Oakland's 2019 paving plan divided the city into these nine areas and staff selected them for hazard assessment. The official service describes the same nine names and their census-tract/geographic construction. |
| Rule | Assign only by exact point-in-polygon against one configured polygon. No ZIP, street, neighborhood, or nearest-name inference. Shared-boundary multiple matches, missing/invalid geometry, outside coverage, and the extra Gateway polygon fail closed to Unknown. |
| Limitation | The service originated in pedestrian/bicycle planning and predates the draft LHMP. Its identity with the LHMP figure is supported by names and stated planning lineage; a future City data-owner confirmation would further strengthen provenance. |

## Citywide PRI versus other evidence

1. **Citywide PRI:** Table 17-1 ranks each hazard for Oakland as a whole. It is draft and uses the plan's weighted formula.
2. **Hazard-map category:** an input such as Very High FHSZ or landslide susceptibility, not overall risk.
3. **Sub-area quantitative result:** a count, percentage, replacement value, modeled damage, displacement, shelter estimate, or lifeline count for one of nine areas.
4. **Sub-area risk rating:** none found in the draft plan.
5. **Address GIS finding:** a separate point intersection/proximity returned by FEMA, CAL FIRE, CGS, or DWR data.

The adopted 2021–2026 LHMP contains older scenario risk-ranking tables by plan area. Those are a different source edition and are not substituted for the draft plan's quantitative sub-area findings in this implementation. StayReady also does not average those older categories into a new draft-plan rating.

## Complete sub-area quantitative table inventory

Every table below has rows or columns by sub-area unless marked otherwise. Values, headers, scenarios, sources, and footnotes remain in the source PDF. The production dataset intentionally promotes only the visually reviewed, display-oriented tables identified under “Implemented evidence”; the remaining tables stay audit inventory until cell-level review.

| Hazard/context | Chapter | Table(s), PDF page(s) | Sub-area result | Classification | Principal LHMP sources / scenario | Important limitation |
|---|---|---|---|---|---|---|
| Baseline community | 4 | 4-5 p98 | EPC population by three thresholds | Equity/community context | City of Oakland 2023 | Not hazard-specific |
| Baseline buildings | 4 | 4-7 p102 | Building count and replacement value | Building exposure baseline | County assessor/Hazus | Not hazard-specific or building condition |
| Baseline lifelines | 4 | 4-9 to 4-25 pp104-119 | Counts by lifeline category/type | Community lifeline inventory | City facility datasets | Presence is not hazard exposure until overlaid |
| Dam failure | 7 | 7-6 p146 | Population/count/percent in combined inundation area | Population exposure | CA DWR; USACE; five named dam areas combined | Census-block intersection overestimates; simultaneous failure unlikely |
| Dam failure | 7 | 7-7 p147 | EPC population by threshold | Equity exposure | City EPC; combined inundation | Community context, not household vulnerability |
| Dam failure | 7 | 7-8 to 7-9 p148 | Exposed buildings and occupancy class | Building exposure | Hazus/combined inundation | Scenario exposure, not failure probability |
| Dam failure | 7 | 7-10 p149 | Exposed lifelines by category | Lifeline exposure | City lifelines/combined inundation | Counts do not quantify service failure |
| Dam failure | 7 | 7-11 to 7-12 p150 | Exposed value and Hazus anticipated building damage | Exposure / modeled damage | Hazus; combined inundation | Modeled combined scenario, not expected annual loss |
| Drought | 8 | 8-5 p163 | EPC population by threshold | Equity/community exposure | City of Oakland 2023; everyone treated as exposed | Repeats community distribution more than spatial hazard differentiation |
| Earthquake | 9 | 9-6 p186 | Displaced households and short-term shelter for Calaveras M6.86, Hayward M7.05, San Andreas M7.38 | Displacement / shelter | USGS 2025d; CGS 2025; Hazus 6.1 | Scenario estimates at census-tract GBS scale |
| Earthquake | 9 | 9-7 p187 | EPC population by threshold | Equity exposure | City of Oakland 2023 | Citywide earthquake exposure; not address intensity |
| Earthquake | 9 | 9-9 pp188-189 | Structure/content damage by three earthquake scenarios | Estimated damage | USGS/CGS; Hazus | Scenario loss, not address expected loss |
| Earthquake | 9 | 9-10 pp189-191 | Lifelines in high/very-high liquefaction susceptibility | Lifeline exposure | CGS/liquefaction overlay | Susceptibility class, not occurrence probability |
| Earthquake | 9 | 9-11 p192 | Debris by earthquake scenario | Other consequence | Hazus | Scenario output |
| Earthquake | 9 | 9-13 p193 | Annualized earthquake structure/content losses | Modeled damage/loss | Hazus probabilistic analysis | Aggregate model; not a property quote |
| Flood | 10 | 10-6 p211 | Population exposed to 1% and 0.2% annual-chance areas | Population exposure | FEMA 2018; Oakland/Hazus | Area exposure, not water entering a building |
| Flood | 10 | 10-7 p212 | Displaced households and short-term shelter by two scenarios | Displacement / shelter | FEMA; Oakland; Hazus 6.1 | Modeled scenario, not address expected loss |
| Flood | 10 | 10-8 to 10-9 pp213-214 | EPC exposure for 1% and 0.2% scenarios | Equity exposure | FEMA; Oakland EPC | Community context |
| Flood | 10 | 10-10 p215 | Buildings in SFHA | Building exposure | FEMA/Oakland/Hazus | SFHA exposure only |
| Flood | 10 | 10-11 p215 | Lifelines in flood areas | Lifeline exposure | FEMA/Oakland | Does not model outage/consequence |
| Flood | 10 | 10-12 to 10-13 pp216-217 | Exposed replacement value for 1% and 0.2% scenarios | Property exposure | Hazus/FEMA | Exposure is not damage |
| Flood | 10 | 10-14 p217 | Debris by scenario | Other consequence | Hazus | Scenario output |
| Flood | 10 | 10-15 to 10-16 p218 | Modeled damage for 1% and 0.2% scenarios | Estimated damage | Hazus | Not an address loss prediction |
| Landslide | 11 | 11-4 pp232-233 | Population in high and very-high deep-seated susceptibility | Population exposure | CGS; Oakland; Hazus | Broader than earthquake-induced landslide address layer |
| Landslide | 11 | 11-5 pp233-234 | EPC population in combined high/very-high areas | Equity exposure | CGS; Oakland EPC | Community context |
| Landslide | 11 | 11-6 pp234-235 | Buildings by high/very-high class | Building exposure | CGS/Hazus | Susceptibility, not damage |
| Landslide | 11 | 11-7 pp235-236 | Exposed lifelines | Lifeline exposure | CGS/Oakland | Does not model disruption |
| Landslide | 11 | 11-8 p236 | Exposed building/contents value | Property exposure | Hazus/CGS | Exposure, not expected loss |
| Sea-level rise | 12 | 12-2 p247 | Population exposed in 2050 and 2100 scenarios | Population exposure | Oakland 2025; RSAP 2025 | Time-horizon scenarios, not predictions |
| Sea-level rise | 12 | 12-3 to 12-4 pp248-249 | EPC exposure by scenario | Equity exposure | Oakland EPC/RSAP | Community context |
| Sea-level rise | 12 | 12-5 p250 | Buildings exposed by scenario | Building exposure | Oakland/RSAP/Hazus | Exposure, not damage |
| Sea-level rise | 12 | 12-6 to 12-7 pp250-251 | Exposed buildings by type | Building exposure | Oakland/RSAP/Hazus | No building-specific condition |
| Sea-level rise | 12 | 12-8 to 12-9 pp252-253 | Lifelines exposed by scenario | Lifeline exposure | Oakland/RSAP | Does not model outage |
| Sea-level rise | 12 | 12-10 to 12-11 p254 | Exposed replacement value by scenario | Property exposure | Hazus/RSAP | Not expected loss |
| Severe weather | 13 | 13-8 pp271-272 | EPC population by threshold | Equity/community exposure | Oakland 2023; everyone treated as exposed | Not event-specific probability or intensity |
| Severe weather | 13 | 13-9 pp273-274 | General building stock replacement values | Baseline property context | Hazus | Not weather damage or hazard exposure |
| Tsunami/seiche | 14 | 14-2 p286 | Population in tsunami hazard area | Population exposure | CGS 2025; Oakland; Hazus | Planning area, not probability |
| Tsunami/seiche | 14 | 14-3 p286 | Displaced households / short-term shelter | Displacement / shelter | FEMA/Oakland/Hazus | Scenario affecting entire hazard area |
| Tsunami/seiche | 14 | 14-4 p287 | EPC exposure | Equity exposure | Oakland EPC | Community context |
| Tsunami/seiche | 14 | 14-5 to 14-6 p288 | Structures and structure types exposed | Building exposure | CGS/Oakland/Hazus | Exposure, not damage |
| Tsunami/seiche | 14 | 14-7 p289 | Exposed lifelines | Lifeline exposure | Oakland/CGS | Does not model disruption |
| Tsunami/seiche | 14 | 14-8 pp289-290 | Building values exposed | Property exposure | Hazus/CGS | Exposure, not loss |
| Tsunami/seiche | 14 | 14-9 p290 | Estimated exposure and total damage value | Modeled damage | Hazus/CGS | Scenario result, not address expected loss |
| Wildfire | 15 | 15-5 pp304-305 | Population in adopted Very High FHSZ | Population exposure | CAL FIRE 2025a; Oakland; Hazus | FHSZ is hazard severity, not risk |
| Wildfire | 15 | 15-6 pp305-306 | EPC population in three-mile FHSZ buffer | Equity/other exposure | Oakland EPC; buffer analysis | Buffer is supplemental and not mapped FHSZ membership |
| Wildfire | 15 | 15-7 to 15-8 pp306-308 | Buildings and structure types in Very High FHSZ | Building exposure | CAL FIRE/Oakland/Hazus | Does not assess ignition vulnerability |
| Wildfire | 15 | 15-9 pp308-309 | Residential stock by construction date | Vulnerability proxy/context | Hazus | Age alone is not building risk |
| Wildfire | 15 | 15-10 p309 | Lifelines in FHSZ | Lifeline exposure | CAL FIRE/Oakland | Does not model failure |
| Wildfire | 15 | 15-11 p310 | Replacement value exposed | Property exposure | Hazus/FHSZ | Exposure, not damage |

### Maps and figures

Figure 4-1 (PDF page 85) shows the nine assessment sub-areas. Hazard figures are citywide spatial maps rather than tables that assign a statistic to each sub-area: dam inundation figures in Chapter 7; earthquake faults, shaking scenarios, liquefaction and UCERF3 figures in Chapter 9; FEMA flood areas in Figure 10-1; landslide susceptibility in Figure 11-1; sea-level-rise scenarios in Chapter 12; tsunami inundation in Figure 14-2; and FHSZ in Figure 15-1. They support the plan's overlays but do not create nine separate risk ratings.

## Implemented evidence and review gate

`data/hazard_priority/sub_area_evidence.json` schema version 3 contains 63 records: nine areas for each of seven selected hazards/context topics—dam failure, earthquake, flood, landslide, sea-level rise, tsunami, and wildfire. The displayed metric is deliberately limited to one clear table per hazard:

- Dam failure Table 7-6 population exposure.
- Earthquake Table 9-6 Hayward M7.05 displacement and shelter scenario.
- Flood Table 10-7 1% and 0.2% annual-chance displacement and shelter scenarios.
- Landslide Table 11-4 high/very-high susceptibility population exposure.
- Sea-level rise Table 12-2 population exposure in 2050/2100 scenarios.
- Tsunami Table 14-2 population in the tsunami hazard area.
- Wildfire Table 15-5 population in the Very High FHSZ.

Each record contains jurisdiction, stable sub-area ID/name, hazard ID, metric type, raw value, unit, scenario, scope, draft status, chapter/page/table, LHMP source, exact display claim, review status/method/date, permitted use, and limitations. Production accepts only records with an allowed review status, all required provenance fields, and `permitted_use: subarea_context_only`. Candidate/unreviewed extraction is withheld.

The records were visually checked against rendered source pages, including table headers, rows, units, sources, and notes. This review is explicitly recorded as Codex source-page visual verification, not represented as City adoption or external expert certification.

## Hazards with and without usable sub-area context

**Usable and implemented for existing canonical hazard rows:** earthquake, flood, wildfire, landslide, tsunami, and dam failure (when a dam-failure row is present). **Reviewed and structured but not added as a new hazard row:** sea-level rise.

**Audited but deliberately not promoted to the Risk Summary:** drought and severe weather. Their sub-area tables largely repeat EPC population distribution while treating the whole city as exposed, so they add community-demographic context but little hazard-specific address prioritization. Baseline buildings/lifelines are also not displayed without a hazard overlay.

## UI and claim behavior

The canonical hazard item now has an independent `subarea_context` object. The Risk Summary displays one concise reviewed finding, draft status, and exact table/page citation, with all reviewed details and limitations in an expansion below. It is not merged into `regional_context`, `address_evidence`, or `vulnerability`.

Presentation precedence is: positive address mapped finding; reviewed sub-area context; reviewed citywide priority; preparedness context; checked no match; unavailable. Both a positive map match and sub-area statistic remain visible. This hierarchy is display logic, not a scientific score.

## Remaining blockers and deferred work

- Obtain a named City data-owner confirmation that the current Plan Area CW geometry is the exact geometry used for draft Figure 4-1.
- Complete cell-level review before promoting any additional tables listed in the inventory.
- Do not expose old adopted-plan area ratings as if they were draft-plan findings; do not average them into a new rating.
- Sea-level rise is structured but awaits a supported canonical hazard row and address-evidence methodology before UI exposure.
- Drought/severe-weather EPC tables, lifeline details, building values, and modeled losses remain deferred to avoid statistics overload and false precision.
- Draft status must be revisited when Oakland adopts a final 2026–2031 plan.
