# Alameda County LHMP geographic-context audit and countywide design

**Audit date:** 2026-06-23  
**Scope:** every enabled entry in `data/lhmp/plan_registry.json`, plus repository-held base plans, older editions, special-district annexes, and unincorporated Alameda County  
**Product boundary:** planning context only; never a property-risk calculation  
**Implementation status:** design and source audit only; no production code, UI, or canonical Risk Summary contract was changed

## 1. Executive conclusion

The county cannot be represented honestly by one Oakland-shaped data model or one translated High/Medium/Low scale. The fifteen enabled registry entries use at least six materially different evidence patterns:

1. formal ratings for named local areas (adopted Oakland 2021-2026);
2. quantitative local-area scenario tables without local-area ratings (draft Oakland 2026-2031);
3. jurisdiction-wide composite rankings (Tri-Valley and draft Oakland);
4. jurisdiction-wide multi-dimensional assessments without a composite score (Alameda 2022, Berkeley 2024, Albany draft, Emeryville draft, Piedmont, and Tri-City annexes);
5. quantitative exposure tables for a county sub-jurisdiction, notably Unincorporated Alameda County; and
6. narrative or mapped hazard context without an LHMP rating suitable for inheritance (Hayward and San Leandro's repository source).

Oakland is presently the only jurisdiction with all three ingredients needed for reliable local-area assignment: a plan-defined subdivision system, structured area-specific evidence, and a usable official polygon source that has been source-page checked. Even there, editions must remain separate: the adopted 2021-2026 plan supplies formal area/scenario ratings, while the draft 2026-2031 plan supplies citywide PRI rankings and sub-area scenario/exposure statistics but no formal sub-area rating.

The immediate countywide production opportunity is therefore **not** to add local-area claims everywhere. It is to build a flexible, edition-aware evidence contract and then onboard reviewed jurisdiction-wide ratings in a controlled batch. Alameda must wait for ingestion of its adopted 2025 update. Albany, Emeryville, Oakland 2026, and Piedmont 2025 must remain visibly draft until adoption is confirmed. San Leandro needs its standalone LHMP, not the Environmental Hazards Element currently registered.

The repository's machine extraction is broad but is not reviewed evidence. `data/lhmp/reviewed/` contains no public-claim records. Except for the separately source-page-verified Oakland datasets, no extracted jurisdictional value is currently eligible for a public claim.

### Decision rules from the audit

- Address GIS evidence, local-area plan evidence, jurisdiction-wide ratings, and regional preparedness context are independent claims.
- A jurisdiction-wide rating can be shown for an address only with explicit jurisdiction-wide wording; it is never an address mapped finding.
- A local-area value can be inherited only after exact point-in-polygon assignment to the official area to which the plan applies.
- Hazard-map classes such as FHSZ, liquefaction susceptibility, flood zones, or tsunami areas are not automatically LHMP overall-risk ratings.
- Scenario losses and exposure counts remain scenario/community context and are never converted to a property prediction or a new High/Medium/Low label.
- Adopted and draft editions may coexist only as separately cited claims. Values are never merged across editions.

## 2. Complete plan inventory

The registry contains Alameda County plus all fourteen incorporated cities in Alameda County. Page counts are physical PDF pages. “Candidate extraction” means machine-extracted and non-publishable. The normal FEMA update cycle is five years; exact expiration depends on the final FEMA approval date and must be stored when known.

| Jurisdiction | Registry source and edition | Pages | Plan relationship | Current-source finding as of audit | Multiple editions | Extraction / verification | Public-claim eligibility |
|---|---|---:|---|---|---|---|---|
| Alameda County | *Alameda County Local Hazard Mitigation Plan*, document calls itself 2021; registry labels 2022 | 161 | County + ACFD + ACFCWCD multi-jurisdiction plan | Board adopted 2022-03-01; FEMA final approval 2022-03-10; expected update by 2027-03-10. County General Plan page now marks the item archived. | No newer county plan in repo | Candidate extraction only | No, until claims are source-page verified |
| Alameda | *Climate Adaptation and Hazard Mitigation Plan*, 2022 | 402 | City-specific integrated climate/LHMP | Superseded. The City adopted a 2025 update on 2025-04-01; the repository does not contain it. The 2022 plan was adopted 2022-06-07 and FEMA approved 2022-06-15. | Older 2022 source + missing adopted 2025 replacement | Candidate extraction for superseded source | No; current source missing |
| Albany | *Draft Local Hazard Mitigation Plan Update*, 2023 | 100 | City-specific | Repository and official registry URL expose a draft that anticipated adoption in 2024; no executed adoption source was found in this audit. Treat status as unresolved draft, not adopted. | 2018 adopted predecessor referenced but absent | Candidate extraction only | No; draft and unverified |
| Berkeley | *Local Hazard Mitigation Plan*, 2024 | 1,210 | City-specific final plan plus appendices/presentations | Adopted by City Council 2025-03-18. Current official source. Update expected within five years of FEMA approval. | 2019 and 2014 predecessors available officially but not audited as current | Candidate extraction only | No, until claim-level verification |
| Dublin | *Tri-Valley LHMP, Volume 2: City Annexes*, 2024 | 211 shared | Multi-jurisdiction annex; Volume 1 contains common method | FEMA approved 2023-12-15. Livermore officially records adoption 2024-03-11; exact Dublin resolution date should be captured before release. | Replaces adopted 2018 plan | Candidate extraction only | No, until adoption/source pages are verified |
| Emeryville | *LHMP 2025-2030 Final Draft* | 346 | City-specific | Final draft in repository; document says adoption follows Cal OES/FEMA approval pending adoption. No executed 2025/2026 adoption record found. The adopted 2019 plan is referenced but absent. | Adopted 2019 predecessor + newer final draft | Candidate extraction only | No; draft and unverified |
| Fremont | *Tri-City MJLHMP: Fremont Annex*, April 2024 | 441 | City annex; separate 485-page base plan | Official sources identify a FEMA-approved plan/additional-jurisdiction approval dated 2024-11-21. Exact local adoption resolution should be attached to the source record. | Replaces 2016 plan | Candidate extraction only | No, until adoption and claim pages are verified |
| Hayward | *Local Resilience Plan Update*, 2021 | 199 | City-specific LHMP/LRP | Official City page describes it as the required five-year update and Fire Department page calls it newly adopted. A new update will soon be due or may already be in procurement; check FEMA approval date before release. | 2015/2016 predecessor incorporated in update | Candidate extraction only | No, until current status and claims are verified |
| Livermore | *Tri-Valley LHMP, Volume 2: City Annexes*, 2024 | 211 shared | Multi-jurisdiction annex | Adopted 2024-03-11 after FEMA approval 2023-12-15. Current official source. | Replaces adopted 2018 plan | Candidate extraction only | No, until claim pages are verified |
| Newark | *Tri-City MJLHMP: Newark Annex*, April 2024 | 274 | City annex; separate base plan | FEMA-approved plan is officially posted; exact Newark adoption resolution/date was not captured in the repository. | Replaces earlier Tri-City plan | Candidate extraction only | No, until adoption and claim pages are verified |
| Oakland | *Draft LHMP 2026-2031* | 388 | City-specific draft | Still draft. Official page says the 2021-2026 plan was adopted 2021-06-15 and that the 2026 plan is in final adoption steps; Planning Commission recommended adoption 2026-05-20. | Adopted 2021-2026 PDF and newer draft both in repo | General extraction candidate; selected draft sub-area tables and adopted area ratings have source-page/internal verification | Eligible only for the already gated, explicitly labeled records; draft status must remain visible |
| Piedmont | *Local Hazard Mitigation Plan*, April 2019 | 426 | City-specific | Repository copy contains an unexecuted sample resolution even though plan text says adopted. Official City page now exposes a July 2025 draft update that is not in the registry. Adoption of that update was not confirmed. | 2019 source + missing 2025 draft | Candidate extraction only | No; current edition absent and status unresolved |
| Pleasanton | *Tri-Valley LHMP, Volume 2: City Annexes*, 2024 | 211 shared | Multi-jurisdiction annex | FEMA approved 2023-12-15; official regional source calls the 2024 plan adopted. Exact Pleasanton resolution date should be recorded. | Replaces adopted 2018 plan | Candidate extraction only | No, until adoption/source pages are verified |
| San Leandro | *Environmental Hazards Element*, 2025 | 84 | General Plan element, not standalone LHMP | Incomplete for this product. The City issued a 2024 RFP for a standalone LHMP with work expected through FEMA approval; no final standalone plan is registered. | Older/general context only | Candidate extraction only | No; correct source missing |
| Union City | *Tri-City MJLHMP: Union City Annex*, April 2024 | 296 | City annex; separate base plan | FEMA-approved plan is officially posted by the regional project; exact Union City resolution/date should be attached. | Replaces earlier Tri-City plan | Candidate extraction only | No, until adoption/source pages are verified |

### Repository sources outside the fifteen registry rows

- `tri-valley-lhmp-2024-volume-1.pdf`: planning-area-wide method and risk assessment required to interpret the three city annexes.
- `tri-city-lhmp-2024-base-plan.pdf`: common definitions, hazard profiles, and vulnerability methodology required to interpret Fremont, Newark, and Union City.
- Tri-City annexes D and E: Alameda County Water District and Union Sanitary District. These special districts are planning partners, not address jurisdictions. Their service-area context may be useful later but must not replace city/county assignment.
- `oakland-lhmp-2021-2026-adopted.pdf`: current adopted Oakland evidence until the 2026-2031 plan is actually adopted.

### Review-status vocabulary

Use composable status fields rather than one ambiguous `reviewed` flag:

| Dimension | Allowed values and meaning |
|---|---|
| Extraction | `candidate_extraction`, `source_page_verified`, `rejected` |
| Internal review | `not_started`, `internally_verified` |
| External review | `not_requested`, `expert_reviewed`, `agency_confirmed` |
| Source status | `adopted_source`, `draft_source`, `superseded`, `archived`, `status_unresolved` |
| Eligibility | `withheld`, `eligible_for_scope`, `rejected` |

`source_page_verified` says only that a human checked the value against the page. It does not imply technical expert review, agency confirmation, adoption, or permission to use the value outside its recorded scope.

## 3. Jurisdiction capability matrix

Two levels are shown because source capability and release readiness are different facts. “Candidate capability” describes what the currently available plan appears capable of supporting. “Effective level now” applies the source-freshness and review gate; under the requested definitions, unverified or superseded material is Level E.

| Jurisdiction | Candidate capability | Effective level now | Jurisdiction rating | Local formal rating | Local statistics | Official polygons | Assignment ready | Priority | Main blocker |
|---|---|---|---|---|---|---|---|---|---|
| Alameda County / unincorporated | B | E | No composite | No | Yes, unincorporated exposure totals | PAM boundary referenced; source not registered | No | 3 | Verify PAM geometry/edition and tables |
| Alameda | C (2022) | E | Likelihood + consequence | No | City/hazard-wide scenario tables | Hazard layers only | N/A | 4 | Ingest adopted 2025 plan; 2022 superseded |
| Albany | C | E | Composite city ranking | No | Limited citywide context | No plan subdivisions | N/A | 3 | Adoption unresolved; claim review |
| Berkeley | C | E | Likelihood + impact severity | No | Mostly city/hazard-wide | Fire zones exist but have no LHMP area rating | N/A | 2 | Source-page verification |
| Dublin | C | E | Composite city ranking | No | Citywide/hazard-zone totals | No rated local-area system | N/A | 2 | Adoption record + source review |
| Emeryville | C | E | Probability + severity | No | Citywide/hazard-zone context | No rated local-area system | N/A | 4 | Draft; adoption unresolved |
| Fremont | C | E | Five-dimension city assessment | No | Extensive citywide hazard-overlay tables | Neighborhood map not tied to ratings | No | 2 | Source review; no local rated units |
| Hayward | D | E | No structured overall rating | No | Hazard-zone and asset context, not stable plan-area rows | Official hazard layers, no LHMP area system | No | 4 | Current-plan status and structured claim design |
| Livermore | C | E | Composite city ranking | No | Citywide/hazard-zone totals | No rated local-area system | N/A | 2 | Source-page verification |
| Newark | C | E | Five-dimension city assessment | No | Extensive citywide hazard-overlay tables | Census tracts/DAC maps are not rating areas | No | 2 | Adoption record + source review |
| Oakland | A adopted + B draft | A/B for gated records only | Adopted and draft citywide systems | Yes, adopted scenario ratings | Yes, draft nine-area tables | Official Plan Area CW polygons | Yes, fail-closed | 1 | Final 2026 adoption; agency geometry confirmation |
| Piedmont | C (2019) | E | Five-dimension city assessment | No | Citywide/hazard-zone totals | No rated local-area system | N/A | 4 | Missing 2025 draft; 2019 resolution ambiguity |
| Pleasanton | C | E | Composite city ranking | No | Citywide/hazard-zone totals | No rated local-area system | N/A | 2 | Adoption record + source review |
| San Leandro | D | E | No LHMP structured rating | No | General Plan context only | Hazard maps, not LHMP areas | No | 5 | Standalone LHMP missing |
| Union City | C | E | Five-dimension city assessment | No | Extensive citywide hazard-overlay tables | No rated local-area system | No | 2 | Adoption record + source review |

No jurisdiction other than Oakland qualifies for a new Level A production implementation from repository evidence. The next production batch should therefore be Level C jurisdiction-wide claims, not invented sub-area claims.

## 4. Geographic subdivision inventory

| Plan | Exact subdivision or geographic unit | Count / names | Boundary evidence | Coverage / overlap | Reliable address assignment? | LHMP relationship and policy |
|---|---|---|---|---|---|---|
| Alameda County 2021 | `Unincorporated Alameda County`; plan also includes ACFD and ACFCWCD service/operational areas | One county sub-jurisdiction plus two special districts | PDF says population was intersected with updated `PAM` unincorporated boundary; CRS and digital URL absent | Unincorporated area excludes cities; special-district areas may overlap jurisdictions | Not yet | Register the exact PAM dataset, CRS, edition, and access date. Do not substitute Eden/Castro Valley/East County General Plan areas unless the LHMP explicitly reports values for them. |
| County General Plan, contextual only | Eden Area, Castro Valley Area, East County; Eden includes Ashland, Cherryland, Hayward Acres, San Lorenzo, and Fairview | Three broad planning systems plus named communities | Official County planning maps/pages; digital polygon lineage not audited here | General Plan coverage, not shown to be LHMP rating units | No for LHMP claims | These names may support jurisdiction display, never an inherited LHMP rating without a plan-table relationship. |
| Alameda 2022 | Alameda Island, Bay Farm Island and coastal priority locations | Hazard/study geographies, not a universal rating system | Static maps and hazard datasets | Hazard-specific, may overlap | No common local-area assignment | Treat as scenario/map context only. Re-audit the adopted 2025 plan. |
| Albany draft | Albany Hill, shoreline and commercial districts appear in narrative | No structured rating-area list | Static/narrative | Partial; not a complete rating geography | No | Do not infer a local area from neighborhood names. |
| Berkeley 2024 | Fire Zones 1, 2, 3; census/community maps; hazard-specific zones | Three fire zones; other units vary | Adopted fire-zone boundaries and static maps; no area-rating table | Fire zones cover the city for fire administration; other systems vary | Technically possible, but irrelevant to an LHMP area rating | The plan gives citywide hazard likelihood/severity, not separate fire-zone LHMP ratings. |
| Tri-Valley | City jurisdictions and DSRSD service area | Dublin, Livermore, Pleasanton, DSRSD | Official jurisdiction/service boundaries | Jurisdictions/service area may overlap | Jurisdiction assignment only | Annex rankings are jurisdiction-wide. Dublin General Plan planning areas and GHADs are not Tri-Valley LHMP rating units. |
| Emeryville draft | Peninsula, shoreline, redevelopment areas and hazard zones | No complete rated subdivision system | Static/hazard maps | Hazard-specific | No | Citywide probability/severity only. |
| Fremont annex | Figure A-2 `Fremont Neighborhoods`; census tracts/DACs; future-development sites; hazard zones | Neighborhood count/names not enumerated in a rating table | Figure A-2 is static; no authoritative neighborhood polygon source tied to ratings was verified | Neighborhoods appear citywide; hazard units overlap | No | No rating or quantitative table is keyed to the neighborhood map. Do not digitize the PDF. |
| Newark annex | Census tracts/DACs; future-development sites; hazard zones | Varies | Official census/GIS inputs, but no LHMP local-rating system | Overlapping hazard/study units | No | Census social-vulnerability displays do not create tract-level LHMP hazard ratings. |
| Union City annex | Census tracts/DACs; three named future-development sites; hazard zones | Three future sites; other units vary | City GIS and census sources | Partial and overlapping | No | Site exposure tables concern development review, not household or neighborhood ratings. |
| Hayward 2021 | Hayward Hills, shoreline, flatlands, tsunami evacuation zone, hazard-specific mapped zones | Hazard-specific, no single area system | Maps cite CGS, FEMA, ABAG/BCDC and others; CRS often WGS 1984 Web Mercator Auxiliary Sphere | Overlapping and not universal | Not as LHMP local areas | Address evidence should come from validated current GIS layers, not PDF-map approximation. |
| Oakland adopted/draft | Nine `Sub-Areas`: Central East Oakland; Coliseum/Airport; Downtown; East Oakland Hills; Eastlake/Fruitvale; Glenview/Redwood Heights; North Oakland Hills; North Oakland/Adams Point; West Oakland | Nine plan areas | City ArcGIS `Accela/Citywide_202410/MapServer/9`, layer `Plan Area CW`; source EPSG:2227, stored GeoJSON lon/lat; service lineage says 2018-06-27 update, snapshot 2026-06-17 | Nine areas cover the intended city geography; source also includes an extra Gateway Industrial District polygon that is not silently merged | Yes, with fail-closed rules | Names and planning lineage match the LHMP. Obtain agency confirmation that the service geometry is exactly the plan geometry. |
| Piedmont 2019 | Planning area is the city; hazard zones only | No smaller formal system | Static/GIS hazard inputs | Hazard-specific | No | Citywide significance only. Re-audit 2025 draft. |
| San Leandro 2025 element | Named neighborhoods, shoreline, dam/tsunami/flood zones | No LHMP subdivision system | General Plan maps/narrative | Partial/overlapping | No | Wait for standalone LHMP. |

### Boundary ambiguity policy

An assignment is valid only when one authoritative polygon contains the address point under a documented spatial predicate. A point matching zero areas is `out_of_coverage`; more than one is `ambiguous` unless the source itself defines precedence. Points on shared boundaries are `boundary_ambiguous`, not guessed. Invalid/missing geometry is `data_unavailable`. No nearest polygon, ZIP, neighborhood string, street name, geocoder label, or manually traced PDF polygon may be used.

## 5. Formal rating inventory

The tables below inventory the formal systems found. Labels are preserved exactly and are not translated across plans.

### Rating systems and methods

| Plan / scope | Formal method | Exact labels / formula | Source page/table | Address inheritance |
|---|---|---|---|---|
| Alameda 2022, citywide | Separate likelihood and consequence, plus `Hazards of Greatest Concern` / `Hazards of Concern` grouping | Likelihood: Likely/Possible/Unlikely. Consequence: Moderate, Catastrophic, or range. No numeric composite. | PDF p59, Table 4-1 | May be displayed only as Alameda-wide context; source is superseded |
| Albany 2023 draft, citywide | Probability + population/property/economy severity scores | Components 0-3; Total is their sum; ranking is ordinal 1-5, not a probability. | PDF pp19-21, hazard ranking table | Citywide only; draft |
| Berkeley 2024, citywide | Separate likelihood and severity of impact | Likely/Possible; Minor/Moderate/Major/Catastrophic and ranges | Main hazard sections; summary reproduced in appendices/presentations, e.g. PDF p716 | Citywide only |
| Tri-Valley 2024, each city | Probability factor × sum of weighted impacts | Probability 0-3; people impact ×3, property ×2, economy ×1. Annex gives numeric score and High/Medium/Low risk category. | Vol. 1 pp262-265; Vol. 2 Tables 1-12, 2-12, 3-12 | Jurisdiction only |
| Emeryville 2025-2030 final draft, citywide | Separate probability and severity assigned by planning team | Probability High = every 1-10 years, Medium = every 10-50, Low = >50; severity Low/Medium/High and ranges | PDF p26 onward, hazard headings | Citywide only; draft |
| Tri-City 2024, each city | Five separate columns: geographic extent, likelihood, magnitude/severity, significance, climate-change influence | Extent Limited/Significant/Extensive; likelihood Highly Likely/Likely/Occasional/Unlikely; magnitude Negligible/Limited/Critical/Catastrophic; significance Low/Medium/High (some source cells use Med-High); climate influence Low/Medium/High | Fremont Table A-22 p50; Newark Table B-18 p37; Union City Table C-19 p35 | Jurisdiction only |
| Oakland 2021-2026 adopted, nine areas | Scenario risk-ranking tables by plan area | Official Low/Medium/High scenario ratings. Multiple scenario values must remain separately cited; any combined display rule is StayReady presentation logic, not a new official score. | Appendix E pp460-480 | Yes, after exact area match and explicit adopted/scenario wording |
| Oakland 2026 draft, citywide | PRI = weighted probability (30%) + impact (30%) + spatial extent (20%) + warning time (10%) + climate change (10%) + QAF | PRI 1-1.9 Low, 2-2.9 Medium, >=3 High; earthquake/wildfire receive 2.0 QAF | PDF pp316-318, Tables 17-1 and 17-2 | Citywide only; draft |
| Piedmont 2019, citywide | Same five-dimension assessment family as Tri-City, no numeric composite | Exact extent, likelihood, magnitude/severity, significance, climate influence labels | PDF pp4-5 / Table ES-2; Table 4-2 | Citywide only |
| Alameda County, Hayward, San Leandro repository source | No qualifying overall formal hazard rating found | Narrative, probability statements, exposure tables, map categories, or mitigation-action priorities are not an overall rating | Various | No |

### Exact citywide results

These values are audit findings, not approved production records.

**Alameda 2022 Table 4-1:** earthquake—Likely/Catastrophic; storm flooding—Likely/Moderate to Catastrophic; sea-level rise—Likely/Catastrophic; tsunami—Possible/Moderate to Catastrophic; heat—Likely/Moderate; drought—Likely/Moderate; wildfire-related smoky air/PSPS—Likely/Moderate; dam breach—Unlikely/Moderate.

**Albany 2023 draft:** earthquake 12/rank 1; public-health epidemic 12/1; critical infrastructure/utilities failure 9/2; flooding 8/3; wildland/urban fire 8/3; extreme temperatures 7/4; damaging winds 7/4; terrorism/mass violence 7/4; hazardous air 7/4; hazardous-material release 6/5; slope failure/landslide 6/5. The total is probability + three severity dimensions; it is not comparable to any other plan's score.

**Berkeley 2024:** earthquake Likely/Catastrophic; WUI fire Likely/Catastrophic; extreme heat Likely/Moderate to Catastrophic; poor air quality Likely/Minor; high wind Likely/Minor; rainfall-triggered landslide Likely/Minor to Catastrophic; tsunami Possible/Minor to Catastrophic; sea-level rise Likely/Minor to Major; floods Likely/Minor; utility interruption Likely/Minor to Major; hazardous-material release Likely/Minor to Catastrophic; infectious disease Likely/Minor to Catastrophic.

**Tri-Valley 2024:**

| Hazard | Dublin score/category | Livermore score/category | Pleasanton score/category |
|---|---|---|---|
| Earthquake | 36 High | 34 High | 36 High |
| Severe Weather | 33 Medium | 33 Medium | 33 Medium |
| Landslide | 28 Medium | 12 Low | 22 Medium |
| Flood | 15 Low | 15 Low | 15 Low |
| Wildfire | 10 Low | 6 Low | 36 High |
| Drought | 9 Low | 9 Low | 9 Low |
| Dam Failure | 6 Low | 8 Low | 18 Medium |

**Emeryville final draft:** earthquake High/High; climate change High/High; drought High/High; sea-level rise High/High; extreme weather High/High; hazardous materials Low/High; infrastructure/utility failure High/High; flood Low/Medium-High; structural fire Low/Medium; wildfire Low/Medium; urban fire Low/Medium-High; biological threats Low/High; civil disturbance Low/Low; terrorism Medium/High. These are paired probability/severity labels, not a composite risk grade.

**Tri-City 2024 significance results** (other four dimensions remain independently available in the cited tables):

| Hazard | Fremont | Newark | Union City |
|---|---|---|---|
| Climate change | Medium | High | High |
| Coastal flooding / sea-level rise | Medium | Medium | Med-High |
| Dam failure | High | High | High |
| Drought / water shortage | Low | Medium | Medium |
| Earthquake | High | High | High |
| 1%/0.2% annual-chance flood | High | High | High |
| Localized/stormwater flood | Medium | Medium | Medium |
| Landslide | Medium | Low | Medium |
| Levee failure | Medium | Low | Low |
| Extreme cold/freeze | Medium | Low | Low |
| Extreme heat | Medium | Medium | Medium |
| Heavy rain/storms | Medium | Medium | Medium |
| High winds/tornadoes | Medium | Low | Low |
| Subsidence | Low | Low | Low |
| Tsunami | Medium | Medium | Low |
| Wildfire | Medium-High | Medium | Medium |

**Oakland 2026 draft PRI:** dam failure 1.6 Low; drought 2.8 Medium; earthquake 5.0 High; flood 2.9 Medium; landslide 2.6 Medium; sea-level rise 2.5 Medium; severe weather 3.5 High; tsunami/seiche 1.7 Low; wildfire 4.8 High. These are Oakland-wide and draft.

**Piedmont 2019 significance:** climate change Medium; dam failure Medium; drought/water shortage Medium; earthquake High; earthquake liquefaction Medium; 1%/0.2% flood Low; localized/stormwater flood Medium; landslide/mudslide/hillside erosion/debris flow Medium; levee failure Low; extreme heat Medium; heavy rain/storms Medium; high winds Low; wildfire High.

## 6. Local-area quantitative-table inventory

The repository candidate extractor found thousands of snippets, including false table candidates and duplicated tables of contents. Candidate counts are therefore a discovery index, not evidence. The raw audit trail remains in each `data/lhmp/extracted/<jurisdiction>/tables_candidates.json`; every promoted cell still requires source-page verification.

| Jurisdiction / plan | Usable quantitative table families found | Local geographic unit | Example source pages/tables | Public-display assessment |
|---|---|---|---|---|
| Alameda County | Land area and population in mapped dam, landslide, tsunami, wildfire and other zones; critical facilities; county versus Unincorporated Alameda County | County and Unincorporated Alameda County | Unincorporated base p19; hazard tables throughout pp4-7 to 4-29 | Potentially concise Level B context after verifying every value and PAM boundary. Do not imply community-level precision. |
| Alameda 2022 | Building/population exposure and scenario consequences for earthquake, flooding/SLR and tsunami; priority coastal-inundation locations | Citywide, hazard areas, named coastal locations | Table 4-1 p59; Appendices E-I | Superseded; re-audit 2025. No promotion. |
| Albany draft | City assets, facilities, events and hazard narratives; no robust table series keyed to stable smaller areas | Citywide | Hazard ranking p21 | Citywide rating only. |
| Berkeley 2024 | Hazard maps, scenario and city asset/vulnerability material; no verified complete table series keyed to local plan areas | Citywide and hazard zones | Hazard sections B.1-B.13 | Use structured citywide likelihood/severity first; withhold local statistics pending focused review. |
| Tri-Valley cities | Jurisdiction event losses, repetitive-loss properties, exposure/vulnerability maps and annex-specific facility/context tables | Citywide/hazard zone | Vol. 2 Tables 1-11 onward, 2-11 onward, 3-11 onward | Citywide supporting detail; no local-area inheritance. |
| Emeryville draft | Earthquake/liquefaction, sea-level rise/flood, utilities, hazardous materials, fire and critical assets | Citywide/hazard zones | Hazard chapters beginning p26 | Draft citywide context; no local-area row system. |
| Fremont/Newark/Union City | Extensive parcel/building value, population, critical-facility, lifeline and future-development exposure tables for SLR, dam failure, earthquake ground failure, flood, landslide, levee, tsunami and wildfire | Citywide hazard polygons and individual future-development sites | Fremont annex A.4; Newark B.4; Union City C.4. Examples: Union City Tables C-22-C-29 onward | Valuable planning evidence but not local-area ratings. A point inside the hazard polygon should use the underlying official GIS finding; city totals may be shown separately as jurisdiction context. |
| Hayward | Critical city assets in Hayward Hills; regional SLR projections; SLR/tide matrix; maps for faults, shaking, liquefaction, tsunami, fire, landslide, flood and SLR | Hazard areas; Hayward Hills/shoreline narrative | Tables 6-8, document pp50, 58, 65 | Not a stable area-by-hazard table system. Do not label Hayward Hills or shoreline with an overall LHMP rating. |
| Oakland adopted | Formal scenario ratings for nine areas | Nine sub-areas | Appendix E pp460-480 | Eligible only through the existing reviewed gate and exact polygon assignment. Preserve scenario and adopted edition. |
| Oakland draft | Baseline EPC/buildings/lifelines plus dam, earthquake, flood, landslide, SLR, tsunami and wildfire exposure/consequence tables | Nine sub-areas | Complete inventory in `docs/oakland-draft-lhmp-subarea-audit.md`; examples Tables 7-6, 9-6, 10-7, 11-4, 12-2, 14-2, 15-5 | Strong Level B context when source-page verified. Never turn counts or modeled loss into a rating. |
| Piedmont 2019 | Parcel/building value, population, critical facilities and exposure by hazard zone | Citywide hazard polygons | Chapter 4 tables; e.g. wildfire Tables 4-65 to 4-67 | Citywide/hazard-zone context, not a named local-area system. Current draft must be ingested. |
| San Leandro element | Narrative and map-based shoreline, dam, tsunami, flood, fire and seismic context | General Plan/hazard areas | Environmental Hazards Element | Not an LHMP evidence source for structured rating display. |

### Required cell schema for any future quantitative record

Every promoted metric must store: hazard, geographic unit ID/name/type, metric name, raw value, unit, denominator where applicable, scenario, time horizon, table/page, plan edition/status, underlying source named by the LHMP, extraction status, review status, permitted use, and limitations. A table row without scenario or denominator is incomplete. A modeled damage amount is `modeled_scenario_damage`, never `expected_loss` unless the plan explicitly reports an expected/annualized quantity.

## 7. Adopted-versus-draft comparison

| Jurisdiction | Adopted evidence | Newer evidence | Material difference | Handling |
|---|---|---|---|---|
| Alameda | 2022 CAHMP | Adopted 2025 update, absent from repo | Unknown until ingestion; official page says 2025 updates the 2022 plan | Mark 2022 superseded. Do not publish it as current or compare values from memory. |
| Albany | 2018 plan referenced, absent | 2023 draft in repo | Draft adds explicit probability + three-severity score/ranking | Keep draft withheld until adoption or agency status confirmation. Never call it current adopted evidence. |
| Emeryville | 2019 adopted plan referenced, absent | 2025-2030 final draft | New paired probability/severity findings and updated source data | Draft may be separately cited after review, but cannot silently replace adopted evidence. Obtain both final adoption and FEMA approval. |
| Oakland | Adopted 2021-2026, in repo | Draft 2026-2031, in repo | Adopted plan has nine-area formal scenario ratings. Draft uses the same nine named assessment areas for quantitative tables but its formal PRI is citywide only and adds QAF. Data years/scenarios changed. | Preserve two independent claims: `adopted_area_scenario_rating` and `draft_subarea_metric`/`draft_citywide_PRI`. Never average or overwrite across editions. |
| Piedmont | 2019 source in repo | July 2025 draft available officially, absent from repo | Update likely changes data, hazard taxonomy and mitigation priorities; not audited here | Current registry evidence is incomplete. Ingest draft; retain visible draft/adopted labels. |

Older adopted evidence remains useful only when it is still the legally adopted source, its geometry and table are traceable, and the UI labels its edition. It becomes historical context—not a current rating—once superseded by an adopted replacement.

## 8. Boundary-source inventory and assignment design

### Boundary sources

| Boundary | Authority | Digital status | CRS / edition | Product disposition |
|---|---|---|---|---|
| Alameda County and incorporated jurisdiction boundaries | County/city authoritative GIS should be the primary source | Not registered in the LHMP evidence contract today | Must store source CRS, normalized CRS, edition, retrieval date and checksum | Required countywide prerequisite |
| Unincorporated Alameda County `PAM` | County plan explicitly names this dataset | Referenced, not located/registered | Unknown in source PDF | Blocked until exact dataset and edition are obtained |
| Eden/Castro Valley/East County General Plan areas | Alameda County Planning | Official maps exist; digital geometry not audited | Unknown | Context only unless an LHMP table explicitly uses the unit |
| Oakland Plan Area CW | City of Oakland ArcGIS REST layer 9 | Official polygon source and repository snapshot exist | EPSG:2227 source; snapshot normalized to lon/lat; service `Citywide_202410` | Ready with existing fail-closed restrictions; seek agency confirmation |
| Census tracts | U.S. Census TIGER/Line | Official digital polygons | Edition must match plan source year | Use only when a verified plan value is actually tract-keyed; SVI/DAC maps alone do not create a hazard rating |
| FEMA/CGS/CAL FIRE/DWR hazard polygons | Respective authoritative agencies | Hazard-specific digital sources may exist | Dataset-specific | These power address GIS evidence, not local-plan area assignment unless the plan explicitly keys a metric to the same edition |
| Static neighborhood/planning maps in PDFs | Plan publisher | Static only unless a matching official service is identified | N/A | Never digitize or approximate for production |

### Assignment result contract

```json
{
  "jurisdiction_assignment": {
    "jurisdiction_id": "oakland",
    "status": "matched",
    "method": "point_in_polygon",
    "boundary_source_id": "...",
    "geometry_edition": "..."
  },
  "local_area_assignment": {
    "area_system_id": "oakland_plan_area_cw",
    "area_id": "downtown",
    "area_name": "Downtown",
    "status": "matched",
    "method": "point_in_polygon",
    "match_count": 1,
    "boundary_status": "official_digital_source",
    "source_crs": "EPSG:2227",
    "normalized_crs": "EPSG:4326",
    "geometry_edition": "Citywide_202410",
    "limitations": []
  }
}
```

Allowed assignment statuses: `matched`, `out_of_coverage`, `boundary_ambiguous`, `multiple_matches`, `geometry_unavailable`, `source_unverified`, and `not_applicable`. Confidence is not a free-form percentage; it is derived from source/assignment status and never converts an invalid match into a valid one.

## 9. Reviewed hazard taxonomy map

Taxonomy is an explicit, versioned mapping table. Each plan term receives a relationship to a canonical display concept; keyword matching is prohibited for production.

| Plan term | Canonical relation | Rule |
|---|---|---|
| Earthquake / seismic hazards | Exact or parent relation to `earthquake` | Preserve plan scope; do not imply all earthquake sub-hazards are mapped at the address. |
| Ground shaking | Child of `earthquake` | May support an earthquake detail, not a separate overall rating unless the plan rates it separately. |
| Surface fault rupture / Alquist-Priolo zone | Child/indicator of `earthquake` | Address finding requires validated official GIS intersection; proximity is not zone membership. |
| Liquefaction | Child/indicator of `earthquake` and exact mapped hazard concept | May appear under earthquake. Do not double-count it as a second overall risk score. |
| Rainfall-induced landslide / general landslide | Exact/child of `landslide` | Never map automatically to earthquake-induced landslide. |
| Earthquake-induced landslide | Related child of both `earthquake` and `landslide` | Retain exact label and dataset; do not merge with general landslide susceptibility. |
| Landslide/mudslide/hillside erosion/debris flow | Plan-defined grouped concept | Preserve grouped wording; it is broader than any single component. |
| Tsunami/seiche | Plan-defined grouping of related but non-equivalent hazards | Store components and grouped source label. Address tsunami polygons do not establish seiche exposure. |
| Coastal flooding | Related to, but not equivalent to, `flood` and `sea_level_rise` | Preserve scenario driver and time horizon. |
| Sea-level rise | Distinct slow-onset hazard/context | Do not collapse into FEMA present-day flood-zone membership. |
| Flood: 1%/0.2% annual chance | Exact scenario relation to `flood` | Preserve annual-chance terminology and map edition. |
| Localized/stormwater flooding | Child of `flood`, not equivalent to FEMA SFHA | Requires its own source; FEMA non-match does not mean no stormwater concern. |
| Wildfire / wildland fire / WUI fire | Exact or scoped relation to `wildfire` | Preserve WUI/citywide scope. |
| Urban/structural fire | Related but non-equivalent to `wildfire` | Must not be merged into wildfire rating. |
| Fire Hazard Severity Zone | Hazard-map severity class | Not an LHMP overall risk grade. Store the exact class and source edition. |
| Dam failure / dam breach inundation | Exact relation to `dam_failure` | Inundation membership does not state failure probability. |
| Severe weather | Plan-defined parent | Preserve child hazards (heat, cold, wind, rain) where separately rated. |
| Drought / water shortage | Often grouped exact relation | Preserve whether the source addresses meteorological drought, supply shortage, or both. |
| Poor/hazardous air quality | Distinct hazard/context | Wildfire smoke can be a cause but is not equivalent to wildfire exposure. |
| Infrastructure/utility failure / PSPS | Distinct cascading or human-caused hazard | Do not treat as physical wildfire membership. |

Each mapping record should include `plan_term`, `canonical_hazard_id`, `relationship` (`exact`, `parent`, `child`, `related_non_equivalent`, `must_not_merge`), plan edition, reviewer, review status, and notes.

## 10. Countywide canonical design

This is a proposed future contract. It does not replace the current production contract in this task.

```json
{
  "jurisdiction_context": {
    "jurisdiction_id": "pleasanton",
    "hazard_id": "wildfire",
    "source_label": "Wildfire",
    "rating": "High",
    "rating_type": "tri_valley_hazard_risk_category",
    "scope": "jurisdiction",
    "methodology_id": "tri_valley_2024_weighted_risk_rating",
    "components": {
      "risk_score": 36
    },
    "plan_edition": "2024",
    "plan_status": "adopted_source",
    "claim_review": "source_page_verified",
    "source_claim": {
      "document_id": "tri_valley_2024_volume_2",
      "table": "3-12",
      "pdf_page": 95,
      "url": "..."
    },
    "limitations": ["Applies citywide; not a property or mapped-address finding."]
  },
  "local_area_context": null,
  "address_evidence": {},
  "vulnerability": {
    "status": "not_assessed"
  }
}
```

The contract must support absent branches. It must not create generic Oakland defaults. Recommended core entities:

- `PlanEdition`: title, jurisdiction(s), edition, publication/adoption/FEMA dates, source status, supersession relationship, URL/path/hash.
- `Methodology`: plan-specific dimensions, definitions, weights, formula, thresholds and source pages.
- `GeographicSystem`: authority, geometry source, CRS, edition, coverage, overlap/precedence rules and LHMP relationship.
- `AreaAssignment`: exact spatial result and fail-closed status.
- `RatingClaim`: exact source label/value, rating type, geographic scope and methodology ID.
- `MetricClaim`: value/unit/denominator/scenario/time horizon without conversion to a rating.
- `AddressEvidence`: independent official GIS result with its own positive/no-match/unavailable/not-applicable states.
- `ReviewRecord`: extraction, internal, expert, agency, source-status and eligibility dimensions.

### Evidence precedence is display ordering, not scoring

1. positive address-level official GIS finding;
2. matched official local-area formal rating;
3. matched official local-area quantitative/scenario context;
4. jurisdiction-wide rating;
5. countywide/regional preparedness context;
6. reviewed local-plan evidence unavailable.

One layer never overwrites another. A FEMA no-match can coexist with a citywide flood priority. A citywide rating can never manufacture a mapped address finding.

## 11. User-facing wording rules

| Capability / evidence | Approved pattern | Prohibited implication |
|---|---|---|
| Level A adopted | “Adopted Oakland 2021-2026 plan-area wildfire scenario rating for East Oakland Hills: High.” | “Your house has high wildfire risk.” |
| Level A draft | “Draft-plan area rating …” only if a draft actually provides one | Omitting `draft` or presenting it as adopted |
| Level B | “Draft Oakland Downtown context: under the modeled Hayward M7.05 scenario, the plan estimates displacement and short-term shelter needs across the entire Downtown sub-area.” | Treating the scenario as a forecast for the entered address |
| Level C composite | “Pleasanton-wide 2024 LHMP wildfire risk category: High.” | “High at your address” or “mapped finding” |
| Level C dimensions | “Berkeley-wide earthquake assessment: Likely, with Catastrophic potential impact.” | Translating it into another plan's composite High |
| Level D | “The local plan discusses earthquake as a planning concern but does not provide a structured rating suitable for display.” | Inventing a Low/Medium/High label |
| Level E | “Reviewed current local-plan rating unavailable.” | Falling back silently to a superseded or candidate value |
| Address GIS no-match | “Official dataset checked; the address did not intersect the mapped area.” | “Low risk” or overriding regional concern |
| GIS unavailable | “The official map check was unavailable.” | Treating failure as no match |
| Not applicable | “This dataset does not apply to this location.” | Calling the hazard low |

Always show scope, plan edition, plan status, methodology label, source link/page, and a short limitation. Avoid “official mapped finding” for any LHMP jurisdiction or sub-area rating.

## 12. Jurisdiction and hazard coverage matrices

Legend: **J** = jurisdiction-wide formal rating/dimensions; **L** = formal local-area rating; **S** = local-area or jurisdiction-subarea statistic; **C** = general structured/narrative context; **—** = no usable structured evidence identified. Asterisks are draft or superseded sources. These are source capabilities, not public eligibility.

### Major StayReady hazards

| Jurisdiction | Earthquake | Flood | Wildfire | Landslide | Tsunami | Dam failure | SLR / coastal | Severe weather / heat | Utility / other |
|---|---|---|---|---|---|---|---|---|---|
| Alameda County | S | S | S | S | S | S | S | C | C |
| Alameda 2022* | J | J | J | C | J | J | J | J | C |
| Albany draft* | J | J | J | J | — | — | C | J | J |
| Berkeley | J | J | J | J | J | — | J | J | J |
| Dublin | J | J | J | J | — | J | — | J | — |
| Emeryville draft* | J | J | J (urban/WF split) | C | C | C | J | J | J |
| Fremont | J | J | J | J | J | J | J | J | C |
| Hayward | C | C | C | C | C | — | C | C | C |
| Livermore | J | J | J | J | — | J | — | J | — |
| Newark | J | J | J | J | J | J | J | J | C |
| Oakland adopted/draft | L + J* + S* | L + J* + S* | L + J* + S* | L + J* + S* | L + J* + S* | J* + S* | J* + S* | J* | C |
| Piedmont 2019 | J | J | J | J | — | J | — | J | — |
| Pleasanton | J | J | J | J | — | J | — | J | — |
| San Leandro element | C | C | C | C | C | C | C | C | C |
| Union City | J | J | J | J | J | J | J | J | C |

### Coverage interpretation

- `S` in the county row means statistics for the explicitly modeled Unincorporated Alameda County unit, not community-level ratings for Ashland, Castro Valley, Sunol, or another named place.
- Tri-City `J` represents the exact multi-column city assessment, not a universal composite.
- Oakland `L` comes only from the adopted 2021-2026 scenario tables. Oakland draft statistics are `S*`; draft Table 17-1 is `J*`.
- Alameda's row is useful only as a superseded-method comparison until the 2025 adopted source is ingested.

## 13. Implementation batching plan

### Batch 0 — countywide evidence and geometry foundations

Build the edition/methodology/review/geometry entities and fail-closed assignment statuses before adding new claims. Register authoritative county/city boundaries. This is prerequisite infrastructure, not a UI redesign.

### Batch 1 — finish and stabilize Oakland

- Keep adopted area scenario ratings and draft sub-area metrics separate.
- Update status if/when the 2026-2031 plan is adopted; do not automatically promote the draft PDF.
- Obtain Oakland data-owner confirmation for Plan Area CW geometry.
- Re-run source-page comparison against the final adopted edition.

### Batch 2 — adopted jurisdiction-wide ratings

Prioritize Berkeley, Livermore, Dublin, Pleasanton, Fremont, Newark, and Union City. They cover large populations, use current plans, and require no speculative local-area assignment. Review the exact method tables and selected hazard rows; show scope as citywide.

### Batch 3 — draft/status-resolution jurisdictions

Albany and Emeryville can be reviewed structurally but remain draft-labeled/withheld according to product policy. Resolve adoption with the agencies. Ingest Piedmont's 2025 draft and compare it with 2019. Ingest Alameda's adopted 2025 plan and retire 2022 current claims.

### Batch 4 — Unincorporated Alameda County

Acquire the official PAM unincorporated geometry and determine whether the plan offers only one unincorporated unit or defensible smaller units. Review county tables cell by cell. General Plan areas and named communities remain contextual unless plan metrics are explicitly keyed to them.

### Batch 5 — incomplete/general-context sources

Hayward requires a fresh plan-status check and focused extraction design. San Leandro requires the completed standalone LHMP. Special-district service areas may be considered only as an additional service/infrastructure layer after jurisdiction evidence works.

## 14. Exact blockers

1. **No general reviewed repository:** `data/lhmp/reviewed/` is empty apart from placeholders.
2. **Alameda stale source:** adopted 2025 update is missing; registered 2022 source is superseded.
3. **Piedmont stale source:** official 2025 draft is missing; the 2019 repository PDF's resolution appendix is unexecuted.
4. **San Leandro wrong document type:** Environmental Hazards Element is not the requested standalone LHMP.
5. **Adoption metadata gaps:** exact local adoption resolutions/dates are not attached for Dublin, Pleasanton, Fremont, Newark, Union City, Albany, or Emeryville.
6. **County boundary gap:** the Alameda County plan's PAM unincorporated dataset is named but not registered with URL, CRS, edition, checksum, or polygons.
7. **No other plan-area geometry/rating pair:** neighborhood, fire-zone, tract, and hazard maps in other plans are not keyed to formal local-area ratings.
8. **PDF-only boundaries:** static maps cannot be approximated for point assignment.
9. **Edition coupling:** Tri-Valley annex scores require Volume 1 methodology; Tri-City annex assessments require the base plan definitions. A claim cannot cite only an annex cell and omit the method source.
10. **Taxonomy review:** repository aliases are discovery aids, not an approved equivalence map.
11. **Scenario semantics:** many tables report exposed assets or modeled consequences, not risk, probability, expected annual loss, or property outcome.
12. **Geometry/source mismatch risk:** an official GIS layer can be current yet differ from the edition used by an older LHMP. Both editions must be traceable.
13. **Oakland draft transition:** the draft is moving through adoption in 2026; status and final table values may change.
14. **Agency confirmation:** no jurisdiction other than existing source documents has confirmed StayReady's proposed interpretation or public wording.

## 15. Recommended next production batch

The next production batch should be **adopted jurisdiction-wide ratings for Berkeley and the Tri-Valley cities**, after claim-level source-page verification.

Recommended order:

1. Berkeley: verify the twelve likelihood/severity pairs from the adopted 2024 plan and record the 2025 adoption/FEMA metadata. Display the two dimensions without translating them.
2. Livermore: verify Table 2-12 plus Volume 1 pp262-265 methodology; adoption evidence is already strong and exact.
3. Dublin and Pleasanton: verify Tables 1-12 and 3-12 plus individual adoption resolutions.
4. Fremont, Newark and Union City: verify the complete five-dimension tables and base-plan definitions. Display selected dimensions or an exact plan “Significance” label, never a translated universal rating.

This batch has the best combination of population/geographic coverage, current source status, low geometry risk, and implementation effort. It deliberately does not create local-area claims. Alameda, Piedmont, Albany, Emeryville, unincorporated communities, Hayward, and San Leandro should remain subsequent batches until their exact blockers are resolved.

## Official status sources consulted

- Alameda County final plan: <https://lhmp.acgov.org/documents/FinalHMP_AlamedaCo_Mar2022.pdf>
- Alameda County General Plan archive/context: <https://www.acgov.org/cda/planning/generalplans/>
- Alameda current CAHMP page: <https://www.alamedaca.gov/Departments/Planning-Building-and-Transportation/Sustainability-and-Resilience/Climate-Adaptation-and-Hazard-Mitigation-Plan>
- Berkeley LHMP/adoption page: <https://berkeleyca.gov/Mitigation/>
- Livermore/Tri-Valley official plan page: <https://www.livermoreca.gov/departments/community-development/planning/hazard-mitigation>
- Fremont Tri-City official plan/news sources: <https://www.fremont.gov/Home/Components/News/News/813/> and <https://www.my.fremont.gov/19448/widgets/86300/documents/62424>
- Hayward Local Resilience Plan: <https://www.hayward-ca.gov/your-government/documents/local-resilience-plan>
- Oakland LHMP status page: <https://www.oaklandca.gov/Public-Safety-Streets/Fire/Ready-Oakland/Stay-Informed/Local-Hazard-Mitigation-Plan-LHMP>
- Piedmont update page: <https://piedmont.ca.gov/services___departments/planning___building/general_plan___other_policy_documents/hazard_mitigation_plan_>
- San Leandro standalone-LHMP procurement: <https://www.sanleandro.org/bids.aspx?bidID=63>

Official pages establish current status. All rating and table findings in this report were checked against repository PDFs; unless explicitly identified as Oakland source-page verified, they remain audit findings requiring the stated production review gate.
