# Address-level hazard prioritization: research and recommended methodology

**Status:** Research recommendation only; no formula or UI implementation  
**Research cut-off:** 2026-06-22  
**Initial geography:** Alameda County, California  

## 1. Executive conclusion

StayReady cannot presently provide a scientifically defensible overall **High / Medium / Low risk rating for an individual address** for any supported hazard. The repository has useful official address evidence and jurisdiction context, but it does not have the building, site, household, probability, intensity, and consequence inputs—or a validated method combining them—needed to claim overall property risk.

The strongest honest launch is hazard-specific:

| Hazard | Final decision | Strongest honest public output now |
|---|---|---|
| Earthquake | **Requires additional data or methodology** | A multi-indicator panel: regional context, mapped surface-rupture zone, mapped ground-failure zones, fault proximity, and unknown building vulnerability. No single rating. |
| Flood | **Ready for official address-level category** | Exact effective FEMA flood-zone/SFHA attributes, after dataset lineage and map-version review. |
| Wildfire | **Ready for official address-level category** | Exact applicable CAL FIRE/locally adopted FHSZ category, after SRA/LRA edition and adoption review. Call it hazard severity, not risk. |
| Liquefaction | **Ready only for mapped evidence label** | Inside / not inside / unavailable for an official CGS Zone of Required Investigation. |
| Earthquake-induced landslide | **Ready only for mapped evidence label** | Inside / not inside / unavailable for an official CGS Zone of Required Investigation. |
| Tsunami | **Ready only for mapped evidence label** | Inside / not inside / unavailable for the CGS evacuation-and-response-planning hazard area. |
| Dam inundation | **Ready only for mapped evidence label** | Inside one or more named DSOD-approved hypothetical failure-scenario boundaries / no mapped intersection / unavailable. |

“Ready” above describes the agency methodology, not the current source-registration state. Every current production geospatial registration is still marked `provisional`; FEMA and CAL FIRE local snapshots lack confirmed lineage/effective dates, and the remote CGS/DWR integrations still require named human viewer validation. Until those gates pass, the application should continue to expose evidence cautiously and must not imply a completed official determination.

The product question—“How much attention should this address give this hazard?”—can be answered now only as **preparedness attention based on separate evidence**, not as risk. Recommended attention states remain: immediate address finding; major reviewed jurisdiction priority; preparedness context; checked with no mapped match; data unavailable; and not applicable. These states must never be collapsed into High/Medium/Low risk.

## Definitions and claim boundary

| Concept | Meaning | Examples | Not interchangeable with |
|---|---|---|---|
| Hazard probability | Chance of an event or intensity threshold in a stated period | 1% annual-chance flood; probability of specified ground shaking in 50 years | Zone membership, damage probability |
| Address exposure | Whether the geocoded point intersects a mapped zone or scenario | FEMA zone, CGS zone, DSOD scenario | Event probability, vulnerability |
| Severity / intensity | Physical conditions if the event occurs | Flood depth, shaking acceleration, fire behavior | Building loss |
| Vulnerability | Susceptibility of the building, site, or household | Foundation, first-floor height, roof/vents, retrofit, mobility | Hazard |
| Consequence | Resulting harm or loss | Injury, displacement, repair cost, evacuation difficulty | Exposure |
| Overall risk | A validated combination of probability and consequence, usually dependent on exposure, intensity, and vulnerability | Expected annual loss under a reviewed model | Any single map category |

## 2. Current evidence inventory

### Repository evidence path

The canonical summary in `risk_summary_view_model.py` correctly keeps regional context, address evidence, vulnerability, preparedness attention, actions, and claims separate. `hazard_engine.py` gathers GIS evidence; `hazard_priority.py` structures mapped findings and regional priority; `resident_guidance_engine.py` supplies actions. The methodology gap in `docs/address-risk-methodology-gap.md` correctly concludes that no address-level combining method exists.

Current address outcomes are appropriately distinct: `mapped_match`, `checked_no_match`, `data_unavailable`, `not_checked`, and `not_applicable`. Fault distance is proximity context, not polygon membership. Tests also enforce fail-closed behavior and keep a checked non-match distinct from unavailable data.

### Evidence by hazard

| Hazard | Address evidence in the repository | Jurisdiction evidence | Material gaps |
|---|---|---|---|
| Earthquake | Provisional local mapped-fault distance; provisional CGS Alquist-Priolo, liquefaction, and earthquake-landslide intersections | Oakland draft LHMP probability, impact, and priority; other LHMP files are extracted candidates but not promoted into reviewed records | Address ground-motion value; authoritative fault dataset lineage; site class; building age/type/retrofit; validated risk method |
| Flood | Provisional local FEMA NFHL polygon and attributes | Oakland draft LHMP context; unreviewed local-plan candidates | Exact snapshot lineage/effective community map; LOMC handling; elevation/depth; first-floor height; building vulnerability |
| Wildfire | Provisional local CAL FIRE FHSZ polygon and category | Oakland draft LHMP context; unreviewed local-plan candidates | SRA versus current adopted LRA provenance; adoption/version; structure ignition and access vulnerability |
| Liquefaction | Provisional official CGS remote intersection | Regional earthquake context only | Site investigation, groundwater/soil detail, foundation and building response, probability/consequence method |
| Earthquake-induced landslide | Provisional official CGS remote intersection | Oakland draft general landslide context is not the same phenomenon | Site slope/material/drainage/retaining conditions; building response; probability/consequence method |
| Tsunami | Provisional official CGS remote hazard-area intersection | Oakland draft tsunami/seiche context | Site intensity/depth/arrival time; evacuation and occupant vulnerability; a likelihood/loss method |
| Dam inundation | Provisional DSOD approved hypothetical failure-boundary intersections | General plan context may exist but is not normalized/reviewed | Complete scenario applicability; depth/velocity/arrival time in usable data; failure probability; evacuation/building vulnerability |

### Jurisdiction-material status

The repository contains adopted, draft, and archived LHMP PDFs plus machine-extracted candidates. The only normalized jurisdiction ranking records are chiefly from the **Draft Oakland 2026–2031 LHMP**, Table 17-1, page 316. They are explicitly draft, citywide planning categories. Berkeley and other plan extractions are unreviewed candidates. Neither category may be converted into an address finding or an address rating.

## 3. Hazard methodology matrix

| Hazard | Probability available? | Address exposure? | Severity/intensity? | Vulnerability? | Official address rating? | Recommended output | Readiness |
|---|---|---|---|---|---|---|---|
| Earthquake | Yes in official USGS models, but not integrated or validated for StayReady claims | Yes, partial indicators | Ground-shaking hazard values exist; not integrated | No | No overall rating | Multiple exact indicators; no H/M/L | More data/methodology |
| Flood | FEMA category embeds defined annual-chance classes for some zones | Yes | Zone/category; BFE/depth only where source fields support it | No | Official hazard-zone category, not risk | Preserve FEMA category and plain-language definition | Source review required; method ready |
| Wildfire | FHSZ model includes likelihood over 30–50 years, not an address event probability | Yes | Official Moderate/High/Very High hazard severity | No | Official hazard-severity category, not risk | Preserve applicable official category | Source/adoption review required; method ready |
| Liquefaction | Strong shaking is considered in zoning; no address event probability | Yes | Susceptibility/regulatory-zone inclusion, not site intensity | No | Official mapped investigation zone only | Mapped evidence label | Viewer validation required |
| Earthquake landslide | Strong shaking and slope/material factors are considered; no address event probability | Yes | Susceptibility/regulatory-zone inclusion | No | Official mapped investigation zone only | Mapped evidence label | Viewer validation required |
| Tsunami | Map includes a probabilistic model input but expressly does not report likelihood at a location/time | Yes | Maximum-considered planning extent, not site loss | No | Official planning area only | Mapped evidence label | Viewer validation required |
| Dam inundation | No failure probability | Yes, scenario-specific | Some approved maps include depth, velocity, arrival time; current boundary service only proves extent | No | Official hypothetical scenario extent only | Named scenario mapped evidence | Viewer/scenario validation required |

No hazard qualifies for Option C, hazard-specific H/M/L overall risk. Option A is appropriate for flood and wildfire and as an exact binary category for CGS/DWR findings. Option B is appropriate for every hazard as preparedness-attention ordering. Option D is narrowly appropriate only where the exact official measure and time horizon can be repeated without translation: FEMA annual-chance definitions and, after a separate integration and scientific review, a USGS ground-shaking exceedance measure. Option E remains necessary for any unsupported overall rating.

## Authoritative source and method register

“Update” below describes the source program where known; StayReady must still record the exact edition/effective date retrieved. “Usage” is a product research note, not legal advice.

| Agency / source | Exact method and category | Resolution / update | Probability | Severity | Vulnerability / direct rating | Address applicability and StayReady interpretation | Usage and limitations |
|---|---|---|---|---|---|---|---|
| FEMA, [NFHL and official FIRM products](https://www.fema.gov/flood-maps/products-tools) and [flood-zone definitions](https://emilms.fema.gov/is_0253b/assets/FB474F19-D5D4-A1D4-41BE-8D58E1302D1A.pdf) | Effective regulatory flood polygons. SFHA represents the 1%-annual-chance flood; shaded X/B is generally between 1% and 0.2%; unshaded X/C is outside those mapped extents. Zone details differ and must be preserved. | Polygon/point intersection; effective maps change through studies, map revisions, and LOMCs, not a fixed annual cadence | Defined for certain zone classes | Zone, floodway, BFE/depth attributes where present | No building vulnerability; no overall property rating | Yes for the geocoded point as exact map category. StayReady repeats—not re-scores—the category. | MSC is the official source. Map category is not a parcel survey, elevation certificate, loss estimate, or proof of no flooding. Repository snapshot lineage and reuse/attribution review are incomplete. |
| FEMA, [Flood Risk Analysis and Mapping technical references](https://www.fema.gov/media-collection/technical-references-flood-risk-analysis-and-mapping) and [Flood Depth Grids](https://www.fema.gov/sites/default/files/documents/fema_flood-depth-grids.pdf) | Non-regulatory risk products may include event-specific depth grids produced from water-surface elevation minus ground elevation. | Raster/product-dependent; not available uniformly | Event frequency tied to product | Depth, where produced | Still no building vulnerability unless coupled to a separate model; no direct address risk rating | Potential future intensity input, only after local availability and datum/quality validation | Do not infer a depth where a grid is absent or substitute BFE for depth at the structure. |
| CAL FIRE/OSFM, [Fire Hazard Severity Zones](https://osfm.fire.ca.gov/osfm/what-we-do/community-wildfire-preparedness-and-mitigation/fire-hazard-severity-zones) | Science-based model scores fire likelihood and behavior using fire history, fuels/vegetation, predicted flame length, embers, terrain, and typical fire weather; outputs Moderate, High, Very High. Hazard is evaluated over 30–50 years without mitigation. | Area polygons. SRA maps effective 2024-04-01; 2025 LRA recommendations were released in four phases and require local-jurisdiction applicability/adoption review | Incorporated in model, not reported as address annual probability | Direct official hazard severity | Explicitly excludes home hardening, fuel reduction, and other mitigation; not risk | Yes when the correct SRA designation or locally applicable LRA map intersects the point. Repeat exact FHSZ category. | Official page says hazard, not risk. Urban/unmapped is “not in a mapped FHSZ in the checked layer,” never Low. Current local file lacks edition, SRA/LRA, and adoption lineage. |
| USGS, [2023 National Seismic Hazard Model](https://www.usgs.gov/programs/earthquake-hazards/science/2023-50-state-long-term-national-seismic-hazard-model) and [100-year damaging-shaking dataset](https://data.usgs.gov/datacatalog/data/USGS%3A64ff8ca8d34ed30c2057b506) | Probabilistic seismic hazard combines fault/source occurrence and ground-motion models; provides ground-motion exceedance values/probabilities. A published product reports chance of MMI VI or greater in 100 years. | National grids/site calculations; 2023 model supersedes prior national editions, without a promised annual update | Yes, for defined shaking threshold/horizon | Expected ground-motion intensity/exceedance | No building/site vulnerability and no direct property-risk rating | Potentially address-locatable after coordinate, model edition, site-condition, parameter, and API/data validation. Repeat the measure and horizon; do not translate to H/M/L. | USGS distinguishes hazard (shaking) from risk (consequence). Model uncertainty, site conditions, and user comprehension require seismologist/engineer review. USGS-authored content/data are generally public domain unless marked otherwise; verify item metadata and attribution. |
| CGS, [Alquist-Priolo Earthquake Fault Zones](https://conservation.ca.gov/cgs/alquist-priolo) | Regulatory zones surround mapped active-fault surface traces; the Act addresses surface rupture and required investigation/development constraints. “Active” for the Act means rupture within about 11,000 years. | Official quadrangle/digital polygons; issued or revised as CGS mapping changes | No address rupture probability | Surface-rupture regulatory exposure | No building vulnerability; no overall earthquake rating | Exact point intersection only. Distance to a fault is a separate proximity fact. | Does not address shaking, liquefaction, landslide, or overall earthquake risk. Official-map status/effective date and publication-license attribution must be retained. |
| CGS, [Seismic Hazard Zones](https://www.conservation.ca.gov/cgs/shma) | Liquefaction zones combine young loose sediments, historic shallow groundwater, subsurface data, and expected strong shaking in 50 years. Earthquake-landslide zones combine expected shaking, existing landslides, slope, and material strength. Both are Zones of Required Investigation. | Official quadrangles about 60 square miles at 1 inch = 2,000 feet; preliminary maps have review before official release; no fixed cadence | Incorporated, not a point probability output | Regional susceptibility sufficient to require investigation | No site/building vulnerability and no direct risk rating | Exact official-zone intersection only; separate indicator labels under the earthquake family | Not every place in a zone will fail; non-match is not proof of absence. Point/boundary/geocoding uncertainty and official/preliminary status matter. Repository service terms require stated CGS attribution. |
| CGS/Cal OES, [California Tsunami Hazard Area Maps](https://conservation.ca.gov/cgs/tsunami/maps) | Planning area uses 2009 inundation maps plus high-resolution 975-year-return-period probabilistic results, extensions for local sources, and practical street/geographic evacuation boundaries. | Coastal planning polygons; statewide coastal-county updates completed in 2022; update when agency republishes | Source page expressly says the map does not provide likelihood at a place over a time period | Maximum-considered planning extent, not a site intensity value | No building/evacuation vulnerability; no risk rating | Point intersection supports only “inside official tsunami hazard area used for evacuation planning.” | Planning only; not a legal/disclosure map or live evacuation order. Copyright/attribution and agency disclaimer apply. Boundaries are partly operationally adjusted, so do not reverse-engineer depth/probability. |
| California DWR/DSOD, [Inundation Maps](https://water.ca.gov/Programs/All-Programs/Division-of-Safety-of-Dams/Inundation-Maps), [FAQ](https://water.ca.gov/Programs/All-Programs/Division-of-Safety-of-Dams/FAQs), and [approved viewer](https://fmds.water.ca.gov/maps/damim/) | Dam-owner engineering models depict flooding from hypothetical failure scenarios; required outputs include boundary, arrival time, maximum depth, and maximum velocity. DSOD approves maps for emergency action planning. | Scenario maps/geospatial boundaries; minimum 10-year update and sooner after material dam/downstream change | No dam-failure probability | Scenario intensity can exist in map products, but current integrated service supplies boundary evidence | No building vulnerability; downstream hazard classification concerns consequences if a dam fails, not dam condition/probability | Point intersection must retain each dam/scenario identity. Never merge it into likelihood. | General emergency-planning information; local officials determine actual evacuation. Coverage may omit non-jurisdictional/federal scenarios; DWR disclaims warranties/liability. |
| Adopted local/county LHMPs | Jurisdiction-specific planning methods rank hazards for mitigation allocation using plan-defined probability, impact, spatial extent, warning, duration, or qualitative adjustments | City/county/plan areas; typically five-year FEMA plan cycle, subject to adoption/update | Sometimes qualitative at jurisdiction scale | Sometimes impact/consequence at jurisdiction scale | May include community vulnerability; not an individual-property rating | Jurisdiction context only, after exact table/method/adoption human review | Methods differ between plans and editions. Draft Oakland records must remain visibly draft. Extracted candidates are not evidence until reviewed. |

## 4. Detailed hazard analyses and launch specifications

### Earthquake

**What is known.** At the address, StayReady has mapped-fault proximity and potential official CGS intersections for Alquist-Priolo surface rupture, liquefaction, and earthquake-induced landslide. At jurisdiction scale, Oakland has draft LHMP planning categories. It lacks an integrated, reviewed probabilistic ground-shaking value and all meaningful building vulnerability inputs.

**Method assessment.** USGS can supply a location-relevant shaking hazard measure for a stated threshold and time horizon. That is a scientifically useful hazard indicator, not a prediction of the next earthquake and not building damage. Fault distance cannot substitute for the NSHM because shaking reflects many sources, magnitude, distance, and geology. An Alquist-Priolo match addresses surface rupture only. CGS ground-failure zones are susceptibility/regulatory indicators, not uniform failure predictions. Building age, structural system, code era, soft-story characteristics, foundation, occupancy, and retrofit status are essential to damage/consequence; citywide impact is not a proxy.

**Recommended launch method.**

- **Inputs:** reviewed jurisdiction context; exact AP/liquefaction/earthquake-landslide intersections; separately labeled fault distance; vulnerability state `unknown`. Later, add one reviewed USGS ground-shaking metric with model version, parameter, site assumption, probability, and horizon.
- **Decision rule:** never combine the indicators. Show each positive, negative, unavailable, or not-applicable state independently. Do not infer AP membership from fault distance. Do not infer building damage from ground shaking.
- **Output:** “Earthquake — multiple indicators,” not High/Medium/Low. Example: “Inside a CGS mapped liquefaction Zone of Required Investigation”; “About X from the nearest loaded mapped fault”; “Building vulnerability not assessed.”
- **Scope:** address-point map evidence and jurisdiction planning context, not parcel, geotechnical, engineering, loss, insurance, or disclosure determination.
- **Failure:** one failed source makes only that indicator unavailable. Other indicators may render, but no aggregate reassuring outcome is permitted.
- **Review:** California engineering geologist/seismologist, structural engineer, emergency manager, product counsel, and accessibility/content review before adding a USGS value.
- **Tests:** model/version and horizon displayed; AP/ground-failure matches remain separate; proximity never becomes membership; positive/negative/unavailable mixtures remain distinct; jurisdiction label never becomes address label; no damage wording without building data.
- **Readiness:** mapped indicators after source validation; overall rating postponed.

**Final decision: Requires additional data or methodology.**

### Flood

**What is known.** StayReady can intersect an address point with a local FEMA NFHL snapshot and preserve zone attributes. Oakland has draft jurisdiction planning context. The local snapshot’s exact download, effective date, community coverage, and official-viewer comparison are not documented.

**Method assessment.** FEMA zone designations are already meaningful official categories. SFHA and shaded-X definitions carry specific annual-chance meanings, but they are hazard categories—not the annual probability of water entering the building. Zone A and AE share the 1%-annual-chance flood basis, while AE generally has detailed BFEs and A may not. Floodways, V/VE, AO/AH, AR/A99, D, and X categories have distinct meanings; a custom three-bin translation would discard material information. Building risk needs elevation relative to flood level, first-floor height, foundation, floodproofing, contents, drainage and a validated depth-damage method.

**Recommended launch method.**

- **Inputs:** current effective NFHL/FIRM polygon and exact attributes (`FLD_ZONE`, `ZONE_SUBTY`, `SFHA_TF`, floodway, static BFE/depth/unit where valid), map effective date/community, coordinate precision, and source status.
- **Decision rule:** report the most specific official category exactly. Describe annual chance only when FEMA defines it for that category. Never calculate building depth from a BFE without authoritative ground/first-floor elevations. Preserve multiple applicable features.
- **Output:** e.g., “FEMA Zone AE — Special Flood Hazard Area (1%-annual-chance flood area).” For checked no-match: “No intersection with a mapped flood-hazard polygon in the effective layer checked,” never “Low flood risk.”
- **Scope:** effective mapped flood-hazard designation at the point, not a parcel/insurance/elevation certificate or loss forecast.
- **Failure:** unavailable/out-of-coverage/version-unknown. Do not fall back to ZIP or citywide rating.
- **Review:** floodplain manager validates zone dictionary, community/map version, LOMC behavior, BFE/depth wording, and coastline/levee nuances; counsel reviews NFIP/insurance claims.
- **Tests:** every supported FEMA code; SFHA consistency; AE versus A wording; shaded/unshaded X; floodway; boundary points; overlapping features; LOMC/version fixture; missing attributes; service failure; non-match/unavailable separation.
- **Readiness:** method ready after snapshot lineage and effective-map validation.

**Final decision: Ready for official address-level category.**

### Wildfire

**What is known.** StayReady can preserve a CAL FIRE FHSZ category from a local snapshot. Oakland has draft jurisdiction context. The repository does not establish whether each polygon is the effective 2024 SRA map, a 2025 LRA recommendation, or an adopted local LRA designation.

**Method assessment.** CAL FIRE’s official Moderate/High/Very High labels may be shown because they are the source categories. They represent modeled **fire hazard severity**, combining likelihood and expected fire behavior over 30–50 years without mitigation. CAL FIRE expressly distinguishes this from risk. LRA maps are recommendations whose local adoption/applicability must be verified; SRA designations follow a state regulatory process. Unzoned urban land is not “Low.” Building risk requires ignition-resistant construction, roof/vents/windows, defensible space, parcel vegetation, access, suppression, and other vulnerability/consequence information.

**Recommended launch method.**

- **Inputs:** correct current SRA designation or locally applicable adopted LRA layer; FHSZ category; responsibility area; edition/effective/adoption date; source feature ID.
- **Decision rule:** select only the legally/applicably current layer for the coordinate and retain the agency category verbatim. Never translate unzoned or unchecked into Low.
- **Output:** “CAL FIRE High Fire Hazard Severity Zone (hazard, not property risk).” No-match wording must name the checked layer.
- **Scope:** area-level modeled wildfire hazard severity at the address point; does not assess structure, smoke, evacuation, utilities, insurance, or expected loss.
- **Failure:** unknown map edition, unresolved SRA/LRA applicability, no coverage, or request failure = unavailable.
- **Review:** OSFM/local fire authority confirms adoption and layer precedence; wildfire scientist validates wording; counsel reviews code/insurance implications.
- **Tests:** all three categories; SRA/LRA precedence; adopted versus recommended LRA; urban unzoned; boundary/overlap; outdated edition; non-match; unavailable; category never relabeled “risk.”
- **Readiness:** method ready after edition/adoption/source-lineage validation.

**Final decision: Ready for official address-level category.**

### Liquefaction

**What is known.** A provisional CGS service can establish point intersection with an official mapped liquefaction Zone of Required Investigation. Jurisdiction earthquake context exists only where reviewed and must stay separate.

**Method assessment.** The CGS zone combines regional geology, groundwater, geotechnical information, and expected strong shaking. Inclusion means a site investigation may be required; it does not mean liquefaction will occur everywhere or predict damage. A meaningful property-risk rating requires site investigation and building/foundation vulnerability. A custom H/M/L would be invented.

**Recommended launch method.** Exact `inside official CGS mapped liquefaction Zone of Required Investigation`, `no intersection in the official layer checked`, `unavailable`, or `not mapped/not applicable`. Retain quadrangle/status/effective date. Failure is non-reassuring. Review with a California engineering geologist and validate known points against EQ Zapp. Test boundary, official versus preliminary, positive, negative, missing coverage, and service failure.

Treat liquefaction as a **distinct earthquake-consequence indicator**, because its map, action, and uncertainty differ from shaking and rupture. It may have a dedicated detail route for comprehension, but it should be grouped under earthquake in cross-hazard prioritization to avoid double-counting the triggering event.

**Final decision: Ready only for mapped evidence label.**

### Earthquake-induced landslide

**What is known.** A provisional CGS service can establish intersection with an official earthquake-induced-landslide Zone of Required Investigation. The Oakland draft “landslide” priority is broader and must not be attached automatically.

**Method assessment.** CGS combines expected shaking, existing landslides, slope, and material strength. The map is neither a general rainfall-landslide forecast nor a uniform failure prediction. Site slope, drainage, retaining systems, material properties, foundation, and building location are necessary for property consequence.

**Recommended launch method.** Use the same four-state mapped-evidence contract as liquefaction, with exact earthquake-induced wording and official-map metadata. Never merge general landslide LHMP context without a documented hazard taxonomy mapping. Review with an engineering geologist/geotechnical engineer. Test positive/negative/unavailable, mapped coverage, boundary tolerance, official status, and separation from non-seismic landslide.

Treat this as a **distinct earthquake-consequence indicator**, grouped under earthquake for ordering but independently visible and actionable.

**Final decision: Ready only for mapped evidence label.**

### Tsunami

**What is known.** A provisional CGS remote service can establish whether the point is in a tsunami hazard area. Oakland has draft tsunami/seiche planning context.

**Method assessment.** The official boundary is an evacuation-and-response-planning product built from a 975-year-return-period model plus other sources and practical boundary adjustments. CGS expressly says it does not report the likelihood of a tsunami affecting a location over a particular time. Therefore, neither the 975-year model input nor “outside” may be translated into annual probability or Low risk. Building and household consequence depends on depth/velocity, building, vertical evacuation, mobility, warning, route, and time.

**Recommended launch method.**

- **Inputs:** current official county hazard-area polygon, edition/citation, point intersection, source status.
- **Decision rule:** exact inside/no-intersection/unavailable only; no probabilistic reverse engineering.
- **Output:** “Inside California’s mapped tsunami hazard area used for evacuation and response planning.”
- **Scope:** planning area, not disclosure map, site intensity, risk rating, or live order.
- **Failure:** unavailable; do not infer inland safety from a failed check.
- **Review/tests:** CGS/Cal OES or local tsunami planner reviews wording and actions; validate known shoreline/inland controls, boundaries, version, failures, and ensure live emergency instructions supersede the product.

**Final decision: Ready only for mapped evidence label.**

### Dam inundation

**What is known.** The provisional DSOD boundary service contains 1,202 statewide records and repository notes report 45 scenarios intersecting the Alameda County window. It can return one or more named hypothetical failure-boundary matches.

**Method assessment.** A match means modeled inundation under a stated hypothetical failure scenario. It says nothing about the chance of failure. A dam’s downstream hazard-potential classification describes consequences if failure occurs and is explicitly unrelated to condition. Even where map PDFs include depth, velocity, and arrival time, the current boundary integration does not establish those values at the address.

**Recommended launch method.**

- **Inputs:** every intersecting approved scenario’s dam ID/name, map date, scenario/loading/appurtenant structure if available, boundary, and service version.
- **Decision rule:** list scenarios separately; never choose the “worst” by an invented ordering and never convert dam hazard class/condition to address probability.
- **Output:** “Inside a DSOD-approved hypothetical dam-failure inundation boundary: [named scenario].”
- **Scope:** emergency-planning scenario extent only. Actual evacuation comes from local officials.
- **Failure:** unavailable, incomplete coverage, or scenario identity missing. A no-intersection applies only to published boundaries checked.
- **Review/tests:** DSOD/local emergency manager validates scenario completeness and wording; counsel reviews public reliance. Test multiple overlaps, scenario identity, federal/non-DSOD coverage limitation, version/date, boundaries, missing attributes, pagination, unavailable, and no-match.

**Final decision: Ready only for mapped evidence label.**

## 5. Earthquake methodology recommendation

Earthquake should use **several indicators rather than one label**:

1. **Regional shaking context:** reviewed LHMP context and, later, an exact USGS probability/intensity measure with time horizon.
2. **Surface rupture:** official Alquist-Priolo intersection.
3. **Ground failure—liquefaction:** official CGS intersection.
4. **Ground failure—earthquake-induced landslide:** official CGS intersection.
5. **Fault proximity:** distance to a documented mapped fault dataset, visibly labeled proximity only.
6. **Building vulnerability:** unknown until the user supplies validated attributes or an authoritative building record is lawfully integrated.

Do not compute an overall value from these indicators. They are correlated, incomplete, and measured on different scales. A future risk model would need a scientifically reviewed chain from earthquake occurrence → site ground motion → ground failure → building response → household consequence, with uncertainty propagation and validation. USGS hazard values can improve the first two steps but do not complete the chain.

Liquefaction and earthquake-induced landslide should remain separate **indicators/detail topics** because they have different official layers, mechanisms, mitigations, and failure states. For navigation and cross-hazard ordering they should be children of the earthquake family, not independent competitors that make one earthquake appear three times more important.

## 6. Cross-hazard prioritization recommendation

### Evaluation

| Approach | Decision | Reason |
|---|---|---|
| Universal numeric score | Reject | Would combine incomparable regulatory zones, severity classes, proximity, scenarios, and citywide planning judgments; no calibration or common consequence unit exists. |
| Universal High/Medium/Low | Reject for risk | Same labels would conceal different meanings and falsely imply comparability. |
| Evidence-strength ordering | Use only as a tie-break/quality signal | Authoritativeness is not urgency; unavailable evidence must not make a hazard seem less important. |
| Address-finding-first | Recommend as primary ordering | A positive official address finding creates the clearest location-specific action signal without claiming common risk magnitude. |
| Jurisdiction-priority ordering | Use within the no-address-finding group | Helpful only when adopted/reviewed and visibly regional; never overrides a positive address finding or becomes an address rating. |
| Separate categories without cross-hazard rank | Recommend as the conceptual model | Most honest; the UI may still need a stable display order within categories. |

### Recommended deterministic order

Group first, then use stable non-scientific tie-breakers:

1. **Positive official address findings**—FEMA/FHSZ categories, CGS zones, tsunami area, DSOD scenarios. Group earthquake consequences beneath earthquake.
2. **Address proximity or other bounded context**—for example, fault distance, clearly not membership.
3. **Major adopted/reviewed jurisdiction priorities** with no positive address finding.
4. **General preparedness context.**
5. **Checked with no mapped match.** This is not Low and remains actionable where regional significance exists.
6. **Not applicable**, when applicability is established by authoritative rules.

**Data unavailable is not a bottom/low-risk rank.** Keep it in a persistent “information unavailable” group near the relevant hazard family, with visible retry/source status. Within a group, use a fixed product taxonomy (life-safety/action deadline first if officially defined, then stable hazard order/alphabetical), not a computed score. Never use evidence confidence to estimate hazard magnitude.

## 7. Minimum additional data and validation work

### Required before the limited launch

1. **FEMA:** replace or document the local snapshot with exact download/service URL, effective date, community coverage, schema, CRS, checksum, retrieval time, map revision/LOMC policy, official-viewer control points, and reuse/attribution review.
2. **CAL FIRE:** identify every feature’s SRA/LRA source, current edition, effective/adoption status, local-jurisdiction precedence, exact download/service URL, metadata, checksum, and official-viewer control points. Do not mix 2024 SRA and 2025 recommended/adopted LRA silently.
3. **CGS:** record ArcGIS item metadata, official/preliminary status and effective dates; named reviewer compares positive, negative, boundary, and out-of-coverage Alameda controls against EQ Zapp/official maps for AP, liquefaction, and landslide.
4. **Tsunami:** validate county edition and known shoreline/inland controls against the official viewer; preserve the official citation and planning-only disclaimer.
5. **DWR:** validate named overlapping scenarios, pagination, map dates, dam identifiers, boundary results, and omissions against the DSOD viewer; document whether federal/non-DSOD dams require a separate official source.
6. **Fault proximity:** replace unknown lineage with a versioned official fault dataset and document geometry, fault type/age, completeness, distance algorithm, and why any displayed threshold exists. Until then, display raw distance only or withhold.
7. **LHMPs:** human-review adopted plans and exact tables/methods before promotion. Keep draft Oakland visibly draft; do not promote extracted candidates automatically.
8. **Geocoding:** establish positional-accuracy, parcel-versus-structure-point, boundary-tolerance, ambiguous-address, multi-building, and coordinate-retention/privacy policies.

### Research needed before any future overall risk

- **Earthquake:** versioned USGS address-level shaking metric; site-condition assumptions; building inventory/consented attributes; reviewed structural vulnerability and consequence model.
- **Flood:** locally available FEMA depth grids or reviewed hydraulic products; reliable ground and first-floor elevations; building/contents vulnerability and validated depth-damage functions.
- **Wildfire:** validated parcel/building ignition vulnerability, defensible-space recency, access/egress, and a public scientific probability/consequence method. FHSZ alone is insufficient.
- **Ground failure:** site-specific geotechnical data and a reviewed building-response/consequence model; regulatory-zone intersection alone is insufficient.
- **Tsunami:** authoritative site intensity/arrival information plus building and evacuation vulnerability and a reviewed consequence method.
- **Dam:** authoritative scenario depth/velocity/arrival rasters, complete scenario coverage, and failure-probability/consequence methodology. Boundary membership alone is insufficient.

## 8. Scientific and legal review needs

Before public release, obtain documented review of:

- **Scientific claims:** California engineering geologist/geotechnical engineer for CGS interpretations; seismologist and structural engineer for USGS/shaking/building claims; certified floodplain/hydraulic specialist for FEMA categories and depth; wildfire scientist/local fire authority for FHSZ; tsunami and dam emergency planners for planning-area/action wording.
- **Regulatory status:** FEMA effective-map and LOMC implications; CAL FIRE SRA versus LRA adoption; official versus preliminary CGS maps; DSOD versus federal dam coverage; draft versus adopted LHMP status.
- **Legal/product claims:** no representation as real-estate disclosure, engineering opinion, insurance determination, code-compliance finding, evacuation order, or guarantee. Review agency disclaimers, attribution, copyright/license terms, map-service automated use, and update obligations.
- **Privacy:** consent and minimization for address, coordinates, building age/retrofit, disability/mobility, household members, and evacuation needs; retention/deletion and access controls; no public logging of precise addresses.
- **Accessibility and emergency use:** plain-language status distinctions, non-color cues, screen-reader semantics, multilingual review, and unambiguous direction to official live alerts during events.
- **Governance:** named source owner, review date, next-review trigger, change log, rollback plan, incident correction, and claim-level provenance for every public result.

The most sensitive claims requiring pre-publication counsel/expert sign-off are: annual-chance wording at a property; “official” or “effective” map status; FHSZ adoption; regulatory/disclosure implications of CGS zones; any implied safety from non-match; any depth, arrival, damage, loss, or insurance statement; and any cross-hazard comparative label.

## 9. Recommended next implementation batch

Implement only a **source-validation and exact-category hardening batch**, not a risk model:

1. Complete named human verification and metadata gates for FEMA, CAL FIRE, CGS, tsunami, and DWR sources.
2. Add controlled, source-derived category dictionaries for exact FEMA and FHSZ wording; preserve raw official values and edition/adoption metadata.
3. Keep CGS, tsunami, and dam outputs as mapped-evidence labels with named layers/scenarios and explicit scope.
4. Group liquefaction and earthquake-induced landslide as earthquake indicators while retaining independent evidence and detail content.
5. Apply the address-finding-first categorical order, with unavailable data visibly segregated—not demoted.
6. Add contract tests for category fidelity, source version, boundary behavior, overlap, non-match, unavailable, applicability, and regional/address separation.

This batch should not add High/Medium/Low risk, a numeric score, an LLM decision, building inference, or new public probability beyond verbatim official definitions. A separate research spike may prototype an exact USGS ground-shaking metric offline, but it should not ship until scientific and comprehension review passes.

## 10. Rejected approaches

1. **Weighted universal formula.** There is no defensible way to average FEMA zone, FHSZ, fault distance, CGS regulatory zones, DSOD scenarios, and LHMP categories.
2. **Relabel every official category High/Medium/Low.** The apparent consistency would erase material scientific and regulatory meanings.
3. **Treat an address intersection as overall risk.** It proves bounded mapped exposure only.
4. **Treat no intersection as Low or safe.** Map scope, unmapped mechanisms, indirect effects, resolution, and changing conditions remain.
5. **Treat unavailable as no match.** This creates false reassurance and suppresses the need to retry.
6. **Convert citywide/LHMP probability, impact, or priority into address risk.** Scope is incompatible; Oakland’s current normalized evidence is also draft.
7. **Use fault distance as earthquake likelihood or shaking.** Nearby faults are not the only sources, and site geology/ground-motion models matter.
8. **Infer tsunami annual probability from the 975-year model input.** CGS expressly says the public planning map provides no place/time likelihood and adjusts boundaries for evacuation practicality.
9. **Infer dam failure probability from inundation extent, downstream hazard class, or condition.** These describe scenarios, consequences, or condition—not event likelihood.
10. **Infer building vulnerability from ZIP, neighborhood, imagery, age alone, or an LLM.** Material attributes require consent, provenance, validation, and an expert-reviewed model.
11. **Adopt proprietary black-box risk scores.** They would frustrate claim-level provenance, reproducibility, public-method review, and long-term source governance.
12. **Double-count earthquake consequences as independent top-level hazards in a universal rank.** Keep the findings separate but group them under the triggering hazard.

## Decision record

| Hazard | Options selected | Final decision |
|---|---|---|
| Earthquake | A for exact indicators; B for attention; D only as a future exact USGS metric; E for overall rating | **Requires additional data or methodology** |
| Flood | A; B; narrowly D only where the FEMA category itself defines annual chance; E for overall property risk | **Ready for official address-level category** |
| Wildfire | A; B; E for overall risk | **Ready for official address-level category** |
| Liquefaction | A as binary mapped-zone evidence; B; E for H/M/L | **Ready only for mapped evidence label** |
| Earthquake-induced landslide | A as binary mapped-zone evidence; B; E for H/M/L | **Ready only for mapped evidence label** |
| Tsunami | A as planning-area evidence; B; E for probability/H/M/L | **Ready only for mapped evidence label** |
| Dam inundation | A as named scenario evidence; B; E for probability/H/M/L | **Ready only for mapped evidence label** |

## Repository materials reviewed

- `docs/address-risk-methodology-gap.md`
- `risk_summary_view_model.py`
- `hazard_engine.py`
- `hazard_priority.py`
- `resident_guidance_engine.py`
- `data/geospatial/datasets.json`
- `data/hazard_priority/official_gis_layer_rules.json`
- `data/hazard_priority/jurisdiction_hazard_rankings.json`
- `data/hazard_priority/source_documents.json`
- `data/lhmp/plan_registry.json`, extraction coverage, reviewed/unreviewed directories, and local LHMP source files
- GIS, hazard-priority, safety-contract, CGS, tsunami, dam, source-page, and canonical-summary tests and fixtures

No production code, formula, canonical contract, or user interface was changed by this research task.
