# StayReady full product reaudit (2026)

Date: 2026-07-21

> Historical audit baseline. The county-authoritative location contract,
> shared production state, fail-closed dataset registry, map controls, and
> privacy changes described in `customer-ready-implementation.md` supersede
> implementation details in this document. Findings are retained for traceability.

## Executive assessment

StayReady has a stronger safety foundation than its previous interface suggested. The application already isolates address and ZIP flows, keeps resident inputs in short-lived server-side state behind an opaque token, runs GIS sources independently, blocks candidate LHMP extraction from the reviewed repository, and contains explicit prohibitions against unsupported address-risk ratings. The baseline suite passed 312 tests.

The central product experience nevertheless had a serious mismatch with the evidence architecture. The live Risk Summary exposed a six-column methodology table, not a resident answer. On a phone every hazard became six stacked boxes, city/sub-area/address/vulnerability/action concepts repeated for many hazards, and exact official results were frequently one disclosure away. The homepage competed with the address journey through previews, audience marketing, and signup content. The map did not inherit which layers mattered to the saved summary.

The highest-impact implementation work in this reaudit therefore changes the canonical evidence presentation and its supporting normalization. It does not create an overall property-risk score.

## Current address-summary accuracy

### Strengths

- `app.py::process_form` gives address and ZIP mutually exclusive paths and `get_zip_risk_row_for_context` returns legacy ZIP data only for an explicit ZIP search.
- `app.py::save_resident_state` stores resident inputs in a short-lived filesystem cache and leaves only a random token in the signed cookie.
- `hazard_engine.py::_collect_location_checks` runs FEMA, CAL FIRE, fault, CGS, tsunami, and dam checks independently and uses a shorter failure cache.
- `hazard_engine.py::check_flood_layer` and `check_wildfire_layer` include polygon boundary touches and preserve returned source categories.
- `risk_summary_view_model.py` already separated regional context, Oakland sub-area context, address evidence, unknown vulnerability, and actions.
- `data/lhmp/README.md` and `lhmp_repository.py` correctly state that machine-extracted candidates are not publishable reviewed facts.

### P0 correctness findings

1. **P0 — fault proximity could be overwritten by CGS checks.** `hazard_engine.py::_apply_cgs_evidence` replaced `additional_geospatial_evidence`. A valid fault-distance calculation and later CGS polygon checks could not coexist as first-class records. The implementation now appends CGS evidence and emits a separate fault-proximity record with miles, near/not-near state, address-point precision, provenance, and limitations.

2. **P0 — exact FEMA/CAL FIRE terminology was not reliably promoted into the canonical summary.** The canonical view model looked for `result_label` while local FEMA and CAL FIRE normalization stored exact categories inside `details.matched_layers`. Synthetic tests passed because their fixtures already used the desired shape. The new evidence normalizer reads the production shape and preserves returned names such as `FEMA flood zone X` and `CAL FIRE Fire Hazard Severity Zone: Very High`.

3. **P0 — one aggregate outcome hid mixed source states.** A hazard can have a valid mapped match, a proximity result, several checked non-matches, and an unavailable source at once. The old model selected one outcome and displayed the rest inconsistently. Each hazard now retains normalized `records`, `mapped_matches`, `proximity`, `nonmatches`, and `unavailable` collections. Valid findings remain visible when another check fails.

4. **P0 — proximity and intersection needed stronger type separation.** Fault distance now uses `claim_type=proximity`, `matched=None`, and `Near address` scope. It cannot become polygon membership through a truthy `matched` flag.

5. **P0 — unsupported overall labels remain in compatibility fields.** `hazard_engine.py::merge_structured_result` still creates `risk_level` and numeric `priority_score` from `exposure_level` for legacy consumers. The redesigned Risk Summary does not render them, but future refactoring should remove or strictly namespace these fields to prevent reuse.

6. **P0 — research coverage labels can be mistaken for review.** `data/lhmp/coverage_matrix.json` describes extraction candidates, not approved facts. The production reviewed directory is effectively empty (`.gitkeep` only), while other hand-curated files provide current display evidence. No extracted candidate was promoted by this work.

## Address-to-evidence pipeline

1. `templates/_home_reference.html` posts either `address` or `zip_code` to `app.py::process_form`.
2. A street address is validated, geocoded with Nominatim, constrained to service bounds, and assigned a ZIP from local boundaries where possible.
3. `location_service.py::location_from_session` derives city/county and identifies address versus ZIP mode.
4. `app.py::get_all_hazards` calls `hazard_engine.py::build_hazard_results`.
5. `hazard_engine.py::_collect_location_checks` runs local and remote sources in parallel and caches by rounded point and ZIP, never by full address.
6. Hazard-specific builders attach map evidence, reviewed guidance, limitations, sources, and actions. `_classify_preparedness_priority` keeps mapped exposure, regional importance, and action priority separate.
7. `resident_guidance_engine.py::build_resident_plan` builds action groups and calls `risk_summary_view_model.py::build_canonical_risk_summary`.
8. The canonical model now emits per-source evidence records, scoped LHMP context, a controlled interpretation, transparent attention category, result-quality summary, unknowns, and actions.
9. `templates/_risk_summary_reference.html` renders the canonical contract as cards and disclosures. It does not calculate facts or ratings.

## LHMP coverage and limitations

- The registry contains 15 county/city plan inputs and the extraction coverage matrix contains broad candidate coverage. Candidate breadth is not human review.
- `local_plans.json` contains 15 jurisdiction entries, all labeled `draft_reviewed`, but most entries identify hazards in scope rather than page-cited probability/impact claims.
- `data/hazard_priority/jurisdiction_hazard_rankings.json` has five real public records, all for Oakland and all from the Draft 2026–2031 LHMP. A sixth record is explicitly a test jurisdiction.
- `data/hazard_priority/sub_area_evidence.json` has 63 reviewed records: nine Oakland plan areas across earthquake, flood, wildfire, landslide, tsunami, dam failure, and sea-level rise.
- Oakland plan-area assignment uses an official polygon snapshot and withholds an unmatched Gateway Industrial District feature rather than guessing.
- `lhmp_location_facts.json` contains 25 manually structured area facts across jurisdictions (23 `draft_reviewed`, 2 `reviewed`), but these facts are not a substitute for exact citywide probability/impact tables.
- `data/lhmp/reviewed/` has no production fact files. `lhmp_repository.py` therefore returns no reviewed extracted facts. This is fail-closed behavior, not evidence that all 15 plans have equivalent reviewed coverage.
- Non-Oakland summaries must not render an Oakland sub-area field. The new card uses an Oakland planning-area block only when an eligible matched record exists; otherwise it adapts to exact citywide context or an explicit unavailability statement.

## Information architecture and UI findings

### P1

- **Six-column core table:** too dense on desktop and structurally repetitive on mobile. Replaced with evidence-first hazard cards.
- **Location quality omitted:** the selected address was visible, but city, ZIP, precision, and failed-check quality were not a coherent first section. Added a result-quality panel without coordinates.
- **Attention language was methodological:** labels such as `Address finding` and `Sub-area context` did not answer practical concern. Replaced with documented StayReady categories: Direct mapped concern; Near a mapped feature; Prepare despite no mapped match; Important local hazard; General awareness; Information unavailable.
- **Actions repeated:** the before/during/recovery matrix repeated AC Alert under multiple hazards. The primary page now uses the existing action-ID deduplication and shows why each top action appears.
- **Map was a separate product:** summary results did not make relevant layers obvious. The summary now deep-links by hazard, saves only non-sensitive relevant layer keys, and the map opens those layers centered on the saved point.
- **Homepage competed with the task:** sample risk/map cards and audience marketing were replaced by a four-part explanation of the result. ZIP remains secondary and household details remain collapsed.
- **Navigation overpromoted empty/redundant routes:** the navigation now follows Check address → My summary (only with a saved result) → Map → Prepare → Sources. Hazard detail remains reachable from the summary.

### P2/P3

- `app.py` remains oversized and mixes route, compatibility, geocoding, presentation, and map API logic.
- `templates/map.html` and `templates/home.html` still contain large inactive `{% if false %}` blocks and large page-specific scripts/styles. The dead Risk Summary version was removed; home/map cleanup remains.
- The project has no root README, so setup, architecture, test installation, and source-review governance require archaeology across `SECURITY.md` and `docs/`.
- Multiple visual systems remain in `templates/base.html`, inline page CSS, and `static/css/stayready.css`.
- The update signup remains visually large relative to the core task even after the main journey was simplified.

## Mobile and accessibility

### Findings

- The old table converted each row into six sequential mobile cells; this met a narrow responsive test but failed rapid comprehension.
- The map layer panel contains clickable `div` elements enhanced with keyboard roles in JavaScript. Native buttons should replace them in a later pass.
- No skip link existed. One is now present and the main landmark is focusable.
- Navigation now sets server-rendered `aria-current` and uses resident-result state to avoid a prominent empty Summary link.
- Evidence status always includes text; color is supplemental.
- The new card grid becomes a single reading column below 991px. At 575px headers, actions, evidence records, unknowns, and map CTA stack explicitly. At 430px the home headline and result grid receive a dedicated narrow-phone rule.

## Performance and reliability

- GIS requests are already parallelized and cached. Failure responses use a shorter TTL.
- Address summary generation still performs all supported remote CGS and dam checks on a cold request; source-specific timing telemetry that excludes coordinates would help prioritize further work.
- The homepage loading sequence previously named individual data sources on a timer, which could imply progress unrelated to actual completion. It now uses truthful general states: preparing checks, waiting for responses, keeping unavailable data separate, and organizing reviewed context.
- `map` avoids repeating summary checks and consumes short-lived relevant-layer keys saved by the completed summary.

## Privacy and security

### Strengths

- Exact resident inputs are in server-side temporary state, not the signed cookie.
- Sensitive surfaces use `ph-no-capture`; PostHog masks all input and text, sanitizes URLs, rejects unsafe property names, and restricts custom properties.
- External map/geocode targets are allowlisted elsewhere in the repository; rate limits and production cookie settings are documented in `SECURITY.md`.

### Remaining risks

- Nominatim and official GIS services receive location data as required to perform checks; the privacy page must continue to state this plainly.
- PostHog session replay and autocapture are still enabled globally. Current masking is strong, but disabling replay for the entire result/map routes would reduce dependence on selector configuration.
- Process-local rate limiting and filesystem-backed resident state do not coordinate across multiple production workers.
- A strict CSP is deferred because of inline scripts and third-party CDN assets.

## Implemented evidence model

Each canonical hazard now has:

- `address_evidence.records[]`: dataset, exact finding, evidence status, claim type, geographic scope, agency, URL, checked time, review status, limitations;
- independent `mapped_matches`, `proximity`, `nonmatches`, and `unavailable` collections;
- `regional_context`: only eligible, cited citywide categories at their real scope;
- `subarea_context`: reviewed Oakland planning-area facts only after official polygon assignment;
- `vulnerability`: explicitly unknown;
- `attention`: transparent StayReady category and basis, never an official rating;
- `interpretation`: controlled composition of the returned address result, eligible local context, failed checks, and unknown vulnerability;
- phase-grouped, source-backed actions.

Precedence is explicit: direct mapped match; nearby mapped feature; prepare despite non-match when reviewed local/major regional context exists; unavailable information; important local hazard; general awareness. Missing data never maps to a low-concern category.

## Prioritized implementation plan

### Completed in this pass

- P0 preserve fault proximity beside CGS polygon checks;
- P0 preserve exact official categories from production evidence shapes;
- P0 normalize mixed match/non-match/proximity/unavailable records;
- P0 keep ZIP and address evidence isolated;
- P1 replace the six-column summary with adaptive resident-centered cards;
- P1 add location/result quality, controlled interpretation, unknowns, and source metadata;
- P1 consolidate first actions and connect summary findings to map layers;
- P1 simplify homepage hierarchy and top navigation;
- P1 add explicit mobile layouts, skip navigation, and truthful loading copy;
- P2 remove the obsolete Risk Summary interface block.

### Next by impact

1. P0 human-review and publish claim-level city/LHMP records for jurisdictions beyond Oakland; do not promote extraction candidates automatically.
2. P0 remove or namespace legacy `risk_level`/numeric `priority_score` compatibility fields.
3. P1 add authoritative city/unincorporated jurisdiction polygons instead of relying primarily on parsed geocoder locality strings.
4. P1 add validated proximity-to-boundary calculations for FEMA/CAL FIRE/CGS polygon layers; retain them separately from intersection.
5. P1 replace map `div` toggles with native controls and give every summary evidence record a direct layer/source mapping.
6. P2 split `app.py` into location, summary, map API, feedback, and public-page blueprints/services.
7. P2 remove remaining dead home/map template branches and move inline CSS/JavaScript into versioned assets.
8. P2 add real responsive visual-regression checks at 320, 375, 430, 768, and desktop widths.
9. P2 add privacy-safe source timing and failure-rate telemetry.
10. P3 reduce footer/signup weight and finish consolidating radius, shadow, and typography tokens.
