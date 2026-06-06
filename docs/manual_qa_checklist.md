# StayReady Manual QA Checklist

Use public landmarks, approximate streets, or clearly fake Alameda County examples. Do not save or publish private residential addresses.

For every scenario, confirm:
- The homepage address flow works and keeps ZIP labeled as fallback.
- `/risk_summary` shows a resident guidance plan, not only technical evidence.
- Top hazards show why they matter, before/during/after actions, recovery prep, sources, and limitations.
- Address-level checks do not claim parcel-level precision.
- A non-match says the layer did not match; it must not say the address is safe.

## Scenarios

1. Berkeley hills scenario
- Expected top hazards: wildfire, earthquake, flood or other local plan concern depending on layer results.
- Expected guidance: Berkeley hills/fire-zone plan context may appear if city is detected; address/fire/fault/flood checks remain clearly labeled.
- Should not claim: official Berkeley parcel fire-zone membership unless an official polygon was checked.

2. Berkeley flats scenario
- Expected top hazards: earthquake and flood context should remain important; wildfire may include smoke/evacuation readiness rather than hills-specific claims.
- Expected guidance: west/flatland or general Berkeley context only where the current rules support it.
- Should not claim: no earthquake or no flood risk from one non-match.

3. Alameda shoreline scenario
- Expected top hazards: flood/shoreline context, earthquake, smoke/wildfire fallback where appropriate.
- Expected guidance: local facts may fall back unless reviewed structured facts exist.
- Should not claim: completed tsunami, sea-level-rise, or groundwater address-zone checks.

4. Hayward near-fault scenario
- Expected top hazards: earthquake should be prominent if mapped fault proximity check matches or jurisdiction context supports it.
- Expected guidance: fault check limits must mention liquefaction/building risk not checked.
- Should not claim: building safety or retrofit status.

5. Fremont residential scenario
- Expected top hazards: earthquake, flood, wildfire/fire context from Tri-City planning where reviewed/draft-reviewed source data exists.
- Expected guidance: if local structured facts are missing, county fallback language should be visible.
- Should not claim: city-specific neighborhood risk without reviewed facts.

6. Oakland hills scenario
- Expected top hazards: wildfire and earthquake should be easy to understand; flood may depend on address layer result.
- Expected guidance: evacuation, smoke, documents, medications, pets, and recovery continuity should appear.
- Should not claim: complete evacuation route analysis.

7. Oakland flatland/shoreline scenario
- Expected top hazards: flood and earthquake context should remain clear; wildfire may focus on smoke/evacuation impacts rather than direct fire zone membership.
- Expected guidance: address-level FEMA flood check status should be visible.
- Should not claim: no flood risk if FEMA layer does not match.

8. Alameda County address with minimal household info
- Expected top hazards: plan still renders without optional fields.
- Expected guidance: general household actions and recovery prep should appear.
- Should not crash: missing optional household factors.

9. Alameda County address with medical needs
- Expected household guidance: prescription copies, refill planning, doctor/pharmacy contacts, medical-device backup power/cooling, and go-bag medication reminders.

10. Alameda County address with pets and no car
- Expected household guidance: pet food/water/carrier/leash/vaccination records, pet-friendly evacuation, transit/ride/neighbor planning, and leave-early language.

11. Renter scenario
- Expected household guidance: renter insurance, landlord contact, lease backup, temporary housing planning, and photos of belongings.

12. Homeowner scenario
- Expected household guidance: insurance review, property photos/video, utility shutoff knowledge, and repair documentation planning.

## Regression Checks

- `/` loads and address input is the main action.
- `/risk_summary` loads after a successful form submission.
- `/api/health`, `/api/sources`, `/api/hazards`, and `/api/top-risks` still return JSON.
- With `SUPABASE_ENABLED=false`, local JSON fallback still works.
- No React, Next.js, Vite, Tailwind, or shadcn dependencies are added.
- Supabase keys are not committed.
