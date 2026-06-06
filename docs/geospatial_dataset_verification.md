# Geospatial Dataset Verification

## Trust Boundary

Only a named human reviewer may change a geospatial dataset version to
`verified`.

Codex may collect metadata, calculate checksums, inspect schemas, count
records, run automated tests, compare StayReady output with documented human
observations, and recommend a status. Codex must not approve a dataset as
verified or fill missing provenance with assumptions.

Existing FEMA, CAL FIRE, USGS, and CGS snapshots remain `provisional` until
the process below is completed.

## Dataset Statuses

- `verified`: A named human reviewer approved this exact dataset version.
- `provisional`: The source appears official, but lineage, edition, coverage,
  licensing, or viewer comparisons are incomplete.
- `invalid`: The file, service, schema, checksum, CRS, or required metadata
  failed validation.
- `data_unavailable`: The source cannot currently be evaluated.
- `retired`: This version was intentionally removed from active use.

Status changes apply to one exact dataset version. A new checksum, service
schema, effective date, or material source change requires a new review.

## Required Dataset Checklist

Create one completed checklist per dataset version:

- Dataset ID and version:
- Official agency:
- Authoritative landing-page URL:
- Exact service or download URL:
- Dataset name:
- Permitted claim:
- Prohibited claims:
- License, terms, and attribution notes:
- Retrieval date or service-access date:
- Effective or publication date:
- Local snapshot, remote service, or live feed:
- SHA-256 checksum for a local snapshot:
- Record count for a local snapshot:
- CRS stated by official metadata:
- CRS after StayReady conversion:
- Expected Alameda County coverage:
- Official map viewer URL:
- Human test locations:
- Reviewer name:
- Review date:
- Discrepancies:
- Known limitations:
- Final status: `verified`, `provisional`, `data_unavailable`, or `retired`

Unknown fields must say `not verified` or remain empty. They must not be
inferred from file names, nearby web pages, or similar datasets.

## Human Comparison Procedure

1. Record the official viewer result before looking at StayReady output.
2. Use three to five locations representing expected matches, non-matches,
   boundaries, and different Alameda County geographies.
3. Record the viewer layer name, legend category, visible date/version, and
   whether the viewer reports the area as evaluated.
4. Run the same location through StayReady.
5. Compare match status, category, source version, and wording.
6. Record every discrepancy. Do not explain away a discrepancy without
   evidence from the agency.
7. Keep the dataset `provisional` if lineage, coverage, edition, licensing,
   or any material discrepancy remains unresolved.
8. Mark it `data_unavailable` if StayReady cannot safely evaluate it.
9. Mark it `retired` if the agency superseded or withdrew it.
10. A second reviewer is recommended before public production use.

## Human Test Case Worksheet

For each location record:

- Test case ID:
- Address or named location:
- Reason selected:
- Official viewer URL:
- Official viewer result:
- Official viewer category:
- Official viewer date/version:
- Was the area evaluated:
- StayReady result:
- StayReady public claim status:
- Match:
- Discrepancy:
- Pass or fail:
- Reviewer initials and date:

## Public Wording

### FEMA Match

The saved address point matched a flood-zone polygon in StayReady's
provisional local FEMA snapshot. This is not a parcel, insurance, elevation,
or building-specific determination. Confirm the result in FEMA's official
viewer.

### FEMA Non-Match

The saved address point did not match a flood-zone polygon in StayReady's
provisional local FEMA snapshot. This does not mean the property is safe from
flooding or outside every flood-related area. Confirm current information with
FEMA and local agencies.

### CAL FIRE Match

The saved address point matched a Fire Hazard Severity Zone polygon in
StayReady's provisional local CAL FIRE snapshot. This describes long-term
physical fire hazard, not the probability, timing, or expected damage from a
specific fire.

### CAL FIRE Non-Match

The saved address point did not match a qualifying polygon in StayReady's
provisional local CAL FIRE snapshot. Wildfire smoke, embers, evacuation,
road closures, and power disruption may still affect the location.

### CGS Zone Match

The saved address point matched a California Geological Survey regulatory or
planning zone in the checked dataset. The zone supports planning and
disclosure decisions; it does not predict that damage will occur at this
address.

### CGS Non-Match

The saved address point did not match the checked CGS zone. This does not
mean there is no earthquake hazard or that the building is safe.

### CGS Area Not Evaluated

The checked dataset does not document coverage for this address, so StayReady
did not evaluate zone membership.

### Fault Proximity

The saved address point is approximately the reported distance from a loaded
mapped fault trace. This is proximity context, not Alquist-Priolo zone
membership and not a prediction of shaking or damage.

### Unavailable Data

The official layer was missing, invalid, outside documented coverage, or
otherwise unavailable. StayReady did not complete this check. Missing data
must not lower the displayed exposure or imply safety.

## Approval Record

When a human approves a dataset:

1. Preserve the completed checklist with the dataset version.
2. Enter the reviewer name and review date.
3. Confirm the checksum or remote service identity has not changed.
4. Change only that version's status to `verified`.
5. Run the full geospatial and safety test suites.
6. Record the approval in version control.

Automated scripts may report that technical checks passed. They may never
write `verified` status.
