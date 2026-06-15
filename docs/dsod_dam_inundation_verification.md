# DWR/DSOD Dam Inundation GIS Verification

Verified on 2026-06-15.

- Official ArcGIS item: `5354d98898194a4ab7b96eb6c85eecae`
- Owner: `Marill.Jacobson@water.ca.gov_DWR`
- Feature layer: `Approved_InundationBoundaries_As_of_Oct01_2025` (layer `100`)
- Endpoint: `https://services.arcgis.com/aa38u6OgfNoCkTJ6/arcgis/rest/services/Approved_InundationBoundaries_As_of_Oct01_2025/FeatureServer/100`
- DWR landing page: `https://water.ca.gov/programs/all-programs/division-of-safety-of-dams/inundation-maps`
- DSOD viewer: `https://fmds.water.ca.gov/maps/damim/`

## Schema

- Dam/scenario: `DamName`, `FailedStr`, `Scenario`, `LoadingScn`
- Hazard classification: `HazardCl`
- Publication date: `PubDate`
- Identifiers: `NID`, `StateID`
- County/jurisdiction: no county or jurisdiction field is published
- Geometry: `esriGeometryPolygon`
- Native projection: California Albers, `WKID 3310`

## Service Limits

- Statewide record count observed: 1,202
- Maximum record count: 2,000
- Query pagination: supported
- Query formats: JSON, GeoJSON, and PBF at the layer level
- Service export formats include GeoJSON, shapefile, GeoPackage, file geodatabase, KML, and CSV

Point-intersection requests use `inSR=4326`, `esriGeometryPoint`, and
`esriSpatialRelIntersects`, so address checks do not require a statewide download.

## Alameda County Filter

Because the layer has no county field, coverage was checked spatially against
`static/countbound.geojson`. The Alameda County polygon intersected 45 published
dam/structure/scenario polygons on 2026-06-15. Some dams are outside the county;
their hypothetical inundation boundaries reach Alameda County.

## Interpretation Limits

This is official mapped planning data. It represents hypothetical dam or critical
appurtenant structure failure scenarios and does not predict failure probability.
Boundaries are approximate, may omit federal dams or redacted information, and may
be updated. StayReady must not use LHMP or EOP screenshots for address-level
conclusions. Actual evacuation zones, timing, routes, and instructions come from
local emergency officials.
