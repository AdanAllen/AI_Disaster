#!/usr/bin/env python3
"""Build the Alameda County FEMA and CAL FIRE evidence snapshots.

The command downloads immutable/identifiable agency sources, clips ArcGIS
services to the Alameda County envelope, and writes deterministic GeoJSON.
It does not change dataset review status or human-approval fields.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from zipfile import ZipFile

import pyogrio
import requests
from requests.adapters import HTTPAdapter
from pyproj import Transformer
from shapely import from_wkb, make_valid
from shapely.geometry import mapping, shape
from shapely.ops import transform
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "StayReadyNow/2.0 evidence-import"
ALAMEDA_ENVELOPE = "-122.42,37.43,-121.42,37.92"
CRS84 = {
    "type": "name",
    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
}

FEMA_DOWNLOAD = "https://msc.fema.gov/portal/downloadProduct?productID=NFHL_06001C"
FEMA_EXPECTED_ARCHIVE = "06001C_20260525.zip"
FEMA_SERVICE = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"
)
FEMA_PANEL_SERVICE = (
    "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/3"
)
CALFIRE_SRA_SERVICE = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "FHSZSRA_23_3/FeatureServer/0"
)
CALFIRE_LRA_RECOMMENDATION_SERVICE = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "FHSALRA25_v1_All/FeatureServer/0"
)
CALFIRE_RESPONSIBILITY_SERVICE = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "State_Responsibility_Area/FeatureServer/0"
)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
        ),
    )
    return session


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_geojson(path: Path, features: list[dict]) -> dict:
    payload = {"type": "FeatureCollection", "crs": CRS84, "features": features}
    raw = _json_bytes(payload)
    path.write_bytes(raw)
    property_fields = sorted(
        {key for feature in features for key in (feature.get("properties") or {})}
    )
    geometry_types = sorted(
        {str((feature.get("geometry") or {}).get("type") or "") for feature in features}
    )
    schema = hashlib.sha256(
        json.dumps(
            {"geometry_types": geometry_types, "property_fields": property_fields},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(features),
        "schema_fingerprint": schema,
    }


def _download_fema(session: requests.Session, directory: Path) -> tuple[Path, str]:
    response = session.get(FEMA_DOWNLOAD, timeout=(5, 120), stream=True)
    response.raise_for_status()
    filename = Path(urlparse(response.url).path).name
    if filename != FEMA_EXPECTED_ARCHIVE:
        raise RuntimeError(
            f"FEMA county package changed from {FEMA_EXPECTED_ARCHIVE} to {filename}; "
            "review the new edition before importing."
        )
    archive = directory / filename
    digest = hashlib.sha256()
    with archive.open("wb") as output:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                digest.update(chunk)
                output.write(chunk)
    with ZipFile(archive) as bundle:
        bundle.extractall(directory / "fema")
    return directory / "fema", digest.hexdigest()


def _shapefile_features(path: Path) -> list[dict]:
    metadata, _, geometry_wkb, columns = pyogrio.raw.read(path)
    rows = [
        dict(zip(metadata["fields"].tolist(), values, strict=True))
        for values in zip(*columns, strict=True)
    ]
    transformer = Transformer.from_crs(metadata["crs"], "EPSG:4326", always_xy=True)
    features = []
    for geometry, properties in zip(from_wkb(geometry_wkb), rows, strict=True):
        if geometry is None or geometry.is_empty:
            continue
        geometry = transform(transformer.transform, geometry)
        clean_properties = {}
        for key, value in properties.items():
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif value is not None and not isinstance(value, (str, int, float, bool)):
                value = str(value)
            clean_properties[key] = value
        features.append(
            {"type": "Feature", "properties": clean_properties, "geometry": mapping(geometry)}
        )
    features.sort(
        key=lambda item: str(
            item["properties"].get("FLD_AR_ID")
            or item["properties"].get("FIRM_ID")
            or ""
        )
    )
    return features


def _arcgis_features(
    session: requests.Session,
    endpoint: str,
    *,
    authority: str | None = None,
    adoption_status: str | None = None,
    where: str = "1=1",
    use_alameda_envelope: bool = True,
) -> tuple[list[dict], dict]:
    metadata_response = session.get(endpoint, params={"f": "json"}, timeout=(5, 30))
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    if metadata.get("error") or not metadata.get("fields"):
        raise RuntimeError(f"Invalid service metadata: {endpoint}")
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "f": "geojson",
            "where": where,
            "outFields": "*",
            "outSR": "4326",
            "returnGeometry": "true",
            "resultOffset": offset,
            "resultRecordCount": 2000,
        }
        if use_alameda_envelope:
            params.update(
                {
                    "geometry": ALAMEDA_ENVELOPE,
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                }
            )
        response = session.get(
            f"{endpoint}/query",
            params=params,
            timeout=(10, 300),
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("features")
        if payload.get("error") or not isinstance(page, list):
            raise RuntimeError(
                f"Invalid ArcGIS query response: {endpoint}: "
                f"{payload.get('error') or sorted(payload)}"
            )
        for feature in page:
            props = feature.setdefault("properties", {})
            if authority:
                props["AUTHORITY"] = authority
            if adoption_status:
                props["ADOPTION_STATUS"] = adoption_status
            geometry = feature.get("geometry")
            if geometry:
                geometry_shape = shape(geometry)
                if not geometry_shape.is_valid:
                    geometry_shape = make_valid(geometry_shape)
                feature["geometry"] = mapping(geometry_shape)
        features.extend(page)
        if len(page) < 2000:
            break
        offset += len(page)
    object_id = metadata.get("objectIdField") or "OBJECTID"
    features.sort(key=lambda item: str((item.get("properties") or {}).get(object_id, "")))
    editing = metadata.get("editingInfo") or {}
    return features, {
        "service_version": str(metadata.get("currentVersion") or ""),
        "last_edit_epoch_ms": editing.get("lastEditDate"),
        "data_last_edit_epoch_ms": editing.get("dataLastEditDate"),
    }


def _arcgis_features_by_ids(
    session: requests.Session,
    endpoint: str,
    *,
    where: str,
) -> tuple[list[dict], dict]:
    """Query MapServer geometry without relying on unsupported pagination."""
    metadata_response = session.get(endpoint, params={"f": "json"}, timeout=(5, 30))
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    ids_response = session.get(
        f"{endpoint}/query",
        params={"f": "json", "where": where, "returnIdsOnly": "true"},
        timeout=(10, 60),
    )
    ids_response.raise_for_status()
    object_ids = sorted(ids_response.json().get("objectIds") or [])
    if not object_ids:
        raise RuntimeError(f"Official service returned no object IDs: {endpoint}")
    features = []
    object_id_field = metadata.get("objectIdField") or "OBJECTID"
    for start in range(0, len(object_ids), 500):
        chunk = ",".join(map(str, object_ids[start : start + 500]))
        response = session.get(
            f"{endpoint}/query",
            params={
                "f": "geojson",
                "where": f"{object_id_field} IN ({chunk})",
                "outFields": "*",
                "outSR": "4326",
                "returnGeometry": "true",
            },
            timeout=(10, 300),
        )
        response.raise_for_status()
        payload = response.json()
        page = payload.get("features")
        if payload.get("error") or not isinstance(page, list):
            raise RuntimeError(
                f"Invalid ArcGIS object-ID response: {endpoint}: "
                f"{payload.get('error') or sorted(payload)}"
            )
        for feature in page:
            geometry = feature.get("geometry")
            if geometry:
                geometry_shape = shape(geometry)
                if not geometry_shape.is_valid:
                    geometry_shape = make_valid(geometry_shape)
                feature["geometry"] = mapping(geometry_shape)
        features.extend(page)
    features.sort(
        key=lambda item: int(
            (item.get("properties") or {}).get(object_id_field, 0) or 0
        )
    )
    editing = metadata.get("editingInfo") or {}
    return features, {
        "service_version": str(metadata.get("currentVersion") or ""),
        "last_edit_epoch_ms": editing.get("lastEditDate"),
        "data_last_edit_epoch_ms": editing.get("dataLastEditDate"),
    }


def _prepared_geojson_pages(paths: list[Path]) -> list[dict]:
    features = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("features"), list):
            raise RuntimeError(f"Invalid saved FEMA service page: {path}")
        for feature in payload["features"]:
            geometry = feature.get("geometry")
            if geometry:
                geometry_shape = shape(geometry)
                if not geometry_shape.is_valid:
                    geometry_shape = make_valid(geometry_shape)
                feature["geometry"] = mapping(geometry_shape)
        features.extend(payload["features"])
    features.sort(
        key=lambda item: int(
            (item.get("properties") or {}).get("OBJECTID", 0) or 0
        )
    )
    if len(features) != 2451:
        raise RuntimeError(
            f"Saved FEMA service pages contain {len(features)} records, expected 2451."
        )
    return features


def build(
    *,
    fema_directory: Path | None = None,
    fema_service_pages: list[Path] | None = None,
) -> dict:
    session = _session()
    retrieved_at = datetime.now(UTC).date().isoformat()
    with tempfile.TemporaryDirectory(prefix="stayready-evidence-") as temp_name:
        temp = Path(temp_name)
        if fema_directory:
            fema_dir = fema_directory
            archive_candidates = [
                fema_directory.with_suffix(".zip"),
                fema_directory.parent / "fema.zip",
                fema_directory.parent / FEMA_EXPECTED_ARCHIVE,
            ]
            archive = next((path for path in archive_candidates if path.exists()), None)
            if archive is None:
                raise RuntimeError(
                    "--fema-directory requires the downloaded archive beside the "
                    "extracted directory so its checksum can be recorded."
                )
            fema_archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        else:
            fema_dir, fema_archive_sha = _download_fema(session, temp)
        panels = _write_geojson(
            ROOT / "static/FemaFirmPanels.geojson",
            _shapefile_features(fema_dir / "S_FIRM_PAN.shp"),
        )

    if fema_service_pages:
        flood_features = _prepared_geojson_pages(fema_service_pages)
        flood_metadata = {
            "service_version": "11.1",
            "last_edit_epoch_ms": None,
            "data_last_edit_epoch_ms": None,
        }
    else:
        flood_features, flood_metadata = _arcgis_features_by_ids(
            session,
            FEMA_SERVICE,
            where="DFIRM_ID='06001C'",
        )
    flood = _write_geojson(ROOT / "static/FldHaz.geojson", flood_features)
    sra_features, sra_metadata = _arcgis_features(
        session,
        CALFIRE_SRA_SERVICE,
        authority="effective_sra",
        adoption_status="state_adopted_effective_2024-04-01",
    )
    lra_features, lra_metadata = _arcgis_features(
        session,
        CALFIRE_LRA_RECOMMENDATION_SERVICE,
        authority="calfire_2025_lra_recommendation",
        adoption_status="recommendation_only_local_adoption_required",
    )
    responsibility_features, responsibility_metadata = _arcgis_features(
        session,
        CALFIRE_RESPONSIBILITY_SERVICE,
        authority="responsibility_area_2026_1",
        adoption_status="not_applicable",
    )
    sra = _write_geojson(ROOT / "static/FireHazSRA.geojson", sra_features)
    lra = _write_geojson(
        ROOT / "static/FireHazLRARecommended.geojson", lra_features
    )
    responsibility = _write_geojson(
        ROOT / "static/FireResponsibilityAreas.geojson", responsibility_features
    )
    combined = _write_geojson(
        ROOT / "static/FireHaz.geojson", [*sra_features, *lra_features]
    )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": (
            "Automated import is not human approval. LRA recommendation polygons "
            "must never be represented as locally adopted without ordinance evidence."
        ),
        "fema": {
            "county_fips": "06001C",
            "download_url": FEMA_DOWNLOAD,
            "resolved_archive": FEMA_EXPECTED_ARCHIVE,
            "archive_sha256": fema_archive_sha,
            "hazard_service": FEMA_SERVICE,
            "panel_service": FEMA_PANEL_SERVICE,
            "service_metadata": flood_metadata,
            "edition": "Alameda County NFHL 06001C_20260525",
            "retrieved_at": retrieved_at,
            "flood_snapshot": flood,
            "panel_snapshot": panels,
        },
        "calfire": {
            "retrieved_at": retrieved_at,
            "sra": {
                "service": CALFIRE_SRA_SERVICE,
                "edition": "FHSZSRA_23_3; adopted 2024-01-31; effective 2024-04-01",
                "snapshot": sra,
                **sra_metadata,
            },
            "lra_recommendation": {
                "service": CALFIRE_LRA_RECOMMENDATION_SERVICE,
                "edition": "FHSALRA25_v1_All recommendations",
                "snapshot": lra,
                **lra_metadata,
            },
            "responsibility_area": {
                "service": CALFIRE_RESPONSIBILITY_SERVICE,
                "edition": "SRA 26_1 as of 2026-04-09",
                "snapshot": responsibility,
                **responsibility_metadata,
            },
            "compatibility_map_snapshot": combined,
        },
    }
    (ROOT / "data/geospatial/fema_calfire_import_manifest.json").write_bytes(
        _json_bytes(manifest)
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print",
        action="store_true",
        help="print the generated manifest after writing snapshots",
    )
    parser.add_argument(
        "--fema-directory",
        type=Path,
        help=(
            "use an already extracted official 06001C package; intended only for "
            "retrying after a network interruption"
        ),
    )
    parser.add_argument(
        "--fema-service-page",
        type=Path,
        action="append",
        default=[],
        help=(
            "use a saved official NFHL GeoJSON query page after a network "
            "interruption; may be repeated"
        ),
    )
    args = parser.parse_args()
    manifest = build(
        fema_directory=args.fema_directory,
        fema_service_pages=args.fema_service_page or None,
    )
    if args.print:
        print(json.dumps(manifest, indent=2))
    else:
        print(
            "Imported FEMA "
            f"{manifest['fema']['flood_snapshot']['record_count']} zones and "
            f"{manifest['fema']['panel_snapshot']['record_count']} panels; CAL FIRE "
            f"{manifest['calfire']['sra']['snapshot']['record_count']} SRA zones, "
            f"{manifest['calfire']['lra_recommendation']['snapshot']['record_count']} "
            "LRA recommendations, and "
            f"{manifest['calfire']['responsibility_area']['snapshot']['record_count']} "
            "responsibility areas."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
