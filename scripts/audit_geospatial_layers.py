#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import requests
from shapely.geometry import shape


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geospatial.registry import DatasetRegistry, DatasetRegistryError


def audit_dataset(dataset, project_root=PROJECT_ROOT):
    report = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "registered_status": dataset.status,
        "technical_status": "data_unavailable",
        "checksum_matches": None,
        "record_count_matches": None,
        "crs_matches": None,
        "valid_geometry_count": 0,
        "errors": [],
        "warnings": [],
    }
    if dataset.source_type == "remote_service":
        return audit_remote_dataset(dataset, report)
    if dataset.source_type != "local_snapshot":
        report["warnings"].append("This source type does not support the geospatial audit.")
        return report

    path = Path(project_root) / dataset.local_path
    try:
        raw = path.read_bytes()
    except OSError:
        report["errors"].append("Local file is missing or unreadable.")
        return report

    report["checksum_matches"] = hashlib.sha256(raw).hexdigest() == dataset.sha256
    if not report["checksum_matches"]:
        report["technical_status"] = "invalid"
        report["errors"].append("SHA-256 checksum mismatch.")
        return report

    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        report["technical_status"] = "invalid"
        report["errors"].append("File is not valid JSON.")
        return report

    features = payload.get("features") if isinstance(payload, dict) else None
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        report["technical_status"] = "invalid"
        report["errors"].append("File is not a GeoJSON FeatureCollection.")
        return report

    report["record_count_matches"] = len(features) == dataset.record_count
    embedded_crs = str(
        ((payload.get("crs") or {}).get("properties") or {}).get("name") or ""
    )
    report["crs_matches"] = embedded_crs == dataset.source_crs

    valid_geometry_count = 0
    for feature in features:
        try:
            geometry = shape(feature.get("geometry") or {})
        except Exception:
            continue
        if not geometry.is_empty and geometry.is_valid:
            valid_geometry_count += 1
    report["valid_geometry_count"] = valid_geometry_count

    if not report["record_count_matches"]:
        report["errors"].append("Record count mismatch.")
    if not report["crs_matches"]:
        report["errors"].append("Embedded CRS mismatch.")
    if valid_geometry_count == 0:
        report["errors"].append("No valid geometry found.")
    elif valid_geometry_count < len(features):
        report["warnings"].append(
            f"{len(features) - valid_geometry_count} records did not contain valid geometry."
        )

    report["technical_status"] = "invalid" if report["errors"] else "valid"
    if dataset.status == "provisional":
        report["warnings"].append(
            "Technical validation passed, but human verification is still required."
        )
    if not dataset.exact_service_or_download_url:
        report["warnings"].append("Exact service or download URL is not verified.")
    if not dataset.effective_date:
        report["warnings"].append("Effective or publication date is not verified.")
    if not dataset.retrieved_at:
        report["warnings"].append("Retrieval date is not verified.")
    return report


def audit_remote_dataset(dataset, report):
    base_url = dataset.exact_service_or_download_url.rstrip("/")
    if not base_url:
        report["errors"].append("Remote service URL is missing.")
        return report
    bounds = dataset.coverage_bbox
    if not bounds:
        report["errors"].append("Remote dataset coverage bounds are missing.")
        return report

    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2
    envelope = (
        center_lon - 0.01,
        center_lat - 0.01,
        center_lon + 0.01,
        center_lat + 0.01,
    )
    try:
        metadata_response = requests.get(
            base_url,
            params={"f": "json"},
            timeout=10,
        )
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        count_response = requests.get(
            f"{base_url}/query",
            params={"f": "json", "where": "1=1", "returnCountOnly": "true"},
            timeout=10,
        )
        count_response.raise_for_status()
        count_payload = count_response.json()
        map_response = requests.get(
            f"{base_url}/query",
            params={
                "f": "geojson",
                "where": "1=1",
                "geometry": ",".join(str(value) for value in envelope),
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "outSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "geometryPrecision": "5",
                "maxAllowableOffset": "0.0001",
                "resultRecordCount": "20",
            },
            timeout=15,
        )
        map_response.raise_for_status()
        map_payload = map_response.json()
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        report["errors"].append(f"Remote audit request failed: {exc.__class__.__name__}.")
        return report

    if metadata.get("error") or metadata.get("type") != "Feature Layer":
        report["errors"].append("Remote service metadata is invalid.")
    if not isinstance(count_payload.get("count"), int) or count_payload["count"] <= 0:
        report["errors"].append("Remote service record count is invalid or empty.")
    if map_payload.get("type") != "FeatureCollection" or not isinstance(map_payload.get("features"), list):
        report["errors"].append("Bounded map query did not return GeoJSON.")

    report["record_count_matches"] = bool(
        isinstance(count_payload.get("count"), int) and count_payload["count"] > 0
    )
    report["crs_matches"] = metadata.get("geometryType") in {
        "esriGeometryPolygon",
        "esriGeometryPolyline",
    }
    report["valid_geometry_count"] = len(map_payload.get("features") or [])
    report["map_response_bytes"] = len(map_response.content)
    report["map_query_feature_count"] = report["valid_geometry_count"]
    if report["map_response_bytes"] > 3_000_000:
        report["warnings"].append("Bounded simplified map response exceeded 3 MB.")
    if dataset.status == "provisional":
        report["warnings"].append(
            "Technical validation passed, but human viewer comparison is still required."
        )
    report["technical_status"] = "invalid" if report["errors"] else "valid"
    return report


def main():
    try:
        registry = DatasetRegistry()
    except DatasetRegistryError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 1

    reports = [audit_dataset(dataset) for dataset in registry.all()]
    print(json.dumps(reports, indent=2))
    return 1 if any(item["technical_status"] != "valid" for item in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
