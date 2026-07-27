#!/usr/bin/env python3
"""Verify StayReady mapped evidence against reproducible official-source snapshots.

Default mode is deterministic and does not use the network. ``--live`` compares
the committed snapshots with current official services. ``--refresh-official``
is the only mode that writes new official observations; it never changes
expected answers or human-review fields.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import Mock, patch

import requests
from requests.adapters import HTTPAdapter
from shapely.geometry import Point, shape
from shapely.ops import nearest_points
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dam_inundation import check_dam_inundation  # noqa: E402
from geospatial.adapters.arcgis_feature_service import _remote_record_count  # noqa: E402
from geospatial.adapters.local_geojson import LocalGeoJSONAdapter  # noqa: E402
from geospatial.models import GeoPoint  # noqa: E402
from geospatial.registry import DatasetRegistry  # noqa: E402
from geospatial.service import GeospatialEvidenceService  # noqa: E402
from hazard_engine import (  # noqa: E402
    _haversine_km,
    check_fault_layer,
    check_flood_layer,
    check_wildfire_layer,
)
from scripts.source_change_report import canonical_schema  # noqa: E402


FIXTURE_PATH = ROOT / "data" / "geospatial" / "evidence_gold_set_fixtures.json"
SNAPSHOT_PATH = (
    ROOT / "data" / "geospatial" / "evidence_gold_set_official_snapshots.json"
)
JSON_REPORT_PATH = ROOT / "docs" / "evidence-verification-report.json"
MARKDOWN_REPORT_PATH = ROOT / "docs" / "evidence-verification-report.md"
WORKSHEET_PATH = ROOT / "docs" / "evidence-review-worksheet.csv"

REMOTE_DATASET_IDS = {
    "cgs_alquist_priolo_remote",
    "cgs_liquefaction_remote",
    "cgs_earthquake_landslide_remote",
    "cgs_tsunami_hazard_area_remote",
    "dwr_dsod_dam_inundation_remote",
}
CASE_TYPES = {"positive", "negative", "boundary", "not_evaluated", "unavailable"}
ATTRIBUTE_ALLOWLIST = {
    "fema_nfhl_local": {
        "OBJECTID",
        "FLD_AR_ID",
        "FLD_ZONE",
        "ZONE_SUBTY",
        "SFHA_TF",
        "SOURCE_CIT",
        "FIRM_PAN",
        "PANEL_TYP",
        "PNP_REASON",
        "EFF_DATE",
    },
    "calfire_fhsz_local": {
        "OBJECTID",
        "SRA",
        "FHSZ",
        "FHSZ_Description",
    },
    "cgs_alquist_priolo_remote": {
        "OBJECTID",
        "QUAD_NAME",
        "MAP_RELEASED",
        "MAP_REVISED",
        "ZN_RELEASED",
        "ZN_REVISED",
        "GEOPDFLINK",
        "REPORTLINK",
    },
    "cgs_liquefaction_remote": {
        "OBJECTID",
        "QUAD_NAME",
        "RELEASED",
        "REVISED",
        "GEOPDFLINK",
        "REPORTLINK",
    },
    "cgs_earthquake_landslide_remote": {
        "OBJECTID",
        "QUAD_NAME",
        "RELEASED",
        "REVISED",
        "GEOPDFLINK",
        "REPORTLINK",
    },
    "cgs_tsunami_hazard_area_remote": {
        "OBJECTID",
        "Evacuate",
        "County",
        "GIS_Link",
        "KMZ_Link",
        "Map_Link",
    },
    "dwr_dsod_dam_inundation_remote": {
        "FID",
        "NID",
        "StateID",
        "DamName",
        "FailedStr",
        "Scenario",
        "LoadingScn",
        "PubDate",
        "HazardCl",
    },
}
_METADATA_CACHE: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _official_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "StayReadyNow/2.0 evidence-verification",
            "Accept": "application/json",
        }
    )
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=4,
                connect=4,
                read=4,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
        ),
    )
    return session


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_attributes(dataset_id: str, attributes: dict[str, Any]) -> dict[str, Any]:
    allowed = ATTRIBUTE_ALLOWLIST.get(dataset_id, set())
    return {
        key: value
        for key, value in attributes.items()
        if key in allowed
    }


def _request_parameters(case: dict[str, Any]) -> dict[str, str]:
    return {
        "f": "json",
        "where": "1=1",
        "geometry": f"{case['longitude']},{case['latitude']}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "false",
    }


def _service_metadata(endpoint: str, session: requests.Session) -> dict[str, Any]:
    if endpoint in _METADATA_CACHE:
        return _METADATA_CACHE[endpoint]
    response = session.get(
        endpoint,
        params={"f": "json"},
        timeout=(3.05, 20),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error") or not payload.get("fields"):
        raise ValueError("Official service metadata was invalid.")
    editing = payload.get("editingInfo") or {}
    metadata = {
        "service_version": str(payload.get("currentVersion") or ""),
        "schema_fingerprint": canonical_schema(payload),
        "last_edit_epoch_ms": editing.get("lastEditDate"),
        "data_last_edit_epoch_ms": editing.get("dataLastEditDate"),
    }
    _METADATA_CACHE[endpoint] = metadata
    return metadata


def _normalize_point_observation(
    dataset_id: str,
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    attributes = [
        item
        for response in responses
        for item in response.get("attributes", [])
    ]
    if dataset_id == "fema_nfhl_local":
        categories = sorted(
            {
                str(item.get("FLD_ZONE") or "").strip()
                for item in attributes
                if item.get("FLD_ZONE")
            }
        )
        matched = any(str(item.get("SFHA_TF") or "").upper() == "T" for item in attributes)
        zone_d = "D" in categories
        panel_records = [item for item in attributes if item.get("PANEL_TYP")]
        status = (
            "not_evaluated"
            if zone_d or (not categories and not panel_records)
            else "matched"
            if matched
            else "not_matched"
        )
    elif dataset_id == "calfire_fhsz_local":
        categories = sorted(
            {
                str(item.get("FHSZ_Description") or "").strip()
                for item in attributes
                if str(item.get("FHSZ_Description") or "").strip().casefold()
                in {"moderate", "high", "very high"}
            }
        )
        matched = bool(categories)
        responsibility = {
            str(item.get("SRA") or "").strip().upper()
            for item in attributes
            if str(item.get("SRA") or "").strip().upper() in {"SRA", "LRA", "FRA"}
        }
        status = (
            "not_evaluated"
            if responsibility == {"FRA"} or len(responsibility) > 1
            else "matched"
            if matched
            else "not_matched"
        )
    elif dataset_id in {
        "cgs_alquist_priolo_remote",
        "cgs_liquefaction_remote",
        "cgs_earthquake_landslide_remote",
    }:
        categories = sorted(
            {
                str(item.get("QUAD_NAME") or "").strip()
                for item in attributes
                if item.get("QUAD_NAME")
            }
        )
        matched = bool(attributes)
        status = "matched" if matched else "not_matched"
    elif dataset_id == "cgs_tsunami_hazard_area_remote":
        categories = sorted(
            {
                str(item.get("Evacuate") or "").strip()
                for item in attributes
                if item.get("Evacuate")
            }
        )
        matched = "Yes, Tsunami Hazard Area" in categories
        status = "matched" if matched else "not_matched"
    elif dataset_id == "dwr_dsod_dam_inundation_remote":
        categories = sorted(
            {
                " · ".join(
                    str(item.get(key) or "").strip()
                    for key in ("DamName", "Scenario", "LoadingScn")
                    if str(item.get(key) or "").strip()
                )
                for item in attributes
            }
            - {""}
        )
        matched = bool(attributes)
        status = "matched" if matched else "not_matched"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")
    return {
        "status": status,
        "categories": categories,
        "feature_count": len(attributes),
    }


def _query_point_case(
    case: dict[str, Any],
    session: requests.Session,
) -> dict[str, Any]:
    parameters = _request_parameters(case)
    responses = []
    metadata = []
    for endpoint in case["official_service_endpoints"]:
        metadata.append(
            {
                "endpoint": endpoint,
                **_service_metadata(endpoint, session),
            }
        )
        payload = None
        for attempt in range(3):
            response = session.get(
                f"{endpoint.rstrip('/')}/query",
                params=parameters,
                timeout=(3.05, 30),
            )
            response.raise_for_status()
            candidate = response.json()
            if not candidate.get("error") and isinstance(
                candidate.get("features"), list
            ):
                payload = candidate
                break
            if attempt == 2:
                raise ValueError("Official point response was invalid after retries.")
        assert payload is not None
        responses.append(
            {
                "endpoint": endpoint,
                "attributes": [
                    _safe_attributes(
                        case["dataset_id"],
                        feature.get("attributes") or {},
                    )
                    for feature in payload["features"]
                ],
            }
        )
    return {
        "retrieved_at": _utc_now(),
        "request_parameters": parameters,
        "service_metadata": metadata,
        "responses": responses,
        "normalized": _normalize_point_observation(case["dataset_id"], responses),
    }


def _query_fault_case(
    case: dict[str, Any],
    session: requests.Session,
) -> dict[str, Any]:
    endpoint = case["official_service_endpoints"][0]
    parameters = {
        "f": "geojson",
        "where": "1=1",
        "geometry": f"{case['longitude']},{case['latitude']}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": "10",
        "units": "esriSRUnit_Kilometer",
        "outSR": "4326",
        "outFields": "*",
        "returnGeometry": "true",
        "resultRecordCount": "2000",
    }
    response = session.get(
        f"{endpoint.rstrip('/')}/query",
        params=parameters,
        timeout=(3.05, 30),
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error") or not isinstance(payload.get("features"), list):
        raise ValueError("Official fault response was invalid.")
    address_point = Point(case["longitude"], case["latitude"])
    nearest = None
    for feature in payload["features"]:
        if not feature.get("geometry"):
            continue
        line = shape(feature["geometry"])
        nearest_point = nearest_points(address_point, line)[1]
        distance_km = _haversine_km(
            case["latitude"],
            case["longitude"],
            nearest_point.y,
            nearest_point.x,
        )
        properties = feature.get("properties") or {}
        candidate = {
            "fault_name": str(properties.get("fault_name") or "Mapped fault trace"),
            "fault_id": str(properties.get("fault_id") or ""),
            "section_id": str(properties.get("section_id") or ""),
            "distance_km": round(distance_km, 4),
        }
        if nearest is None or candidate["distance_km"] < nearest["distance_km"]:
            nearest = candidate
    if nearest is None:
        raise ValueError("Official fault service returned no usable nearby geometry.")
    threshold = 2.0
    return {
        "retrieved_at": _utc_now(),
        "request_parameters": parameters,
        "service_metadata": [
            {
                "endpoint": endpoint,
                **_service_metadata(endpoint, session),
            }
        ],
        "responses": [{"endpoint": endpoint, "nearest": nearest}],
        "normalized": {
            "status": "matched" if nearest["distance_km"] <= threshold else "not_matched",
            "categories": [nearest["fault_name"]],
            "distance_km": nearest["distance_km"],
            "feature_count": len(payload["features"]),
        },
    }


def query_official(case: dict[str, Any], session: requests.Session) -> dict[str, Any]:
    if case["verification_strategy"] == "official_point_query":
        return _query_point_case(case, session)
    if case["verification_strategy"] == "official_fault_proximity":
        return _query_fault_case(case, session)
    raise ValueError("This case does not have a live official query.")


def refresh_official_snapshots(fixtures: dict[str, Any]) -> dict[str, Any]:
    session = _official_session()
    observations = {}
    for case in fixtures["cases"]:
        if case["verification_strategy"] not in {
            "official_point_query",
            "official_fault_proximity",
        }:
            continue
        observations[case["case_id"]] = query_official(case, session)
    payload = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "policy": (
            "Technical observations only. Refreshing this file never changes expected "
            "answers, dataset review status, or human approval."
        ),
        "cases": observations,
    }
    SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _fake_response(payload: dict[str, Any]) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def _remote_stayready_result(
    case: dict[str, Any],
    observation: dict[str, Any],
    registry: DatasetRegistry,
) -> dict[str, Any]:
    attributes = [
        item
        for response in observation["responses"]
        for item in response.get("attributes", [])
    ]
    count_response = _fake_response({"count": max(1, len(attributes))})
    point_response = _fake_response(
        {"features": [{"attributes": item} for item in attributes]}
    )
    _remote_record_count.cache_clear()
    previous = os.environ.pop("STAYREADY_REQUIRE_VERIFIED_DATASETS", None)
    try:
        with patch(
            "geospatial.adapters.arcgis_feature_service.requests.get",
            side_effect=[count_response, point_response],
        ):
            service = GeospatialEvidenceService(project_root=ROOT, registry=registry)
            if case["dataset_id"] == "dwr_dsod_dam_inundation_remote":
                result = check_dam_inundation(
                    case["latitude"],
                    case["longitude"],
                    service=service,
                )
                return {
                    "status": (
                        "matched"
                        if result["inside_inundation_boundary"]
                        else "not_matched"
                        if result["inside_inundation_boundary"] is False
                        else "data_unavailable"
                    ),
                    "categories": sorted(
                        {
                            " · ".join(
                                str(item.get(key) or "").strip()
                                for key in ("dam_name", "scenario", "loading_scenario")
                                if str(item.get(key) or "").strip()
                            )
                            for item in result["matched_dam_scenarios"]
                        }
                        - {""}
                    ),
                    "claim_type": "scenario",
                    "limitations": result["limitations"],
                }
            evidence = service.check_point(
                case["dataset_id"],
                case["latitude"],
                case["longitude"],
            )
    finally:
        if previous is not None:
            os.environ["STAYREADY_REQUIRE_VERIFIED_DATASETS"] = previous
    categories = []
    for feature in evidence.matched_features:
        feature_attributes = feature.get("attributes") or {}
        if case["dataset_id"] == "cgs_tsunami_hazard_area_remote":
            value = feature_attributes.get("Evacuate")
        else:
            value = feature_attributes.get("QUAD_NAME")
        if value:
            categories.append(str(value))
    return {
        "status": (
            "matched"
            if evidence.matched is True
            else "not_matched"
            if evidence.matched is False
            else "data_unavailable"
        ),
        "categories": sorted(set(categories)),
        "claim_type": evidence.claim_type,
        "limitations": evidence.limitations,
    }


def _simulate_unavailable(
    case: dict[str, Any],
    registry: DatasetRegistry,
) -> dict[str, Any]:
    dataset = registry.get(case["dataset_id"])
    if case["dataset_id"] == "usgs_cgs_faults_local":
        with patch(
            "hazard_engine.load_geojson",
            return_value={"available": False, "data": None},
        ):
            result = check_fault_layer(case["latitude"], case["longitude"])
        return {
            "status": "data_unavailable" if not result["checked"] else "unexpected",
            "categories": [],
            "limitations": ["Simulated unreadable local fault layer."],
        }
    if dataset.source_type == "local_snapshot":
        unavailable_dataset = dataset.model_copy(
            update={
                "status": "data_unavailable",
                "local_path": "data/geospatial/does-not-exist.geojson",
            }
        )
        evidence = LocalGeoJSONAdapter(ROOT).check_point(
            unavailable_dataset,
            GeoPoint(lat=case["latitude"], lon=case["longitude"]),
        )
        return {
            "status": "data_unavailable"
            if evidence.evidence_status == "data_unavailable"
            else "unexpected",
            "categories": [],
            "limitations": evidence.limitations,
        }
    previous = os.environ.pop("STAYREADY_REQUIRE_VERIFIED_DATASETS", None)
    try:
        _remote_record_count.cache_clear()
        with patch(
            "geospatial.adapters.arcgis_feature_service.requests.get",
            side_effect=requests.Timeout("deterministic timeout"),
        ):
            evidence = GeospatialEvidenceService(
                project_root=ROOT,
                registry=registry,
            ).check_point(
                case["dataset_id"],
                case["latitude"],
                case["longitude"],
            )
    finally:
        if previous is not None:
            os.environ["STAYREADY_REQUIRE_VERIFIED_DATASETS"] = previous
    return {
        "status": "data_unavailable"
        if evidence.evidence_status == "data_unavailable"
        else "unexpected",
        "categories": [],
        "limitations": evidence.limitations,
    }


def evaluate_stayready(
    case: dict[str, Any],
    observation: dict[str, Any] | None,
    registry: DatasetRegistry,
) -> dict[str, Any]:
    strategy = case["verification_strategy"]
    if strategy == "scope_review_required":
        return {
            "status": "not_evaluated",
            "categories": [],
            "limitations": case["limitations"],
            "skipped_reason": (
                "The registered source does not contain a separate authoritative "
                "evaluation-coverage control."
            ),
        }
    if strategy == "simulated_unavailable":
        return _simulate_unavailable(case, registry)
    if case["dataset_id"] == "fema_nfhl_local":
        result = check_flood_layer(case["latitude"], case["longitude"])
        return {
            "status": result.get(
                "evaluation_status",
                "matched" if result["inside"] else "not_matched",
            ),
            "categories": sorted(
                {
                    str(item.get("zone"))
                    for item in result.get("layers", [])
                    if item.get("zone")
                }
            ),
            "claim_type": "regulatory_zone",
            "limitations": (
                result.get("geospatial_evidence", {}).get("limitations")
                or case["limitations"]
            ),
        }
    if case["dataset_id"] == "calfire_fhsz_local":
        result = check_wildfire_layer(case["latitude"], case["longitude"])
        return {
            "status": result.get(
                "evaluation_status",
                "matched" if result["inside"] else "not_matched",
            ),
            "categories": sorted(
                {
                    str(item.get("hazard_class"))
                    for item in result["layers"]
                    if item.get("hazard_class")
                }
            ),
            "claim_type": "hazard_zone",
            "limitations": case["limitations"],
        }
    if case["dataset_id"] == "usgs_cgs_faults_local":
        result = check_fault_layer(case["latitude"], case["longitude"])
        nearest = result.get("nearest_layer") or {}
        return {
            "status": "matched" if result.get("near") else "not_matched",
            "categories": [nearest["name"]] if nearest.get("name") else [],
            "distance_km": nearest.get("distance_km"),
            "claim_type": "proximity",
            "limitations": case["limitations"],
        }
    if case["dataset_id"] in REMOTE_DATASET_IDS and observation is not None:
        return _remote_stayready_result(case, observation, registry)
    raise ValueError(f"Unsupported case: {case['case_id']}")


def _compare_case(
    case: dict[str, Any],
    observation: dict[str, Any] | None,
    registry: DatasetRegistry,
) -> dict[str, Any]:
    stayready = evaluate_stayready(case, observation, registry)
    if case["verification_strategy"] == "simulated_unavailable":
        official = {"status": "unavailable", "categories": []}
    elif case["verification_strategy"] == "scope_review_required":
        official = {"status": "not_evaluated", "categories": []}
    elif observation:
        official = observation["normalized"]
    else:
        official = {"status": "missing_snapshot", "categories": []}

    checks = {
        "official_status": official["status"] == case["expected_official_status"],
        "official_category": sorted(official.get("categories", []))
        == sorted(case["expected_official_category"]),
        "stayready_status": stayready["status"] == case["expected_stayready_status"],
        "stayready_category": sorted(stayready.get("categories", []))
        == sorted(case["expected_stayready_category"]),
    }
    expectation = case["comparison_expectation"]
    if expectation == "external_review":
        outcome = "skipped_external_review"
    elif all(checks.values()):
        outcome = "known_mismatch" if expectation == "known_mismatch" else "passed"
    else:
        outcome = "failed"
    return {
        "case_id": case["case_id"],
        "dataset_id": case["dataset_id"],
        "test_case_type": case["test_case_type"],
        "public_fixture": case["public_fixture"],
        "expected_scope": case["expected_scope"],
        "official": official,
        "stayready": stayready,
        "checks": checks,
        "outcome": outcome,
        "comparison_expectation": expectation,
        "limitations": case["limitations"],
        "uncertainty": case["uncertainty"],
        "external_review_required": case["external_review_required"],
        "official_response_reference": (
            f"data/geospatial/evidence_gold_set_official_snapshots.json"
            f"#cases.{case['case_id']}"
            if observation
            else "not applicable"
        ),
        "stayready_result_reference": (
            "hazard_engine.py / geospatial adapters / dam_inundation.py"
        ),
    }


def run_verification(
    fixtures: dict[str, Any],
    snapshots: dict[str, Any],
) -> dict[str, Any]:
    registry = DatasetRegistry()
    dataset_ids = {dataset.dataset_id for dataset in registry.all()}
    fixture_ids = {case["dataset_id"] for case in fixtures["cases"]}
    if fixture_ids != dataset_ids:
        raise ValueError("Gold set does not cover exactly the registered datasets.")
    for dataset_id in dataset_ids:
        case_types = {
            case["test_case_type"]
            for case in fixtures["cases"]
            if case["dataset_id"] == dataset_id
        }
        if case_types != CASE_TYPES:
            raise ValueError(f"{dataset_id} does not contain all required case types.")

    results = [
        _compare_case(
            case,
            snapshots.get("cases", {}).get(case["case_id"]),
            registry,
        )
        for case in fixtures["cases"]
    ]
    summary = {
        "passed": sum(item["outcome"] == "passed" for item in results),
        "known_mismatch": sum(
            item["outcome"] == "known_mismatch" for item in results
        ),
        "failed": sum(item["outcome"] == "failed" for item in results),
        "skipped_external_review": sum(
            item["outcome"] == "skipped_external_review" for item in results
        ),
        "total": len(results),
    }
    by_dataset = {}
    for dataset in registry.all():
        rows = [item for item in results if item["dataset_id"] == dataset.dataset_id]
        by_dataset[dataset.dataset_id] = {
            "edition": dataset.edition,
            "dataset_version": dataset.dataset_version,
            "passed": sum(item["outcome"] == "passed" for item in rows),
            "known_mismatch": sum(
                item["outcome"] == "known_mismatch" for item in rows
            ),
            "failed": sum(item["outcome"] == "failed" for item in rows),
            "skipped_external_review": sum(
                item["outcome"] == "skipped_external_review" for item in rows
            ),
            "publication_status": dataset.status,
            "external_review_required": True,
        }
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "mode": "deterministic_snapshot",
        "policy": (
            "Automated technical comparison is not human approval. Provisional "
            "datasets remain withheld in production."
        ),
        "summary": summary,
        "datasets": by_dataset,
        "cases": results,
    }


def compare_live(
    fixtures: dict[str, Any],
    snapshots: dict[str, Any],
) -> dict[str, Any]:
    session = _official_session()
    results = []
    for case in fixtures["cases"]:
        if case["verification_strategy"] not in {
            "official_point_query",
            "official_fault_proximity",
        }:
            continue
        recorded = snapshots.get("cases", {}).get(case["case_id"])
        if not recorded:
            results.append(
                {"case_id": case["case_id"], "outcome": "failed", "reason": "missing_snapshot"}
            )
            continue
        try:
            live = query_official(case, session)
            recorded_normalized = recorded["normalized"]
            live_normalized = live["normalized"]
            distance_matches = True
            if "distance_km" in recorded_normalized:
                distance_matches = (
                    abs(
                        float(recorded_normalized["distance_km"])
                        - float(live_normalized["distance_km"])
                    )
                    <= 0.02
                )
            matches = (
                recorded_normalized["status"] == live_normalized["status"]
                and sorted(recorded_normalized.get("categories", []))
                == sorted(live_normalized.get("categories", []))
                and distance_matches
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "outcome": "passed" if matches else "failed",
                    "recorded": recorded_normalized,
                    "live": live_normalized,
                }
            )
        except Exception as error:
            results.append(
                {
                    "case_id": case["case_id"],
                    "outcome": "unavailable",
                    "reason": type(error).__name__,
                }
            )
    return {
        "generated_at": _utc_now(),
        "mode": "optional_live_source_check",
        "summary": {
            "passed": sum(item["outcome"] == "passed" for item in results),
            "failed": sum(item["outcome"] == "failed" for item in results),
            "unavailable": sum(item["outcome"] == "unavailable" for item in results),
            "total": len(results),
        },
        "cases": results,
    }


def _markdown_report(report: dict[str, Any], live: dict[str, Any] | None) -> str:
    summary = report["summary"]
    lines = [
        "# StayReady evidence verification report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Automated comparison is technical evidence only. It does not constitute "
        "human GIS, emergency-management, legal, accessibility, or publication approval.",
        "",
        "## Summary",
        "",
        f"- Deterministic cases: {summary['total']}",
        f"- Passed: {summary['passed']}",
        f"- Known source/snapshot mismatches: {summary['known_mismatch']}",
        f"- Failed unexpectedly: {summary['failed']}",
        f"- Skipped pending authoritative scope review: {summary['skipped_external_review']}",
        "",
        "## Dataset results",
        "",
        "| Dataset | Edition/version | Passed | Known mismatch | Failed | Scope-review skip | Publication |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for dataset_id, item in report["datasets"].items():
        lines.append(
            f"| `{dataset_id}` | {item['edition']} / `{item['dataset_version']}` "
            f"| {item['passed']} | {item['known_mismatch']} | {item['failed']} "
            f"| {item['skipped_external_review']} | {item['publication_status']} |"
        )
    lines.extend(["", "## Case results", ""])
    for item in report["cases"]:
        lines.extend(
            [
                f"### {item['case_id']}",
                "",
                f"- Outcome: **{item['outcome']}**",
                f"- Fixture: {item['public_fixture']}",
                f"- Official: `{item['official']['status']}`; "
                f"categories: {', '.join(item['official'].get('categories', [])) or 'none'}",
                f"- StayReady: `{item['stayready']['status']}`; "
                f"categories: {', '.join(item['stayready'].get('categories', [])) or 'none'}",
                f"- Scope: {item['expected_scope']}",
                f"- Uncertainty: {item['uncertainty'] or 'None recorded by automation.'}",
                f"- External review: {'required' if item['external_review_required'] else 'not required for this simulated case'}",
                "",
            ]
        )
    if live:
        lines.extend(
            [
                "## Optional live-source check",
                "",
                f"- Passed: {live['summary']['passed']}",
                f"- Failed: {live['summary']['failed']}",
                f"- Unavailable: {live['summary']['unavailable']}",
                f"- Total queried: {live['summary']['total']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Commands",
            "",
            "```bash",
            "python scripts/verify_evidence_gold_set.py",
            "python scripts/verify_evidence_gold_set.py --live",
            "python scripts/verify_evidence_gold_set.py --refresh-official --live",
            "```",
            "",
            "`--refresh-official` updates observations only. Expected results and human "
            "approval fields are never updated automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_worksheet(
    fixtures: dict[str, Any],
    report: dict[str, Any],
) -> None:
    registry = {item.dataset_id: item for item in DatasetRegistry().all()}
    results = {item["case_id"]: item for item in report["cases"]}
    fieldnames = [
        "dataset_id",
        "dataset_version",
        "test_case_type",
        "public_fixture",
        "fixture_address_or_identifier",
        "expected_category",
        "expected_scope",
        "expected_status",
        "official_viewer_url",
        "official_service_endpoint",
        "official_response_reference",
        "stayready_result_reference",
        "automated_comparison_result",
        "automated_review_method",
        "automated_reviewed_at",
        "limitations",
        "uncertainty",
        "external_review_required",
        "human_reviewer",
        "human_reviewed_at",
        "human_approval",
    ]
    with WORKSHEET_PATH.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for case in fixtures["cases"]:
            dataset = registry[case["dataset_id"]]
            result = results[case["case_id"]]
            writer.writerow(
                {
                    "dataset_id": case["dataset_id"],
                    "dataset_version": dataset.dataset_version,
                    "test_case_type": case["test_case_type"],
                    "public_fixture": case["public_fixture"],
                    "fixture_address_or_identifier": (
                        case["fixture_address"] or case["public_fixture"]
                    ),
                    "expected_category": " | ".join(case["expected_official_category"]),
                    "expected_scope": case["expected_scope"],
                    "expected_status": case["expected_stayready_status"],
                    "official_viewer_url": dataset.official_viewer_url
                    or dataset.authoritative_landing_url,
                    "official_service_endpoint": " | ".join(
                        case["official_service_endpoints"]
                    ),
                    "official_response_reference": result[
                        "official_response_reference"
                    ],
                    "stayready_result_reference": result[
                        "stayready_result_reference"
                    ],
                    "automated_comparison_result": result["outcome"],
                    "automated_review_method": (
                        "Deterministic official-response snapshot plus StayReady "
                        f"{case['verification_strategy']} logic"
                    ),
                    "automated_reviewed_at": report["generated_at"],
                    "limitations": " | ".join(case["limitations"]),
                    "uncertainty": case["uncertainty"],
                    "external_review_required": str(
                        case["external_review_required"]
                    ).lower(),
                    "human_reviewer": "not provided",
                    "human_reviewed_at": "",
                    "human_approval": "pending external review",
                }
            )


def write_reports(
    fixtures: dict[str, Any],
    report: dict[str, Any],
    live: dict[str, Any] | None,
) -> None:
    payload = dict(report)
    if live:
        payload["live"] = live
    JSON_REPORT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_REPORT_PATH.write_text(
        _markdown_report(report, live),
        encoding="utf-8",
    )
    _write_worksheet(fixtures, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-official",
        action="store_true",
        help="Explicitly refresh official observations without changing expectations.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Compare current official services with committed observations.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not rewrite reports or the technical worksheet.",
    )
    args = parser.parse_args()

    fixtures = _load_json(FIXTURE_PATH)
    if args.refresh_official:
        snapshots = refresh_official_snapshots(fixtures)
    elif SNAPSHOT_PATH.exists():
        snapshots = _load_json(SNAPSHOT_PATH)
    else:
        print("ERROR: official evidence snapshots are missing.", file=sys.stderr)
        return 2

    report = run_verification(fixtures, snapshots)
    live_report = compare_live(fixtures, snapshots) if args.live else None
    if not args.no_write:
        write_reports(fixtures, report, live_report)

    print(
        "Deterministic evidence verification: "
        f"{report['summary']['passed']} passed, "
        f"{report['summary']['known_mismatch']} known mismatch, "
        f"{report['summary']['failed']} failed, "
        f"{report['summary']['skipped_external_review']} scope-review skip."
    )
    if live_report:
        print(
            "Live official-source verification: "
            f"{live_report['summary']['passed']} passed, "
            f"{live_report['summary']['failed']} failed, "
            f"{live_report['summary']['unavailable']} unavailable."
        )
    if report["summary"]["failed"]:
        return 1
    if live_report and live_report["summary"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
