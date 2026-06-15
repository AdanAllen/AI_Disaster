import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional

from shapely.geometry import Point, shape
from shapely.ops import nearest_points

from action_library_service import select_actions
from dam_inundation import check_dam_inundation
from geospatial.registry import DatasetRegistryError
from geospatial.service import GeospatialEvidenceService
from pydantic_models import HazardResult, PreparednessAction, RecoveryQuestion, ResidentGuidanceItem, SpecializedGuidance
from source_registry import get_city_chunks, get_local_plan_for_city, get_resident_guidance, get_source, get_sources_for_hazard


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CGS_DATASETS = {
    "cgs_alquist_priolo_remote": {
        "source_id": "cgs_alquist_priolo",
        "map_layer_key": "alquist-priolo",
        "match_label": "Inside a CGS mapped Alquist-Priolo fault rupture zone.",
        "plain_definition": "Alquist-Priolo fault zones are official zones around active faults where the ground could rupture at the surface during an earthquake.",
        "match_meaning": "The location appears inside a mapped surface fault rupture zone.",
        "nonmatch_meaning": "No mapped surface fault rupture zone matched the location in this checked dataset.",
        "does_not_mean": "This is not a general earthquake shaking score. Areas outside the zone can still experience strong shaking.",
    },
    "cgs_liquefaction_remote": {
        "source_id": "cgs_liquefaction",
        "map_layer_key": "liquefaction",
        "match_label": "Inside a CGS mapped liquefaction zone.",
        "plain_definition": "Liquefaction can happen when strong earthquake shaking makes loose, wet soil temporarily lose strength.",
        "match_meaning": "The location appears inside a CGS mapped area where liquefaction may occur during a strong earthquake.",
        "nonmatch_meaning": "No CGS mapped liquefaction zone matched the location in this checked dataset.",
        "does_not_mean": "This does not guarantee damage and is not a site-specific engineering report.",
    },
    "cgs_earthquake_landslide_remote": {
        "source_id": "cgs_earthquake_landslide",
        "map_layer_key": "earthquake-landslide",
        "match_label": "Inside a CGS mapped earthquake-induced landslide zone.",
        "plain_definition": "Earthquake-induced landslides are slope failures triggered by strong earthquake shaking.",
        "match_meaning": "The location appears inside a CGS mapped area where earthquake shaking may trigger landslides.",
        "nonmatch_meaning": "No CGS mapped earthquake-induced landslide zone matched the location in this checked dataset.",
        "does_not_mean": "This does not predict a landslide at the exact property and does not cover every landslide risk.",
    },
    "cgs_tsunami_hazard_area_remote": {
        "source_id": "cgs_tsunami_hazard_area",
        "map_layer_key": "tsunami",
        "match_label": "Inside a CGS mapped tsunami hazard area.",
        "plain_definition": "Tsunami hazard areas are places that could be flooded during a tsunami event and are mainly used for evacuation planning.",
        "match_meaning": "The location appears inside a CGS mapped tsunami hazard area.",
        "nonmatch_meaning": "No CGS mapped tsunami hazard area matched the location in this checked dataset.",
        "does_not_mean": "This is not a real-time evacuation order, a legal property determination, or a replacement for official alerts.",
    },
}

CGS_HAZARD_ALIASES = {
    "earthquake": "earthquake",
    "tsunami": "tsunami",
    "tsunami-seiche": "tsunami",
}

_LOCATION_CHECK_CACHE = {}
_LOCATION_CHECK_CACHE_LOCK = threading.Lock()
_LOCATION_CHECK_CACHE_TTL_SECONDS = 300
_LOCATION_CHECK_FAILURE_TTL_SECONDS = 30

RECOVERY_CITATION = {
    "source_id": "ready_recovery",
    "source_name": "Ready.gov Recovering from Disaster",
    "source_url": "https://www.ready.gov/recovering-disaster",
    "source_summary": "Ready.gov recovery guidance covers safety, documents, insurance, immediate needs, and household recovery.",
}

RECOVERY_QUESTIONS = [
    RecoveryQuestion(
        id="documents",
        question="Where are copies of your IDs, insurance papers, lease or mortgage records, and medical documents?",
        **RECOVERY_CITATION,
    ),
    RecoveryQuestion(
        id="housing",
        question="Where could your household stay for the first week if your home is unsafe?",
        **RECOVERY_CITATION,
    ),
    RecoveryQuestion(
        id="medical",
        question="How would you replace medications, medical devices, or backup power if services were disrupted?",
        **RECOVERY_CITATION,
    ),
    RecoveryQuestion(
        id="family-continuity",
        question="How will family members, pets, children, school, and work needs be handled during recovery?",
        **RECOVERY_CITATION,
    ),
    RecoveryQuestion(
        id="transportation",
        question="If your normal vehicle, transit route, or rideshare option is unavailable, what is your backup transportation plan?",
        **RECOVERY_CITATION,
    ),
    RecoveryQuestion(
        id="financial-recovery",
        question="How would you cover deductibles, temporary supplies, lost work time, or urgent repairs during the first month of recovery?",
        **RECOVERY_CITATION,
    ),
]

def _hazard_type(hazard: Dict) -> str:
    return (
        hazard.get("hazard_id")
        or hazard.get("slug")
        or hazard.get("name", "")
        or hazard.get("label", "")
    ).strip().lower()


def _hazard_label(hazard: Dict, fallback: str) -> str:
    return hazard.get("label") or hazard.get("name") or fallback.title()


def display_scope(scope: str) -> str:
    labels = {
        "address_level": "Address-level",
        "jurisdiction_level": "Jurisdiction-level",
        "zip_estimate": "ZIP estimate",
        "county_fallback": "County fallback",
    }
    return labels.get(scope, scope.replace("_", " ").title())


def display_data_status(status: str) -> str:
    labels = {
        "checked": "Mapped match found",
        "not_checked": "Not checked",
        "data_unavailable": "Not checked — map data unavailable",
        "not_in_layer": "No mapped match found",
        "fallback_used": "Regional preparedness priority",
        "needs_review": "Not determined from checked map data",
    }
    return labels.get(status, status.replace("_", " ").title())


def _dedupe_text(items: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for item in items:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _actions_for(location_result, hazard_type: str, *time_buckets: str, limit: int = 4) -> List[PreparednessAction]:
    return select_actions(
        hazards=[hazard_type],
        time_buckets=time_buckets or ["before"],
        city=getattr(location_result, "city", ""),
        county=getattr(location_result, "county", ""),
        trigger_types=["general", "hazard_result", "location"],
        limit=limit,
    )


def _resident_guidance_items(result: HazardResult, *phases: str) -> List:
    items = []
    guidance = result.specialized_guidance.resident_guidance or {}
    for phase in phases:
        items.extend(guidance.get(phase, []))
    return items


def _local_summary(result: HazardResult) -> str:
    location_context = _dedupe_text(result.specialized_guidance.location_specific_context)
    if location_context:
        return location_context[0]
    context = _dedupe_text(result.specialized_guidance.city_context)
    if context:
        return context[0]
    return result.why_shown


def _local_impact(result: HazardResult) -> str:
    recovery_questions = [
        item.recovery_question
        for item in _resident_guidance_items(result, "recovery")
        if getattr(item, "recovery_question", "")
    ]
    if recovery_questions:
        return "Recovery planning question: " + recovery_questions[0]
    if result.specialized_guidance.recovery_needs:
        return " ".join(result.specialized_guidance.recovery_needs[:2])
    return "Reviewed recovery actions are listed separately with their supporting sources."


def _local_top_risks(result: HazardResult) -> List[str]:
    risks = [
        item.plain_language
        for item in _resident_guidance_items(result, "hazard_priority", "local_context", "limitations")
    ]
    risks.extend(result.limitations[:2])
    return _dedupe_text(risks)[:4]


def _local_locations(result: HazardResult) -> List[str]:
    locations = []
    if result.local_plan_match:
        name = result.local_plan_match.get("name")
        plan_name = result.local_plan_match.get("plan_name")
        if name:
            locations.append(f"Resolved jurisdiction: {name}")
        if plan_name:
            locations.append(f"Local source context: {plan_name}")
    if result.matched_layers:
        locations.extend(layer.get("name", "") for layer in result.matched_layers)
    locations.append(f"Precision shown by app: {result.location_precision.replace('_', ' ')}")
    return _dedupe_text(locations)


@lru_cache(maxsize=4)
def load_geojson(filename: str) -> Dict:
    path = os.path.join(BASE_DIR, "static", filename)
    if not os.path.exists(path):
        return {
            "available": False,
            "data": None,
            "message": "Official layer unavailable — not checked.",
        }
    try:
        with open(path, "r", encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {
            "available": False,
            "data": None,
            "message": "Official layer unavailable — not checked.",
        }
    if (
        not isinstance(data, dict)
        or data.get("type") != "FeatureCollection"
        or not isinstance(data.get("features"), list)
        or not data.get("features")
    ):
        return {
            "available": False,
            "data": None,
            "message": "Official layer unavailable — not checked.",
        }
    return {"available": True, "data": data, "message": ""}


def _not_checked_result(message: str = "Location coordinates were unavailable.") -> Dict:
    return {
        "checked": False,
        "data_status": "not_checked",
        "message": message,
        "inside": None,
        "near": None,
        "layers": [],
    }


def _unavailable_result() -> Dict:
    return {
        "checked": False,
        "data_status": "data_unavailable",
        "message": "Official layer unavailable — not checked.",
        "inside": None,
        "near": None,
        "layers": [],
    }


def check_flood_layer(lat: Optional[float], lon: Optional[float]) -> Dict:
    if lat is None or lon is None:
        return _not_checked_result()
    try:
        evidence = GeospatialEvidenceService(project_root=BASE_DIR).check_point(
            "fema_nfhl_local",
            float(lat),
            float(lon),
        )
    except (DatasetRegistryError, TypeError, ValueError):
        return _unavailable_result()

    evidence_payload = evidence.model_dump(mode="json")
    if evidence.evidence_status != "checked":
        result = _unavailable_result()
        result["data_status"] = (
            "not_checked"
            if evidence.evidence_status == "not_checked"
            else "data_unavailable"
        )
        result["message"] = evidence.limitations[0]
        result["geospatial_evidence"] = evidence_payload
        return result
    polygon_match = bool(evidence.matched)
    sfha_layers = [
        layer for layer in evidence.matched_features
        if _fema_layer_is_sfha(layer)
    ]
    return {
        "checked": True,
        "data_status": "checked" if sfha_layers else "not_in_layer",
        "message": "",
        "inside": bool(sfha_layers),
        "polygon_match": polygon_match,
        "sfha_match": bool(sfha_layers),
        "sfha_layers": sfha_layers,
        "layers": evidence.matched_features,
        "geospatial_evidence": evidence_payload,
    }


def check_wildfire_layer(lat: Optional[float], lon: Optional[float]) -> Dict:
    if lat is None or lon is None:
        return _not_checked_result()

    point = Point(float(lon), float(lat))
    loaded = load_geojson("FireHaz.geojson")
    if not loaded["available"]:
        return _unavailable_result()
    data = loaded["data"]
    layers = []
    valid_features = 0
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            polygon = shape(geometry)
        except Exception:
            continue
        if polygon.is_empty or not polygon.is_valid:
            continue
        valid_features += 1
        if polygon.contains(point) or polygon.touches(point):
            props = feature.get("properties", {})
            hazard_class = props.get("HAZ_CLASS") or props.get("VH_REC") or "Unknown"
            layers.append({
                "layer_id": f"calfire_fhsz_{props.get('HAZ_CODE', 'unknown')}",
                "name": f"CAL FIRE Fire Hazard Severity Zone: {hazard_class}",
                "hazard_class": hazard_class,
                "state_responsibility_area": props.get("SRA", ""),
                "incorporated": props.get("INCORP", ""),
            })

    if valid_features == 0:
        return _unavailable_result()
    return {
        "checked": True,
        "data_status": "checked" if layers else "not_in_layer",
        "message": "",
        "inside": bool(layers),
        "layers": layers,
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    value = min(1.0, max(0.0, value))
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def check_fault_layer(lat: Optional[float], lon: Optional[float], threshold_km: float = 2.0) -> Dict:
    if lat is None or lon is None:
        return _not_checked_result()

    point = Point(float(lon), float(lat))
    loaded = load_geojson("Fault_lines.Geojson")
    if not loaded["available"]:
        return _unavailable_result()
    data = loaded["data"]
    nearest = None
    nearest_km = None
    valid_features = 0
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            line = shape(geometry)
        except Exception:
            continue
        if line.is_empty or not line.is_valid:
            continue
        valid_features += 1
        nearest_point = nearest_points(point, line)[1]
        distance_km = _haversine_km(
            float(lat),
            float(lon),
            nearest_point.y,
            nearest_point.x,
        )
        if nearest_km is None or distance_km < nearest_km:
            props = feature.get("properties", {})
            nearest_km = distance_km
            nearest = {
                "layer_id": props.get("fault_id") or "mapped_fault",
                "name": props.get("fault_name") or "Mapped fault line",
                "distance_km": round(distance_km, 2),
                "line_type": props.get("linetype", ""),
                "age": props.get("age", ""),
                "source_url": props.get("fault_url", ""),
            }

    if valid_features == 0:
        return _unavailable_result()
    near = nearest_km is not None and nearest_km <= threshold_km
    return {
        "checked": True,
        "data_status": "checked",
        "message": "",
        "near": near,
        "layers": [nearest] if near and nearest else [],
        "nearest_layer": nearest,
    }


def _cgs_public_evidence(dataset_id: str, evidence) -> Dict:
    payload = evidence.model_dump(mode="json")
    config = CGS_DATASETS[dataset_id]
    checked = evidence.evidence_status == "checked"
    available = evidence.evidence_status not in {"data_unavailable", "not_checked"}
    matched = evidence.matched if checked else None
    if not available:
        exposure = "data_unavailable"
        result_label = "Official CGS layer unavailable — not checked."
        status_label = "Not checked"
        what_this_means = "The official map service was unavailable, so StayReady did not evaluate this location with this layer."
    elif not checked:
        exposure = "unknown"
        result_label = "This address was not evaluated by the registered CGS dataset."
        status_label = "Not determined from checked map data"
        what_this_means = "This location was not evaluated by this map layer."
    elif matched:
        exposure = "mapped_match"
        result_label = config["match_label"]
        status_label = "Mapped match found"
        what_this_means = config["match_meaning"]
    else:
        exposure = "no_mapped_match"
        result_label = "No mapped match found in the checked CGS dataset."
        status_label = "No mapped match found"
        what_this_means = config["nonmatch_meaning"]
    provenance = payload.get("provenance") or {}
    return {
        "dataset_id": dataset_id,
        "map_layer_key": config["map_layer_key"],
        "dataset_name": provenance.get("dataset_name", ""),
        "checked": checked,
        "data_available": available,
        "matched": matched,
        "exposure": exposure,
        "result_label": result_label,
        "status_label": status_label,
        "plain_definition": config["plain_definition"],
        "what_this_means": what_this_means,
        "why_it_matters": provenance.get("source_summary") or provenance.get("intended_claim", ""),
        "what_this_does_not_mean": config["does_not_mean"],
        "priority_band": "Mapped evidence" if matched else "Not an address-level priority ranking",
        "ranking_score": None,
        "evidence_tier": "official_mapped_data",
        "claim_type": payload.get("claim_type", ""),
        "precision": payload.get("precision", "address_point"),
        "source_summary": provenance.get("source_summary") or provenance.get("intended_claim", ""),
        "source_id": config["source_id"],
        "source_agency": payload.get("source_agency", ""),
        "source_url": payload.get("source_url", ""),
        "effective_date": payload.get("effective_date"),
        "checked_at": payload.get("checked_at"),
        "public_claim_status": payload.get("public_claim_status", ""),
        "matched_features": payload.get("matched_features") or [],
        "limitations": payload.get("limitations") or [],
    }


def check_cgs_layers(
    lat: Optional[float],
    lon: Optional[float],
    dataset_ids: Optional[List[str]] = None,
) -> List[Dict]:
    selected = dataset_ids or list(CGS_DATASETS)
    if lat is None or lon is None:
        return []
    service = GeospatialEvidenceService(project_root=BASE_DIR)
    def check_one(dataset_id):
        try:
            evidence = service.check_point(dataset_id, float(lat), float(lon))
        except (DatasetRegistryError, TypeError, ValueError):
            return {
                "dataset_id": dataset_id,
                "map_layer_key": CGS_DATASETS[dataset_id]["map_layer_key"],
                "dataset_name": "",
                "checked": False,
                "data_available": False,
                "matched": None,
                "exposure": "data_unavailable",
                "result_label": "Official CGS layer unavailable — not checked.",
                "status_label": "Not checked",
                "plain_definition": CGS_DATASETS[dataset_id]["plain_definition"],
                "what_this_means": "The official map service was unavailable, so StayReady did not evaluate this location with this layer.",
                "why_it_matters": "",
                "what_this_does_not_mean": CGS_DATASETS[dataset_id]["does_not_mean"],
                "priority_band": "Not an address-level priority ranking",
                "ranking_score": None,
                "evidence_tier": "official_mapped_data",
                "claim_type": "regulatory_zone",
                "precision": "address_point",
                "source_summary": "",
                "source_id": CGS_DATASETS[dataset_id]["source_id"],
                "source_agency": "California Geological Survey",
                "source_url": "",
                "effective_date": None,
                "checked_at": None,
                "public_claim_status": "official_unavailable",
                "matched_features": [],
                "limitations": [
                        "Official CGS layer unavailable — not checked.",
                    "Missing or failed data does not lower exposure or establish a location-wide risk conclusion.",
                ],
            }
        return _cgs_public_evidence(dataset_id, evidence)

    with ThreadPoolExecutor(max_workers=min(4, len(selected))) as executor:
        futures = [executor.submit(check_one, dataset_id) for dataset_id in selected]
        return [future.result() for future in futures]


def _dam_inundation_public_evidence(result: Dict) -> Dict:
    checked = result.get("data_status") == "checked"
    matched = result.get("inside_inundation_boundary") if checked else None
    scenarios = result.get("matched_dam_scenarios") or []
    if not checked:
        exposure = "data_unavailable"
        result_label = "Official DWR/DSOD dam inundation layer unavailable — not checked."
        status_label = "Not checked"
        meaning = "The official map service was unavailable, so StayReady did not evaluate this location for published dam-failure inundation boundaries."
    elif matched:
        exposure = "mapped_match"
        result_label = "Inside one or more DSOD-approved hypothetical dam-failure inundation boundaries."
        status_label = "Mapped match found"
        meaning = (
            "The selected address point intersects these published planning scenarios: "
            + "; ".join(result.get("matched_dam_scenario_names") or [])
            + "."
        )
    else:
        exposure = "no_mapped_match"
        result_label = "No mapped dam-failure inundation boundary matched the selected point."
        status_label = "No mapped match found"
        meaning = "The selected point did not intersect a published boundary in the checked DWR/DSOD layer."
    return {
        "dataset_id": "dwr_dsod_dam_inundation_remote",
        "map_layer_key": "dam-inundation",
        "dataset_name": "DSOD Approved Dam Inundation Boundaries",
        "checked": checked,
        "data_available": checked,
        "matched": matched,
        "exposure": exposure,
        "result_label": result_label,
        "status_label": status_label,
        "plain_definition": "Dam-failure inundation boundaries show areas that could be flooded in a hypothetical failure scenario used for emergency planning.",
        "what_this_means": meaning,
        "why_it_matters": "This official mapped planning data helps residents review evacuation readiness and local emergency information.",
        "what_this_does_not_mean": "It is not a prediction that a dam will fail, a live warning, or an evacuation order.",
        "priority_band": "Mapped evidence" if matched else "Not an address-level priority ranking",
        "ranking_score": None,
        "evidence_tier": "official_mapped_data",
        "claim_type": "scenario",
        "precision": "address_point",
        "source_summary": "DWR/DSOD publishes approved hypothetical inundation boundaries for emergency action planning.",
        "source_id": "dwr_dsod_dam_inundation",
        "source_agency": result.get("source_agency", ""),
        "source_url": result.get("source_landing_url") or result.get("source_url", ""),
        "service_url": result.get("source_url", ""),
        "effective_date": result.get("effective_date"),
        "checked_at": result.get("checked_at"),
        "public_claim_status": result.get("public_claim_status", ""),
        "matched_features": scenarios,
        "limitations": result.get("limitations") or [],
    }


def clear_location_check_cache():
    with _LOCATION_CHECK_CACHE_LOCK:
        _LOCATION_CHECK_CACHE.clear()


def _collect_location_checks(lat: float, lon: float) -> Dict:
    # Coordinates are held only in short-lived process memory and rounded to
    # roughly meter scale. Full addresses are never part of this cache key.
    cache_key = (
        round(float(lat), 5),
        round(float(lon), 5),
        id(check_flood_layer),
        id(check_wildfire_layer),
        id(check_fault_layer),
        id(check_cgs_layers),
        id(check_dam_inundation),
    )
    now = time.monotonic()
    with _LOCATION_CHECK_CACHE_LOCK:
        cached = _LOCATION_CHECK_CACHE.get(cache_key)
        if cached and cached["expires_at"] > now:
            return deepcopy(cached["payload"])

    with ThreadPoolExecutor(max_workers=6) as executor:
        flood_future = executor.submit(check_flood_layer, lat, lon)
        wildfire_future = executor.submit(check_wildfire_layer, lat, lon)
        fault_future = executor.submit(check_fault_layer, lat, lon)
        earthquake_cgs_future = executor.submit(
            check_cgs_layers,
            lat,
            lon,
            [
                "cgs_alquist_priolo_remote",
                "cgs_liquefaction_remote",
                "cgs_earthquake_landslide_remote",
            ],
        )
        tsunami_cgs_future = executor.submit(
            check_cgs_layers,
            lat,
            lon,
            ["cgs_tsunami_hazard_area_remote"],
        )
        dam_inundation_future = executor.submit(check_dam_inundation, lat, lon)
        payload = {
            "flood": flood_future.result(),
            "wildfire": wildfire_future.result(),
            "fault": fault_future.result(),
            "earthquake_cgs": earthquake_cgs_future.result(),
            "tsunami_cgs": tsunami_cgs_future.result(),
            "dam_inundation": _dam_inundation_public_evidence(
                dam_inundation_future.result()
            ),
        }

    unavailable = any(
        check.get("data_status") == "data_unavailable"
        for check in (payload["flood"], payload["wildfire"], payload["fault"])
    ) or any(
        not evidence.get("data_available")
        for evidence in (
            payload["earthquake_cgs"]
            + payload["tsunami_cgs"]
            + [payload["dam_inundation"]]
        )
    )
    ttl = (
        _LOCATION_CHECK_FAILURE_TTL_SECONDS
        if unavailable
        else _LOCATION_CHECK_CACHE_TTL_SECONDS
    )
    with _LOCATION_CHECK_CACHE_LOCK:
        _LOCATION_CHECK_CACHE[cache_key] = {
            "expires_at": now + ttl,
            "payload": deepcopy(payload),
        }
    return payload


def _apply_cgs_evidence(result: HazardResult, checks: List[Dict]) -> HazardResult:
    if not checks:
        return result
    result.additional_geospatial_evidence = checks
    checked = [item for item in checks if item.get("checked")]
    matched = [item for item in checked if item.get("matched")]
    unavailable = [item for item in checks if not item.get("data_available")]
    result.scope = "address_level" if checked else result.scope
    result.basis = "gis_overlay" if checked else result.basis
    result.location_precision = "address_point" if checked else result.location_precision
    if checked:
        result.data_status = "checked" if matched else "not_in_layer"
        if matched:
            result.is_in_hazard_zone = True
            if result.match_type not in {"near_fault", "near"}:
                result.match_type = "inside"
            labels = " ".join(item["result_label"] for item in matched)
            result.why_shown = f"{result.why_shown} CGS address-point evidence: {labels}".strip()
        else:
            result.why_shown = (
                f"{result.why_shown} These CGS layers did not find an address-level match. "
                "This result applies only to those datasets and does not establish overall location risk."
            ).strip()
    elif unavailable:
        result.limitations.insert(0, "Official CGS layers were unavailable — not checked.")
    result.limitations = _dedupe_text(
        result.limitations
        + [limitation for item in checks for limitation in item.get("limitations", [])]
    )
    source_ids = [item.get("source_id") for item in checks if item.get("source_id")]
    result.sources = _source_payload(result.hazard_type, [
        *result.specialized_guidance.source_ids,
        *source_ids,
        "ready_recovery",
    ])
    return result


def _apply_dam_inundation_evidence(result: HazardResult, check: Dict) -> HazardResult:
    if not check:
        return result
    result.additional_geospatial_evidence = [
        *result.additional_geospatial_evidence,
        check,
    ]
    if check.get("checked"):
        result.scope = "address_level"
        result.basis = "gis_overlay"
        result.location_precision = "address_point"
        if check.get("matched"):
            result.data_status = "checked"
            result.is_in_hazard_zone = True
            result.match_type = "inside"
            scenario_names = [
                item.get("display_name")
                for item in check.get("matched_features") or []
                if item.get("display_name")
            ]
            result.matched_layers = [
                {
                    "name": name,
                    "source": check.get("source_agency"),
                }
                for name in scenario_names
            ]
            result.why_shown = (
                f"{result.why_shown} DWR/DSOD address-point evidence found "
                f"{len(scenario_names)} overlapping hypothetical inundation "
                f"{'scenario' if len(scenario_names) == 1 else 'scenarios'}."
            ).strip()
        else:
            result.data_status = "not_in_layer"
            result.is_in_hazard_zone = False
            result.match_type = "none"
            result.why_shown = (
                f"{result.why_shown} The checked DWR/DSOD layer did not find a mapped "
                "dam-failure inundation boundary at the selected point. This is not a safety determination."
            ).strip()
    else:
        result.data_status = "data_unavailable"
        result.is_in_hazard_zone = None
        result.match_type = "none"
        result.limitations.insert(
            0,
            "Official DWR/DSOD dam inundation layer unavailable — not checked.",
        )
    result.limitations = _dedupe_text(
        result.limitations + (check.get("limitations") or [])
    )
    result.sources = _source_payload(result.hazard_type, [
        *result.specialized_guidance.source_ids,
        "dwr_dsod_dam_inundation",
        "ready_recovery",
    ])
    return result


def _fema_layer_is_sfha(layer: Dict) -> bool:
    zone = str(layer.get("zone") or "").upper()
    sfha = str(layer.get("sfha") or "").upper()
    return sfha in {"T", "TRUE", "Y", "YES"} or zone.startswith(("A", "V"))


def _normalized_evidence_record(
    *,
    dataset_id: str,
    label: str,
    status: str,
    matched: Optional[bool],
    claim_type: str,
    source_agency: str,
    source_url: str,
    details: Optional[Dict] = None,
) -> Dict:
    return {
        "dataset_id": dataset_id,
        "label": label,
        "status": status,
        "matched": matched,
        "claim_type": claim_type,
        "source_agency": source_agency,
        "source_url": source_url,
        "details": details or {},
    }


def _build_normalized_evidence(result: HazardResult) -> List[Dict]:
    records = []
    if result.geospatial_evidence:
        evidence = result.geospatial_evidence
        records.append(_normalized_evidence_record(
            dataset_id=evidence.get("dataset_id", result.hazard_id),
            label=(evidence.get("provenance") or {}).get("dataset_name", result.label),
            status=result.data_status,
            matched=result.is_in_hazard_zone,
            claim_type=evidence.get("claim_type", result.claim_type),
            source_agency=evidence.get("source_agency", result.source_agency),
            source_url=evidence.get("source_url", result.source_url),
            details={
                "matched_layers": result.matched_layers,
                "public_claim_status": evidence.get("public_claim_status", ""),
            },
        ))
    if result.hazard_type == "wildfire" and not result.geospatial_evidence:
        records.append(_normalized_evidence_record(
            dataset_id="calfire_fhsz_local",
            label="CAL FIRE Fire Hazard Severity Zones",
            status=result.data_status,
            matched=result.is_in_hazard_zone,
            claim_type="hazard_zone",
            source_agency="California Department of Forestry and Fire Protection",
            source_url=result.source_url,
            details={"matched_layers": result.matched_layers},
        ))
    if result.hazard_type == "earthquake":
        records.append(_normalized_evidence_record(
            dataset_id="usgs_cgs_faults_local",
            label="Loaded mapped fault traces",
            status="proximity_context" if result.match_type == "near_fault" else "checked",
            matched=result.match_type == "near_fault",
            claim_type="proximity",
            source_agency="United States Geological Survey and contributing California Geological Survey records",
            source_url=get_source("usgs_faults").url,
            details={"matched_layers": result.matched_layers},
        ))
    for evidence in result.additional_geospatial_evidence:
        records.append(_normalized_evidence_record(
            dataset_id=evidence.get("dataset_id", ""),
            label=evidence.get("dataset_name", ""),
            status=evidence.get("exposure", "not_checked"),
            matched=evidence.get("matched"),
            claim_type=evidence.get("claim_type", ""),
            source_agency=evidence.get("source_agency", ""),
            source_url=evidence.get("source_url", ""),
            details={"matched_features": evidence.get("matched_features") or []},
        ))
    return records


def _classify_preparedness_priority(
    result: HazardResult,
    user_context: Optional[Dict] = None,
) -> HazardResult:
    user_context = user_context or {}
    mapped_match = bool(result.is_in_hazard_zone) or any(
        item.get("matched") for item in result.additional_geospatial_evidence
    )
    proximity = result.match_type in {"near", "near_fault"}
    unavailable = result.data_status in {"not_checked", "data_unavailable"}
    no_match = result.data_status == "not_in_layer"

    if mapped_match:
        result.hazard_exposure = "mapped_match"
    elif proximity:
        result.hazard_exposure = "proximity_context"
    elif unavailable:
        result.hazard_exposure = "not_checked"
    elif no_match:
        result.hazard_exposure = "no_mapped_match"
    else:
        result.hazard_exposure = "regional_context"

    local_plan_supported = bool(
        result.local_plan_match
        and result.local_plan_match.get("hazard_supported")
        and result.local_plan_match.get("review_status") in {"reviewed", "draft_reviewed"}
    )
    if result.hazard_type == "earthquake":
        result.hazard_importance = "major_regional"
    elif local_plan_supported or result.specialized_guidance.guidance_source_status == "local_reviewed":
        result.hazard_importance = "local_context"
    else:
        result.hazard_importance = "general"

    if mapped_match or proximity:
        result.action_priority = "start_here"
    elif result.hazard_importance in {"major_regional", "local_context"}:
        result.action_priority = "important"
    else:
        result.action_priority = "keep_in_plan"

    if result.location_precision == "address_point":
        if result.public_claim_status == "official_verified":
            result.source_confidence = "official_verified_address"
        elif result.data_status == "data_unavailable":
            result.source_confidence = "official_source_unavailable"
        else:
            result.source_confidence = "official_provisional_address"
    elif result.scope == "jurisdiction_level":
        result.source_confidence = "reviewed_area_context"
    else:
        result.source_confidence = "general_guidance"

    reasons = []
    if mapped_match:
        reasons.append("An official mapped layer found an address-level match.")
    elif proximity:
        reasons.append("The address is near a loaded mapped fault trace; this is proximity context, not zone membership.")
    elif unavailable:
        reasons.append("An official layer was unavailable, so missing data was not used to lower this preparedness priority.")
    elif no_match:
        reasons.append("The checked layer did not find an address-level match; regional or indirect impacts may still apply.")
    if result.hazard_importance == "major_regional":
        reasons.append("Earthquake preparedness remains important throughout Alameda County, even without a single zone match.")
    elif result.hazard_importance == "local_context":
        reasons.append("Reviewed local planning context supports keeping this hazard prominent.")

    household_tags = user_context.get("household_tags") or []
    if household_tags:
        reasons.append("Household needs were used to prioritize actions, not to change mapped exposure.")
        if result.action_priority == "keep_in_plan":
            result.action_priority = "important"

    result.priority_reasons = _dedupe_text(reasons)
    result.normalized_mapped_evidence = _build_normalized_evidence(result)
    return result


def _priority_sort_key(item: HazardResult):
    exposure_rank = {
        "mapped_match": 5,
        "proximity_context": 4,
        "regional_context": 1,
        "no_mapped_match": 1,
        "not_checked": 1,
    }
    importance_rank = {"major_regional": 3, "local_context": 2, "general": 1}
    action_rank = {"start_here": 3, "important": 2, "keep_in_plan": 1}
    stable_hazard_rank = {"earthquake": 3, "flood": 2, "wildfire": 1, "tsunami-seiche": 0}
    return (
        exposure_rank.get(item.hazard_exposure, 0),
        action_rank.get(item.action_priority, 0),
        importance_rank.get(item.hazard_importance, 0),
        stable_hazard_rank.get(item.hazard_id, 0),
    )


def _fire_layer_is_hazard_zone(layer: Dict) -> bool:
    hazard_class = str(layer.get("hazard_class") or "").strip().lower()
    return hazard_class in {"moderate", "high", "very high"}


def _source_payload(hazard_type: str, extra_source_ids: Optional[List[str]] = None):
    sources = []
    existing = set()
    for source_id in extra_source_ids or []:
        if source_id and source_id not in existing:
            sources.append(get_source(source_id))
            existing.add(source_id)
    for source in get_sources_for_hazard(hazard_type):
        if source.source_id not in existing:
            sources.append(source)
            existing.add(source.source_id)
    return sources


def _legacy_score(zip_snapshot: Dict, hazard_type: str) -> Optional[float]:
    item = (zip_snapshot or {}).get(hazard_type) or {}
    score = item.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _zip_explanation(zip_snapshot: Dict, hazard_type: str) -> str:
    return ((zip_snapshot or {}).get(hazard_type) or {}).get("explanation", "")


def _local_plan_match(location_result, hazard_type: str) -> Optional[Dict]:
    plan = get_local_plan_for_city(location_result.city)
    if not plan:
        return None
    hazard_key = (hazard_type or "").strip().lower()
    plan_hazards = {item.lower() for item in plan.get("hazards", [])}
    return {
        "jurisdiction_id": plan.get("jurisdiction_id"),
        "name": plan.get("name"),
        "plan_name": plan.get("plan_name"),
        "plan_group": plan.get("plan_group"),
        "review_status": plan.get("review_status"),
        "url": plan.get("url"),
        "hazard_supported": hazard_key in plan_hazards or "all" in plan_hazards,
        "notes": plan.get("notes", ""),
    }


def _location_specific_context(location_result, hazard_type: str) -> List[str]:
    # Automatic LHMP area claims are handled only by validated location facts.
    # Keeping this empty prevents legacy coordinate heuristics from bypassing
    # the reviewed alias and bounded-geography matching rules.
    return []


def _specialized_guidance(location_result, hazard_type: str, user_context: Optional[Dict] = None) -> SpecializedGuidance:
    user_context = user_context or {}
    city_chunks = get_city_chunks(location_result.city, hazard_type)
    guidance_chunks = get_resident_guidance(location_result.city, hazard_type)
    plan = get_local_plan_for_city(location_result.city)
    location_specific_context = _location_specific_context(location_result, hazard_type)
    grouped_guidance = {
        "hazard_priority": [],
        "local_context": [],
        "before": [],
        "during": [],
        "after": [],
        "recovery": [],
        "limitations": [],
    }
    guidance_source_status = "county_fallback"
    source_ids = list(dict.fromkeys(chunk.get("source_id") for chunk in city_chunks if chunk.get("source_id")))
    if location_specific_context and (location_result.city or "").strip().lower() == "berkeley":
        source_ids.append("berkeley_lhmp")
        if hazard_type == "wildfire":
            source_ids.append("berkeley_fire_evacuation")

    for chunk in guidance_chunks:
        try:
            item = ResidentGuidanceItem(**chunk)
        except Exception:
            continue
        grouped_guidance.setdefault(item.resident_phase, []).append(item)
        if item.recovery_question and item.resident_phase != "recovery":
            grouped_guidance["recovery"].append(item)
        if item.source_id not in source_ids:
            source_ids.append(item.source_id)
        if (
            location_result.city
            and item.jurisdiction.strip().lower() == location_result.city.strip().lower()
            and item.review_status in {"reviewed", "draft_reviewed"}
        ):
            guidance_source_status = "local_reviewed"

    city_context = [
        item.plain_language
        for phase in ("hazard_priority", "local_context", "limitations")
        for item in grouped_guidance.get(phase, [])
    ]
    if plan and plan.get("review_status") == "needs_source_review" and guidance_source_status != "local_reviewed":
        guidance_source_status = "needs_source_review"
        city_context.insert(
            0,
            f"{location_result.city} is represented in the Alameda County registry, but its city-specific source still needs review. Countywide guidance is shown instead.",
        )
    if not city_context:
        city_context = [chunk.get("text", "") for chunk in city_chunks if chunk.get("text")]
    household_factors = []
    access_functional_needs = []
    special_needs = str(user_context.get("special_needs") or "").strip()

    if special_needs:
        access_functional_needs.append(f"Reported household need: {special_needs}.")

    if not city_context and location_result.city:
        if plan and plan.get("review_status") == "needs_source_review":
            guidance_source_status = "needs_source_review"
            city_context.append(
                f"{location_result.city} is represented in the Alameda County registry, but its city-specific source still needs review. Countywide guidance is shown instead."
            )
        else:
            city_context.append(
                f"No reviewed {location_result.city} resident guidance chunk is available for this hazard yet, so city-specific claims remain limited."
            )

    return SpecializedGuidance(
        location_specific_context=location_specific_context,
        city_context=city_context,
        household_factors=household_factors,
        access_functional_needs=access_functional_needs,
        recovery_needs=[],
        resident_guidance=grouped_guidance,
        guidance_source_status=guidance_source_status,
        source_ids=source_ids,
    )


def _jurisdiction_result(hazard: Dict, location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = _hazard_type(hazard)
    label = _hazard_label(hazard, hazard_type)
    city = location_result.city or "Alameda County"
    score = _legacy_score(zip_snapshot, hazard_type)
    limitations = [
        "This result uses official planning/source context for the jurisdiction, not a point-in-polygon address check.",
        "Jurisdiction and ZIP context do not determine exposure for an individual address.",
    ]
    if score is not None:
        limitations.append("ZIP score is included only as supporting fallback context, not as the primary location system.")

    local_plan_match = _local_plan_match(location_result, hazard_type)
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)
    if local_plan_match and local_plan_match.get("hazard_supported") and local_plan_match.get("review_status") in {"reviewed", "draft_reviewed"}:
        limitations.append("City plan context is included as reviewed planning context, not as address-level GIS membership.")

    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=label,
        scope="jurisdiction_level",
        basis="official_registry",
        location_precision="city" if location_result.city else "county",
        data_status="fallback_used",
        exposure_level="unknown",
        is_in_hazard_zone=None,
        match_type="jurisdiction_match",
        matched_layers=[],
        source_url=(get_sources_for_hazard(hazard_type)[0].url if get_sources_for_hazard(hazard_type) else ""),
        confidence="mixed_support",
        review_status="draft",
        why_shown=(
            f"{label} is shown because {city} and Alameda County source context "
            "identify it as relevant for local preparedness planning."
        ),
        limitations=limitations,
        recommended_actions=_actions_for(location_result, hazard_type, "before", "today", "this_week"),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(
            hazard_type,
            specialized_guidance.source_ids + ["alameda_county_emergency", "ready_recovery"],
        ),
        local_plan_match=local_plan_match,
        specialized_guidance=specialized_guidance,
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def _zip_result(hazard: Dict, location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = _hazard_type(hazard)
    label = _hazard_label(hazard, hazard_type)
    score = _legacy_score(zip_snapshot, hazard_type)
    explanation = _zip_explanation(zip_snapshot, hazard_type)
    zip_code = location_result.zip_code or "the selected ZIP"
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)

    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=label,
        scope="zip_estimate",
        basis="zip_csv_heuristic",
        location_precision="zip",
        data_status="fallback_used",
        exposure_level="unknown",
        is_in_hazard_zone=None,
        match_type="zip_match",
        matched_layers=[],
        source_url="",
        confidence="mixed_support",
        review_status="draft",
        why_shown=(
            f"{label} is shown from ZIP-level fallback context for {zip_code}. "
            f"{explanation}".strip()
        ),
        limitations=[
            "The numeric ZIP value is retained only as an internal fallback ranking signal; it is not an exposure or safety determination.",
            "ZIP context does not prove whether an individual address is inside or outside a mapped hazard area.",
            "Other hazards, indirect impacts, evacuation issues, smoke, infrastructure outages, and unmapped conditions may still apply.",
            "Use official maps and emergency instructions for final decisions.",
        ],
        recommended_actions=_actions_for(location_result, hazard_type, "before", "today", "this_week"),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(
            hazard_type,
            specialized_guidance.source_ids + ["alameda_county_emergency", "ready_recovery"],
        ),
        local_plan_match=_local_plan_match(location_result, hazard_type),
        specialized_guidance=specialized_guidance,
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def _county_result(hazard: Dict) -> HazardResult:
    hazard_type = _hazard_type(hazard)
    label = _hazard_label(hazard, hazard_type)
    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=label,
        scope="county_fallback",
        basis="county_guidance",
        location_precision="county",
        data_status="fallback_used",
        exposure_level="unknown",
        is_in_hazard_zone=None,
        match_type="fallback",
        matched_layers=[],
        source_url="",
        confidence="needs_review",
        review_status="draft",
        why_shown=f"{label} is shown as general Alameda County preparedness guidance.",
        limitations=[
            "This is countywide guidance only. No address, city, or ZIP-specific hazard check is available for this result.",
            "Countywide guidance does not determine exposure or safety for an individual location.",
        ],
        recommended_actions=select_actions(
            hazards=[hazard_type],
            time_buckets=["before", "today", "this_week"],
            county="Alameda County",
            trigger_types=["general", "hazard_result", "location"],
            limit=4,
        ),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(hazard_type, ["alameda_county_emergency", "ready_recovery"]),
        legacy_priority_score=hazard.get("priority_score"),
    )


def _flood_address_result(hazard: Dict, location_result, flood_check: Dict, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = "flood"
    score = _legacy_score(zip_snapshot, hazard_type)
    layers = flood_check.get("layers") or []
    hazard_layers = flood_check.get("sfha_layers") or [
        layer for layer in layers if _fema_layer_is_sfha(layer)
    ]
    polygon_match = bool(flood_check.get("polygon_match", layers))
    inside = bool(flood_check.get("sfha_match", hazard_layers))
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)
    geospatial_evidence = flood_check.get("geospatial_evidence") or {}
    status = "checked" if inside else "not_in_layer"
    exposure = "high" if inside else "unknown"
    limitations = [
        "This checks the saved address point against StayReady's provisional local FEMA flood polygon snapshot.",
        "Point checks are not parcel determinations and should be confirmed with official FEMA and local flood resources.",
    ]
    if not inside:
        limitations.append(
            "No matching Special Flood Hazard Area exposure was found for the point in the checked snapshot. "
            "This does not establish overall flood or water-related impact risk for the location."
        )
    limitations = _dedupe_text(
        limitations + list(geospatial_evidence.get("limitations") or [])
    )

    zone_text = f" and matched {hazard_layers[0]['name']}" if hazard_layers else ""
    context_text = ""
    if polygon_match and layers and not inside:
        zone = str(layers[0].get("zone") or "other")
        context_text = (
            f" The address matched FEMA Zone {zone}, a mapped category that is not treated here "
            "as Special Flood Hazard Area membership."
        )
    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=_hazard_label(hazard, "Flood"),
        scope="address_level",
        basis="gis_overlay",
        location_precision="address_point",
        data_status=status,
        exposure_level=exposure,
        is_in_hazard_zone=inside,
        match_type="inside" if inside else "none",
        matched_layers=layers,
        geospatial_evidence=geospatial_evidence or None,
        claim_type=geospatial_evidence.get("claim_type", ""),
        checked_at=geospatial_evidence.get("checked_at", ""),
        effective_date=geospatial_evidence.get("effective_date") or "",
        public_claim_status=geospatial_evidence.get("public_claim_status", ""),
        source_agency=geospatial_evidence.get("source_agency", ""),
        source_url=geospatial_evidence.get("source_url") or get_source("fema_nfhl").url,
        confidence="mixed_support",
        review_status="draft_reviewed",
        why_shown=(
            f"Flood is shown because the saved address point was checked against StayReady's provisional local FEMA snapshot{zone_text}."
            if inside else
            f"Flood is shown because the FEMA layer did not find an address-level Special Flood Hazard Area match. "
            f"This is informational, not a safety determination.{context_text}"
        ),
        limitations=limitations,
        recommended_actions=_actions_for(location_result, hazard_type, "before", "today", "this_week"),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(
            hazard_type,
            specialized_guidance.source_ids + ["fema_nfhl", "ready_floods", "ready_recovery"],
        ),
        local_plan_match=_local_plan_match(location_result, hazard_type),
        specialized_guidance=specialized_guidance,
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def _wildfire_address_result(hazard: Dict, location_result, wildfire_check: Dict, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = "wildfire"
    score = _legacy_score(zip_snapshot, hazard_type)
    layers = wildfire_check.get("layers") or []
    hazard_layers = [layer for layer in layers if _fire_layer_is_hazard_zone(layer)]
    inside = bool(hazard_layers)
    hazard_class = (hazard_layers[0].get("hazard_class", "") if hazard_layers else "").lower()
    exposure = "high" if "very high" in hazard_class or "high" in hazard_class else "medium" if inside else "unknown"
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)
    geospatial_evidence = wildfire_check.get("geospatial_evidence") or {}
    limitations = [
        "This checks the saved address point against StayReady's provisional local CAL FIRE fire hazard polygon snapshot.",
        "Fire hazard zones do not capture all wildfire smoke, evacuation, power shutoff, or ember exposure.",
    ]
    if not inside:
        limitations.append(
            "No matching moderate, high, or very-high Fire Hazard Severity Zone exposure was found for the point in the checked snapshot. "
            "This does not establish overall wildfire, smoke, ember, evacuation, or regional fire-impact risk."
        )
    limitations = _dedupe_text(
        limitations + list(geospatial_evidence.get("limitations") or [])
    )

    zone_text = f" and matched {hazard_layers[0]['name']}" if hazard_layers else ""
    context_text = f" The address matched {layers[0]['name']}, which is not treated here as a moderate/high/very-high fire hazard zone." if layers and not inside else ""
    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=_hazard_label(hazard, "Wildfire"),
        scope="address_level",
        basis="gis_overlay",
        location_precision="address_point",
        data_status="checked" if inside else "not_in_layer",
        exposure_level=exposure,
        is_in_hazard_zone=inside,
        match_type="inside" if inside else "none",
        matched_layers=layers,
        geospatial_evidence=geospatial_evidence or None,
        claim_type=geospatial_evidence.get("claim_type", ""),
        checked_at=geospatial_evidence.get("checked_at", ""),
        effective_date=geospatial_evidence.get("effective_date") or "",
        public_claim_status=geospatial_evidence.get("public_claim_status", ""),
        source_agency=geospatial_evidence.get("source_agency", ""),
        source_url=geospatial_evidence.get("source_url") or get_source("calfire_fhsz").url,
        confidence="mixed_support",
        review_status="draft_reviewed",
        why_shown=(
            f"Wildfire is shown because the saved address point was checked against StayReady's provisional local CAL FIRE snapshot{zone_text}."
            if inside else
            f"Wildfire is shown because no matching moderate, high, or very-high Fire Hazard Severity Zone exposure was found for the address point in StayReady's provisional local CAL FIRE snapshot. This is informational, not a safety determination.{context_text}"
        ),
        limitations=limitations,
        recommended_actions=_actions_for(location_result, hazard_type, "before", "today", "this_week"),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(
            hazard_type,
            specialized_guidance.source_ids + ["calfire_fhsz", "ready_kit", "ready_plan", "ready_recovery"],
        ),
        local_plan_match=_local_plan_match(location_result, hazard_type),
        specialized_guidance=specialized_guidance,
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def _earthquake_address_result(hazard: Dict, location_result, fault_check: Dict, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = "earthquake"
    score = _legacy_score(zip_snapshot, hazard_type)
    layers = fault_check.get("layers") or []
    nearest = fault_check.get("nearest_layer") or {}
    near = bool(fault_check.get("near"))
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)
    limitations = [
        "This checks address proximity to loaded mapped fault lines only; it does not check liquefaction, shaking intensity, landslide susceptibility, building retrofit status, or parcel-level seismic risk.",
        "Earthquake risk can still be significant even when the address is not near the nearest loaded fault trace.",
    ]
    if nearest and not near:
        limitations.append(f"The nearest loaded mapped fault line is {nearest.get('distance_km')} km away: {nearest.get('name')}.")

    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=_hazard_label(hazard, "Earthquake"),
        scope="address_level",
        basis="gis_overlay",
        location_precision="address_point",
        data_status="checked",
        exposure_level="unknown",
        is_in_hazard_zone=None,
        match_type="near_fault" if near else "fault_proximity_context",
        matched_layers=layers,
        source_url=get_source("usgs_faults").url,
        confidence="mixed_support",
        review_status="draft_reviewed",
        why_shown=(
            f"Earthquake is shown because the saved address point is within about 2 km of a loaded mapped fault line: {layers[0].get('name')}. This is proximity context, not hazard-zone membership."
            if near and layers else
            "Earthquake is shown because fault proximity was checked for the saved address point. No loaded fault trace was found within about 2 km, but this is not a hazard-zone clearance and citywide seismic risk still requires preparedness."
        ),
        limitations=limitations,
        recommended_actions=_actions_for(location_result, hazard_type, "before", "today", "this_week"),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(
            hazard_type,
            specialized_guidance.source_ids + ["usgs_faults", "ready_earthquakes", "ready_recovery"],
        ),
        local_plan_match=_local_plan_match(location_result, hazard_type),
        specialized_guidance=specialized_guidance,
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def build_hazard_results(hazards: List[Dict], location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> List[HazardResult]:
    results = []
    flood_check = _not_checked_result()
    wildfire_check = _not_checked_result()
    fault_check = _not_checked_result()
    earthquake_cgs_checks = []
    tsunami_cgs_checks = []
    dam_inundation_check = {}
    if location_result.lat is not None and location_result.lon is not None and location_result.formatted_address:
        lat, lon = location_result.lat, location_result.lon
        checks = _collect_location_checks(lat, lon)
        flood_check = checks["flood"]
        wildfire_check = checks["wildfire"]
        fault_check = checks["fault"]
        earthquake_cgs_checks = checks["earthquake_cgs"]
        tsunami_cgs_checks = checks["tsunami_cgs"]
        dam_inundation_check = checks.get("dam_inundation") or {}

    for raw_hazard in hazards:
        hazard = deepcopy(raw_hazard)
        hazard_type = _hazard_type(hazard)

        if hazard_type == "flood" and flood_check.get("checked"):
            result = _flood_address_result(hazard, location_result, flood_check, zip_snapshot, user_context)
        elif hazard_type == "wildfire" and wildfire_check.get("checked"):
            result = _wildfire_address_result(hazard, location_result, wildfire_check, zip_snapshot, user_context)
        elif hazard_type == "earthquake" and fault_check.get("checked"):
            result = _earthquake_address_result(hazard, location_result, fault_check, zip_snapshot, user_context)
        elif location_result.city or location_result.county:
            result = _jurisdiction_result(hazard, location_result, zip_snapshot, user_context)
            check = {
                "flood": flood_check,
                "wildfire": wildfire_check,
                "earthquake": fault_check,
            }.get(hazard_type)
            if check and location_result.formatted_address:
                result.data_status = check.get("data_status", "not_checked")
                unavailable = result.data_status == "data_unavailable"
                if unavailable:
                    result.exposure_level = "unknown"
                    result.is_in_hazard_zone = None
                    result.match_type = "none"
                    result.confidence = "needs_review"
                result.limitations.insert(
                    0,
                    "Official layer unavailable — not checked."
                    if unavailable else
                    "Address-level GIS membership has not been checked for this hazard yet.",
                )
                result.why_shown = (
                    f"{result.label} is shown from jurisdiction-level source context. "
                    + (
                        "The official GIS layer was unavailable, so no address-level check was completed."
                        if unavailable else
                        "The app has not completed an address-level GIS check for this hazard yet."
                    )
                )
        elif location_result.zip_code and zip_snapshot:
            result = _zip_result(hazard, location_result, zip_snapshot, user_context)
        else:
            result = _county_result(hazard)

        cgs_hazard_type = CGS_HAZARD_ALIASES.get(hazard_type)
        if cgs_hazard_type == "earthquake":
            result = _apply_cgs_evidence(result, earthquake_cgs_checks)
        elif cgs_hazard_type == "tsunami":
            result = _apply_cgs_evidence(result, tsunami_cgs_checks)
        if hazard_type in {"dam_failure", "dam-failure"}:
            result = _apply_dam_inundation_evidence(result, dam_inundation_check)
        results.append(_classify_preparedness_priority(result, user_context))

    existing_hazard_ids = {item.hazard_id for item in results}
    tsunami_match = any(item.get("matched") for item in tsunami_cgs_checks)
    tsunami_plan = _local_plan_match(location_result, "tsunami")
    reviewed_tsunami_context = bool(
        tsunami_plan
        and tsunami_plan.get("hazard_supported")
        and tsunami_plan.get("review_status") in {"reviewed", "draft_reviewed"}
    )
    if "tsunami-seiche" not in existing_hazard_ids and (tsunami_match or reviewed_tsunami_context):
        tsunami_hazard = {
            "name": "Tsunami",
            "label": "Tsunami",
            "slug": "tsunami-seiche",
            "priority_score": 0,
        }
        if location_result.city or location_result.county:
            tsunami_result = _jurisdiction_result(
                tsunami_hazard,
                location_result,
                zip_snapshot,
                user_context,
            )
        else:
            tsunami_result = _county_result(tsunami_hazard)
        tsunami_result = _apply_cgs_evidence(tsunami_result, tsunami_cgs_checks)
        results.append(_classify_preparedness_priority(tsunami_result, user_context))

    return sorted(results, key=_priority_sort_key, reverse=True)


def merge_structured_result(hazard: Dict, result: HazardResult) -> Dict:
    merged = deepcopy(hazard)
    payload = result.model_dump(mode="json")
    local_summary = _local_summary(result)
    merged["structured_result"] = payload
    merged["scope_label"] = display_scope(result.scope)
    merged["data_status_label"] = display_data_status(result.data_status)
    merged["evidence_status_label"] = display_data_status(result.data_status)
    merged["priority_label"] = {
        "start_here": "Start here",
        "important": "Important preparedness priority",
        "keep_in_plan": "Keep in your plan",
    }.get(result.action_priority, "Regional preparedness priority")
    merged["basis_label"] = result.basis.replace("_", " ").title()
    merged["location_precision_label"] = result.location_precision.replace("_", " ").title()
    merged["exposure_level"] = result.exposure_level.title()
    merged["risk_level"] = result.exposure_level.title()
    merged["priority_score"] = {
        "high": 8,
        "medium": 5,
        "low": 2,
        "unknown": 0,
    }.get(result.exposure_level, 0)
    merged["summary"] = local_summary
    merged["what_this_means_for_you"] = local_summary
    merged["personalized_what_this_means_for_you"] = local_summary
    merged["what_could_realistically_happen"] = local_summary
    merged["real_world_impact"] = _local_impact(result)
    merged["priority_reason"] = result.why_shown
    merged["hazard_exposure"] = result.hazard_exposure
    merged["hazard_importance"] = result.hazard_importance
    merged["action_priority"] = result.action_priority
    merged["source_confidence"] = result.source_confidence
    merged["priority_reasons"] = result.priority_reasons
    merged["normalized_mapped_evidence"] = result.normalized_mapped_evidence
    merged["action_steps"] = [item.model_dump(mode="json") for item in result.recommended_actions]
    merged["top_risks"] = _local_top_risks(result)
    merged["locations"] = _local_locations(result)
    merged["at_risk_groups"] = _dedupe_text(
        result.specialized_guidance.household_factors + result.specialized_guidance.access_functional_needs
    )
    merged["historical_examples"] = []
    merged["key_stats"] = _dedupe_text([
        f"Scope: {display_scope(result.scope)}",
        f"Data status: {display_data_status(result.data_status)}",
        f"Precision: {result.location_precision.replace('_', ' ')}",
        f"Source status: {result.specialized_guidance.guidance_source_status.replace('_', ' ')}",
    ])
    merged["why_shown"] = result.why_shown
    merged["limitations"] = result.limitations
    merged["recommended_actions"] = [item.model_dump(mode="json") for item in result.recommended_actions]
    merged["recovery_questions"] = [item.model_dump(mode="json") for item in result.recovery_questions]
    merged["sources"] = [item.model_dump() for item in result.sources]
    merged["local_plan_match"] = result.local_plan_match
    merged["specialized_guidance"] = result.specialized_guidance.model_dump()
    merged["matched_layers"] = result.matched_layers
    merged["geospatial_evidence"] = result.geospatial_evidence
    merged["additional_geospatial_evidence"] = result.additional_geospatial_evidence
    merged["claim_type"] = result.claim_type
    merged["checked_at"] = result.checked_at
    merged["effective_date"] = result.effective_date
    merged["public_claim_status"] = result.public_claim_status
    merged["source_agency"] = result.source_agency
    merged["is_in_hazard_zone"] = result.is_in_hazard_zone
    merged["match_type"] = result.match_type
    merged["data_status"] = result.data_status
    merged["scope"] = result.scope
    merged["basis"] = result.basis
    return merged
