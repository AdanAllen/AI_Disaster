import json
import math
import os
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional

from shapely.geometry import Point, shape
from shapely.ops import nearest_points

from pydantic_models import HazardResult, PreparednessAction, RecoveryQuestion, ResidentGuidanceItem, SpecializedGuidance
from source_registry import get_city_chunks, get_local_plan_for_city, get_resident_guidance, get_source, get_sources_for_hazard


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RECOVERY_QUESTIONS = [
    RecoveryQuestion(
        id="documents",
        question="Where are copies of your IDs, insurance papers, lease or mortgage records, and medical documents?",
        source_id="ready_recovery",
    ),
    RecoveryQuestion(
        id="housing",
        question="Where could your household stay for the first week if your home is unsafe?",
        source_id="ready_recovery",
    ),
    RecoveryQuestion(
        id="medical",
        question="How would you replace medications, medical devices, or backup power if services were disrupted?",
        source_id="ready_recovery",
    ),
    RecoveryQuestion(
        id="family-continuity",
        question="How will family members, pets, children, school, and work needs be handled during recovery?",
        source_id="ready_recovery",
    ),
    RecoveryQuestion(
        id="transportation",
        question="If your normal vehicle, transit route, or rideshare option is unavailable, what is your backup transportation plan?",
        source_id="ready_recovery",
    ),
    RecoveryQuestion(
        id="financial-recovery",
        question="How would you cover deductibles, temporary supplies, lost work time, or urgent repairs during the first month of recovery?",
        source_id="ready_recovery",
    ),
]

HAZARD_RECOVERY_NEEDS = {
    "wildfire": [
        "Document smoke damage and evacuation expenses.",
        "Plan for temporary housing if air quality, utilities, or access roads make home unsafe.",
        "Keep medication, pet, child, school, and work continuity plans ready for rapid evacuation.",
    ],
    "flood": [
        "Photograph damage before cleanup when it is safe.",
        "Keep insurance, lease, title, and repair contact information accessible above floor level.",
        "Plan where to stay if utilities, mold, or structural damage make the home unsafe.",
    ],
    "earthquake": [
        "Prepare for extended utility outages and delayed building inspections.",
        "Know how to access medications, mobility devices, and backup power if roads or elevators are disrupted.",
        "Plan for school, work, child, pet, and family reunification if communication systems are limited.",
    ],
}

DEFAULT_ACTIONS = {
    "wildfire": [
        PreparednessAction(id="alerts", label="Turn on official local emergency alerts.", source_id="alameda_county_emergency"),
        PreparednessAction(id="go-bag", label="Pack a go-bag with medications, chargers, documents, and water.", source_id="ready_kit"),
        PreparednessAction(id="routes", label="Identify two ways to leave your neighborhood before smoke or road closures arrive.", source_id="ready_plan"),
    ],
    "flood": [
        PreparednessAction(id="higher-ground", label="Know the fastest route to higher ground.", source_id="ready_floods"),
        PreparednessAction(id="documents", label="Store key documents and valuables above floor level.", source_id="ready_floods"),
        PreparednessAction(id="avoid-water", label="Do not walk or drive through floodwater.", source_id="ready_floods"),
    ],
    "earthquake": [
        PreparednessAction(id="secure-items", label="Secure heavy furniture, shelves, and televisions.", source_id="ready_earthquakes"),
        PreparednessAction(id="safe-spots", label="Pick safe spots in each room away from windows.", source_id="ready_earthquakes"),
        PreparednessAction(id="drill", label="Practice drop, cover, and hold on.", source_id="ready_earthquakes"),
    ],
}


def normalize_exposure(level: str) -> str:
    value = (level or "").strip().lower()
    if value in {"high", "very high"}:
        return "high"
    if value in {"medium", "moderate"}:
        return "medium"
    if value == "low":
        return "low"
    return "unknown"


def exposure_from_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


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
        "checked": "Checked",
        "not_checked": "Not checked yet",
        "data_unavailable": "Official layer unavailable — not checked",
        "not_in_layer": "Checked, not in layer",
        "fallback_used": "Fallback used",
        "needs_review": "Needs review",
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


def _resident_guidance_items(result: HazardResult, *phases: str) -> List:
    items = []
    guidance = result.specialized_guidance.resident_guidance or {}
    for phase in phases:
        items.extend(guidance.get(phase, []))
    return items


def _resident_actions(result: HazardResult) -> List[str]:
    actions = []
    for item in _resident_guidance_items(result, "before", "during", "after", "limitations"):
        actions.append(item.recommended_action or item.plain_language)
    actions.extend(action.label for action in result.recommended_actions)
    return _dedupe_text(actions)


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
    return "Recovery planning should account for documents, insurance, housing, medications, pets, transportation, and school or work continuity."


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

    point = Point(float(lon), float(lat))
    loaded = load_geojson("FldHaz.geojson")
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
            layers.append({
                "layer_id": props.get("FLD_AR_ID") or props.get("SOURCE_CIT") or "fema_flood_layer",
                "name": f"FEMA flood zone {props.get('FLD_ZONE', 'unknown')}",
                "zone": props.get("FLD_ZONE", "Unknown"),
                "source_citation": props.get("SOURCE_CIT", ""),
                "sfha": props.get("SFHA_TF", ""),
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


def _fema_layer_is_sfha(layer: Dict) -> bool:
    zone = str(layer.get("zone") or "").upper()
    sfha = str(layer.get("sfha") or "").upper()
    return sfha in {"T", "TRUE", "Y", "YES"} or zone.startswith(("A", "V"))


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

    household = str(user_context.get("household") or "").strip()
    preparedness = str(user_context.get("preparedness") or "").strip()
    special_needs = str(user_context.get("special_needs") or "").strip()

    if household:
        household_factors.append(f"Plan supplies, evacuation, and recovery around a household size of {household}.")
    if preparedness and preparedness not in {"", "Unknown"}:
        household_factors.append(f"Current preparedness was marked as {preparedness}; keep next steps practical and staged.")
    if special_needs:
        access_functional_needs.append(f"Account for reported medical or access needs: {special_needs}.")
    else:
        access_functional_needs.append("Check whether anyone depends on medications, mobility equipment, powered medical devices, caregivers, or accessible transportation.")

    recovery_needs = HAZARD_RECOVERY_NEEDS.get(hazard_type, [
        "Keep documents, insurance, housing, medications, transportation, and family continuity plans accessible.",
    ])
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
        recovery_needs=recovery_needs,
        resident_guidance=grouped_guidance,
        guidance_source_status=guidance_source_status,
        source_ids=source_ids,
    )


def _jurisdiction_result(hazard: Dict, location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = _hazard_type(hazard)
    label = _hazard_label(hazard, hazard_type)
    city = location_result.city or "Alameda County"
    score = _legacy_score(zip_snapshot, hazard_type)
    exposure = exposure_from_score(score) if score is not None else normalize_exposure(hazard.get("risk_level"))
    limitations = [
        "This result uses official planning/source context for the jurisdiction, not a point-in-polygon address check.",
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
        exposure_level=exposure,
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
        recommended_actions=DEFAULT_ACTIONS.get(hazard_type, DEFAULT_ACTIONS["earthquake"]),
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
    exposure = exposure_from_score(score)
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
        exposure_level=exposure,
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
            "ZIP estimates are fallback guidance and do not prove whether an individual address is inside a hazard zone.",
            "Use official maps and emergency instructions for final decisions.",
        ],
        recommended_actions=DEFAULT_ACTIONS.get(hazard_type, DEFAULT_ACTIONS["earthquake"]),
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
        exposure_level=normalize_exposure(hazard.get("risk_level")),
        is_in_hazard_zone=None,
        match_type="fallback",
        matched_layers=[],
        source_url="",
        confidence="needs_review",
        review_status="draft",
        why_shown=f"{label} is shown as general Alameda County preparedness guidance.",
        limitations=[
            "This is countywide guidance only. No address, city, or ZIP-specific hazard check is available for this result.",
        ],
        recommended_actions=DEFAULT_ACTIONS.get(hazard_type, DEFAULT_ACTIONS["earthquake"]),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(hazard_type, ["alameda_county_emergency", "ready_recovery"]),
        legacy_priority_score=hazard.get("priority_score"),
    )


def _flood_address_result(hazard: Dict, location_result, flood_check: Dict, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = "flood"
    score = _legacy_score(zip_snapshot, hazard_type)
    layers = flood_check.get("layers") or []
    hazard_layers = [layer for layer in layers if _fema_layer_is_sfha(layer)]
    inside = bool(hazard_layers)
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)
    status = "checked" if inside else "not_in_layer"
    exposure = "high" if inside else "low"
    limitations = [
        "This checks the saved address point against the current FEMA flood polygon layer in the app.",
        "Point checks are not parcel determinations and should be confirmed with official FEMA and local flood resources.",
    ]
    if not inside:
        limitations.append("Not being inside the loaded layer does not mean flood impacts are impossible.")

    zone_text = f" and matched {hazard_layers[0]['name']}" if hazard_layers else ""
    context_text = f" The address matched {layers[0]['name']}, which is not treated here as Special Flood Hazard Area membership." if layers and not inside else ""
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
        source_url=get_source("fema_nfhl").url,
        confidence="source_backed",
        review_status="reviewed",
        why_shown=(
            f"Flood is shown because the saved address point was checked against an official FEMA flood layer{zone_text}."
            if inside else
            f"Flood is shown because the saved address point was checked against the FEMA flood layer and was not inside a loaded Special Flood Hazard Area.{context_text}"
        ),
        limitations=limitations,
        recommended_actions=DEFAULT_ACTIONS["flood"],
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
    exposure = "high" if "very high" in hazard_class or "high" in hazard_class else "medium" if inside else "low"
    specialized_guidance = _specialized_guidance(location_result, hazard_type, user_context)
    limitations = [
        "This checks the saved address point against the loaded CAL FIRE fire hazard polygon layer.",
        "Fire hazard zones do not capture all wildfire smoke, evacuation, power shutoff, or ember exposure.",
    ]
    if not inside:
        limitations.append("Not being inside the loaded fire hazard polygon does not mean smoke, evacuation, or regional wildfire impacts are impossible.")

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
        source_url=get_source("calfire_fhsz").url,
        confidence="source_backed",
        review_status="draft_reviewed",
        why_shown=(
            f"Wildfire is shown because the saved address point was checked against a CAL FIRE fire hazard layer{zone_text}."
            if inside else
            f"Wildfire is shown because the saved address point was checked against the CAL FIRE fire hazard layer and was not inside a loaded moderate, high, or very-high fire hazard zone.{context_text}"
        ),
        limitations=limitations,
        recommended_actions=DEFAULT_ACTIONS["wildfire"],
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
        recommended_actions=DEFAULT_ACTIONS["earthquake"],
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
    if location_result.lat is not None and location_result.lon is not None and location_result.formatted_address:
        flood_check = check_flood_layer(location_result.lat, location_result.lon)
        wildfire_check = check_wildfire_layer(location_result.lat, location_result.lon)
        fault_check = check_fault_layer(location_result.lat, location_result.lon)

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

        results.append(result)

    def sort_key(item: HazardResult):
        scope_rank = {"address_level": 4, "jurisdiction_level": 3, "zip_estimate": 2, "county_fallback": 1}
        exposure_rank = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
        return (
            scope_rank.get(item.scope, 0),
            exposure_rank.get(item.exposure_level, 0),
            item.legacy_score or 0,
            item.legacy_priority_score or 0,
        )

    return sorted(results, key=sort_key, reverse=True)


def merge_structured_result(hazard: Dict, result: HazardResult) -> Dict:
    merged = deepcopy(hazard)
    payload = result.model_dump()
    local_summary = _local_summary(result)
    resident_actions = _resident_actions(result)
    merged["structured_result"] = payload
    merged["scope_label"] = display_scope(result.scope)
    merged["data_status_label"] = display_data_status(result.data_status)
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
    merged["action_steps"] = resident_actions
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
    merged["recommended_actions"] = [item.model_dump() for item in result.recommended_actions]
    merged["recovery_questions"] = [item.model_dump() for item in result.recovery_questions]
    merged["sources"] = [item.model_dump() for item in result.sources]
    merged["local_plan_match"] = result.local_plan_match
    merged["specialized_guidance"] = result.specialized_guidance.model_dump()
    merged["matched_layers"] = result.matched_layers
    merged["is_in_hazard_zone"] = result.is_in_hazard_zone
    merged["match_type"] = result.match_type
    merged["data_status"] = result.data_status
    merged["scope"] = result.scope
    merged["basis"] = result.basis
    return merged
