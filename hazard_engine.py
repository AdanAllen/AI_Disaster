import json
import os
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional

from shapely.geometry import Point, shape

from pydantic_models import HazardResult, PreparednessAction, RecoveryQuestion, SpecializedGuidance
from source_registry import get_city_chunks, get_local_plan_for_city, get_source, get_sources_for_hazard


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
        "not_in_layer": "Checked, not in layer",
        "fallback_used": "Fallback used",
        "needs_review": "Needs review",
    }
    return labels.get(status, status.replace("_", " ").title())


@lru_cache(maxsize=4)
def load_geojson(filename: str) -> Dict:
    path = os.path.join(BASE_DIR, "static", filename)
    if not os.path.exists(path):
        return {"type": "FeatureCollection", "features": []}
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def check_flood_layer(lat: Optional[float], lon: Optional[float]) -> Dict:
    if lat is None or lon is None:
        return {"checked": False, "inside": None, "layers": []}

    point = Point(float(lon), float(lat))
    data = load_geojson("FldHaz.geojson")
    layers = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            polygon = shape(geometry)
        except Exception:
            continue
        if polygon.contains(point) or polygon.touches(point):
            props = feature.get("properties", {})
            layers.append({
                "layer_id": props.get("FLD_AR_ID") or props.get("SOURCE_CIT") or "fema_flood_layer",
                "name": f"FEMA flood zone {props.get('FLD_ZONE', 'unknown')}",
                "zone": props.get("FLD_ZONE", "Unknown"),
                "source_citation": props.get("SOURCE_CIT", ""),
                "sfha": props.get("SFHA_TF", ""),
            })

    return {"checked": True, "inside": bool(layers), "layers": layers}


def _source_payload(hazard_type: str, extra_source_ids: Optional[List[str]] = None):
    sources = get_sources_for_hazard(hazard_type)
    existing = {source.source_id for source in sources}
    for source_id in extra_source_ids or []:
        if source_id and source_id not in existing:
            sources.append(get_source(source_id))
            existing.add(source_id)
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


def _specialized_guidance(location_result, hazard_type: str, user_context: Optional[Dict] = None) -> SpecializedGuidance:
    user_context = user_context or {}
    city_chunks = get_city_chunks(location_result.city, hazard_type)
    city_context = [chunk.get("text", "") for chunk in city_chunks if chunk.get("text")]
    source_ids = list(dict.fromkeys(chunk.get("source_id") for chunk in city_chunks if chunk.get("source_id")))
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
        city_context.append(
            f"No reviewed {location_result.city} hazard chunk is available for this hazard yet, so city-specific claims remain limited."
        )

    return SpecializedGuidance(
        city_context=city_context,
        household_factors=household_factors,
        access_functional_needs=access_functional_needs,
        recovery_needs=recovery_needs,
        source_ids=source_ids,
    )


def _jurisdiction_result(hazard: Dict, location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = hazard.get("slug") or hazard.get("name", "").lower()
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
        label=hazard.get("name", hazard_type.title()),
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
            f"{hazard.get('name', hazard_type.title())} is shown because {city} and Alameda County source context "
            "identify it as relevant for local preparedness planning."
        ),
        limitations=limitations,
        recommended_actions=DEFAULT_ACTIONS.get(hazard_type, DEFAULT_ACTIONS["earthquake"]),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(hazard_type, ["alameda_county_emergency", "ready_recovery"]),
        local_plan_match=local_plan_match,
        specialized_guidance=specialized_guidance,
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def _zip_result(hazard: Dict, location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> HazardResult:
    hazard_type = hazard.get("slug") or hazard.get("name", "").lower()
    score = _legacy_score(zip_snapshot, hazard_type)
    exposure = exposure_from_score(score)
    explanation = _zip_explanation(zip_snapshot, hazard_type)
    zip_code = location_result.zip_code or "the selected ZIP"

    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=hazard.get("name", hazard_type.title()),
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
            f"{hazard.get('name', hazard_type.title())} is shown from ZIP-level fallback context for {zip_code}. "
            f"{explanation}".strip()
        ),
        limitations=[
            "ZIP estimates are fallback guidance and do not prove whether an individual address is inside a hazard zone.",
            "Use official maps and emergency instructions for final decisions.",
        ],
        recommended_actions=DEFAULT_ACTIONS.get(hazard_type, DEFAULT_ACTIONS["earthquake"]),
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(hazard_type, ["alameda_county_emergency", "ready_recovery"]),
        local_plan_match=_local_plan_match(location_result, hazard_type),
        specialized_guidance=_specialized_guidance(location_result, hazard_type, user_context),
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def _county_result(hazard: Dict) -> HazardResult:
    hazard_type = hazard.get("slug") or hazard.get("name", "").lower()
    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=hazard.get("name", hazard_type.title()),
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
        why_shown=f"{hazard.get('name', hazard_type.title())} is shown as general Alameda County preparedness guidance.",
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
    inside = bool(flood_check.get("inside"))
    layers = flood_check.get("layers") or []
    status = "checked" if inside else "not_in_layer"
    exposure = "high" if inside else "low"
    limitations = [
        "This checks the saved address point against the current FEMA flood polygon layer in the app.",
        "Point checks are not parcel determinations and should be confirmed with official FEMA and local flood resources.",
    ]
    if not inside:
        limitations.append("Not being inside the loaded layer does not mean flood impacts are impossible.")

    zone_text = f" and matched {layers[0]['name']}" if layers else ""
    return HazardResult(
        hazard_id=hazard_type,
        hazard_type=hazard_type,
        label=hazard.get("name", "Flood"),
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
            "Flood is shown because the saved address point was checked against the FEMA flood layer and was not inside the loaded flood polygon."
        ),
        limitations=limitations,
        recommended_actions=DEFAULT_ACTIONS["flood"],
        recovery_questions=RECOVERY_QUESTIONS,
        sources=_source_payload(hazard_type, ["fema_nfhl", "ready_floods", "ready_recovery"]),
        local_plan_match=_local_plan_match(location_result, hazard_type),
        specialized_guidance=_specialized_guidance(location_result, hazard_type, user_context),
        legacy_score=score,
        legacy_priority_score=hazard.get("priority_score"),
    )


def build_hazard_results(hazards: List[Dict], location_result, zip_snapshot: Dict, user_context: Optional[Dict] = None) -> List[HazardResult]:
    results = []
    flood_check = {"checked": False, "inside": None, "layers": []}
    if location_result.lat is not None and location_result.lon is not None and location_result.formatted_address:
        flood_check = check_flood_layer(location_result.lat, location_result.lon)

    for raw_hazard in hazards:
        hazard = deepcopy(raw_hazard)
        hazard_type = hazard.get("slug") or hazard.get("name", "").strip().lower()

        if hazard_type == "flood" and flood_check.get("checked"):
            result = _flood_address_result(hazard, location_result, flood_check, zip_snapshot, user_context)
        elif location_result.city or location_result.county:
            result = _jurisdiction_result(hazard, location_result, zip_snapshot, user_context)
            if hazard_type in {"wildfire", "earthquake"} and location_result.formatted_address:
                result.data_status = "not_checked"
                result.limitations.insert(0, "Address-level GIS membership has not been checked for this hazard yet.")
                result.why_shown = (
                    f"{result.label} is shown from jurisdiction-level source context. "
                    "The app has not completed an address-level GIS check for this hazard yet."
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
    merged["structured_result"] = payload
    merged["scope_label"] = display_scope(result.scope)
    merged["data_status_label"] = display_data_status(result.data_status)
    merged["basis_label"] = result.basis.replace("_", " ").title()
    merged["location_precision_label"] = result.location_precision.replace("_", " ").title()
    merged["exposure_level"] = result.exposure_level.title()
    merged["why_shown"] = result.why_shown
    merged["limitations"] = result.limitations
    merged["recommended_actions"] = [item.model_dump() for item in result.recommended_actions]
    merged["recovery_questions"] = [item.model_dump() for item in result.recovery_questions]
    merged["sources"] = [item.model_dump() for item in result.sources]
    merged["local_plan_match"] = result.local_plan_match
    merged["specialized_guidance"] = result.specialized_guidance.model_dump()
    merged["matched_layers"] = result.matched_layers
    merged["is_in_hazard_zone"] = result.is_in_hazard_zone
    merged["data_status"] = result.data_status
    merged["scope"] = result.scope
    merged["basis"] = result.basis
    return merged
