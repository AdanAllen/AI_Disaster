import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point, shape
from shapely.ops import nearest_points


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "hazard_priority")

SUPPORTED_HAZARDS = ["earthquake", "wildfire", "flood", "landslide", "tsunami"]
UNKNOWN = "Unknown"
LOCAL_GENERAL = "General citywide exposure"
LOCAL_DIRECT = "Direct mapped exposure"
LOCAL_ADDITIONAL = "Additional mapped concern"
LOCAL_NONE = "No mapped exposure identified in checked layers"
LOCAL_UNAVAILABLE = "Data unavailable"
LOCAL_NOT_APPLICABLE = "Not applicable"
MAPPED_SIGNIFICANT = "significant_official_finding"
MAPPED_NO_SIGNIFICANT = "successful_check_no_significant_match"
MAPPED_UNAVAILABLE = "data_unavailable"
MAPPED_NOT_SUPPORTED = "not_supported"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _load_json(filename: str, default):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def load_priority_data() -> Dict:
    return {
        "methodology": _load_json("methodology.json", {}),
        "rankings": _load_json("jurisdiction_hazard_rankings.json", {"records": []}),
        "sources": _load_json("source_documents.json", {"sources": {}}),
        "aliases": _load_json("hazard_aliases.json", {"aliases": {}}),
        "layer_rules": _load_json("official_gis_layer_rules.json", {"rules": []}),
        "top_four": _load_json("top_four_rules.json", {}),
        "sub_areas": _load_json("jurisdiction_sub_areas.json", {"jurisdictions": {}}),
        "sub_area_evidence": _load_json("sub_area_evidence.json", {"records": []}),
        "area_scenario_ratings": _load_json("oakland_lhmp_area_scenario_ratings.json", {"records": []}),
        "templates": _load_json("explanation_templates.json", {"templates": {}}),
    }


def _canonical_hazard(value: str) -> str:
    normalized = _slug(value).replace("_seiche", "")
    aliases = load_priority_data().get("aliases", {}).get("aliases", {})
    for hazard, values in aliases.items():
        if normalized == _slug(hazard) or normalized in {_slug(item) for item in values}:
            return hazard
    if "earthquake_landslide" in normalized or "landslide" in normalized:
        return "landslide"
    return normalized


def _hazard_label(hazard: str) -> str:
    return {
        "earthquake": "Earthquake",
        "wildfire": "Wildfire",
        "flood": "Flood",
        "landslide": "Landslide",
        "tsunami": "Tsunami",
    }.get(hazard, hazard.replace("_", " ").title())


def _ranking_record(jurisdiction: str, hazard: str) -> Optional[Dict]:
    jurisdiction_key = _slug(jurisdiction)
    hazard_key = _canonical_hazard(hazard)
    for record in load_priority_data().get("rankings", {}).get("records", []):
        if _slug(record.get("jurisdiction")) == jurisdiction_key and _canonical_hazard(record.get("hazard")) == hazard_key:
            return record
    return None


def calculate_citywide_priority(probability: str, impact: str) -> str:
    probability = probability if probability in {"Low", "Medium", "High"} else UNKNOWN
    impact = impact if impact in {"Low", "Medium", "High"} else UNKNOWN
    if UNKNOWN in {probability, impact}:
        return UNKNOWN
    matrix = load_priority_data().get("methodology", {}).get("matrix", {})
    return matrix.get(probability, {}).get(impact, UNKNOWN)


def _source_payload(record: Dict) -> Dict:
    source_id = record.get("source_document_id") or ""
    source = (load_priority_data().get("sources", {}).get("sources") or {}).get(source_id, {})
    status = record.get("document_status") or source.get("document_status") or ""
    return {
        "name": record.get("source_document") or source.get("source_document") or "Source unavailable",
        "url": source.get("source_url") or "",
        "reference": f"{record.get('source_table') or 'Source table'}; page {record.get('source_page')}" if record.get("source_page") else record.get("source_table") or "",
        "document_status": status,
        "status_label": status.title() if status else "Unknown Status",
        "publication_date": record.get("publication_date") or source.get("publication_date") or "",
    }


def _iter_gis_records(address_gis_results) -> List[Dict]:
    if not address_gis_results:
        return []
    if isinstance(address_gis_results, dict):
        if isinstance(address_gis_results.get("hazards"), list):
            candidates = address_gis_results["hazards"]
        else:
            candidates = list(address_gis_results.values())
    else:
        candidates = address_gis_results

    records = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        parent = item.get("slug") or item.get("hazard_id") or item.get("hazard") or item.get("hazard_type")
        records.append({**item, "_parent_hazard": parent})
        for key in ("additional_geospatial_evidence", "official_mapped_evidence", "normalized_mapped_evidence"):
            for evidence in item.get(key) or []:
                if isinstance(evidence, dict):
                    records.append({**evidence, "_parent_hazard": parent})
        if isinstance(item.get("geospatial_evidence"), dict):
            evidence = {**item["geospatial_evidence"], "_parent_hazard": parent}
            records.append(evidence)
            for feature in evidence.get("matched_features") or []:
                if isinstance(feature, dict):
                    records.append({**feature, "_parent_hazard": parent, "_matched_layer": True})
        for layer in item.get("matched_layers") or []:
            if isinstance(layer, dict):
                records.append({**layer, "_parent_hazard": parent, "_matched_layer": True})
        for layer in item.get("layers") or []:
            if isinstance(layer, dict):
                records.append({**layer, "_parent_hazard": parent, "_matched_layer": True})
        for feature in item.get("matched_features") or []:
            if isinstance(feature, dict):
                records.append({**feature, "_parent_hazard": parent, "_matched_layer": True})
    return records


def _record_text(record: Dict) -> str:
    fields = [
        record.get("hazard"),
        record.get("hazard_id"),
        record.get("hazard_type"),
        record.get("slug"),
        record.get("_parent_hazard"),
        record.get("map_layer_key"),
        record.get("dataset_id"),
        record.get("dataset_name"),
        record.get("name"),
        record.get("layer_id"),
        record.get("result_label"),
    ]
    return " ".join(str(value or "") for value in fields).lower().replace("_", "-")


def _record_applies_to_hazard(record: Dict, hazard: str, rule: Dict) -> bool:
    text = _record_text(record)
    if hazard == "landslide" and "earthquake-landslide" in text:
        return True
    if hazard == "tsunami" and "tsunami" in text:
        return True
    return any(str(token).lower() in text for token in rule.get("dataset_match", []))


def _record_direct_match(record: Dict) -> bool:
    if record.get("matched") is True:
        return True
    if record.get("is_in_hazard_zone") is True:
        return True
    if record.get("inside") is True or record.get("sfha_match") is True:
        return True
    if record.get("hazard_exposure") == "mapped_match":
        return True
    if record.get("match_type") == "inside":
        return True
    return False


def _record_proximity_match(record: Dict) -> bool:
    if record.get("near") is True:
        return True
    if record.get("match_type") == "near_fault":
        return True
    return record.get("matched") is True and (
        record.get("claim_type") == "proximity"
        or record.get("status") == "proximity_context"
    )


def _record_checked(record: Dict) -> bool:
    return (
        record.get("checked") is True
        or record.get("data_status") in {"checked", "not_in_layer"}
        or record.get("exposure") in {"mapped_match", "no_mapped_match"}
    )


def _record_unavailable(record: Dict) -> bool:
    return record.get("data_status") == "data_unavailable" or record.get("data_available") is False


def _terms(record: Dict, fields: List[str]) -> List[str]:
    output = []
    for field in fields:
        value = record.get(field)
        if value:
            output.append(str(value))
    props = record.get("properties") or {}
    if isinstance(props, dict):
        for field in fields:
            value = props.get(field)
            if value:
                output.append(str(value))
    return _dedupe(output)


def _dedupe(items: List[str]) -> List[str]:
    deduped = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _rating_value(rating: str) -> int:
    return {"Unknown": 0, "Low": 1, "Medium": 2, "High": 3}.get(rating or UNKNOWN, 0)


def _rating_from_average(score: Optional[float]) -> str:
    if score is None:
        return UNKNOWN
    if 1.0 <= score <= 1.49:
        return "Low"
    if 1.5 <= score <= 2.49:
        return "Medium"
    if 2.5 <= score <= 3.0:
        return "High"
    return UNKNOWN


def combine_scenario_ratings(scenario_records: List[Dict]) -> Dict:
    """Combine official categorical scenario ratings without mixing sources."""
    ratings = []
    valid_records = []
    excluded_records = []
    for record in scenario_records:
        rating = record.get("official_rating") or record.get("rating")
        value = _rating_value(rating)
        if value:
            ratings.append(value)
            valid_records.append(record)
        else:
            excluded_records.append(record)

    if not ratings:
        score = None
        combined = UNKNOWN
    else:
        score = sum(ratings) / len(ratings)
        combined = _rating_from_average(score)

    if len(valid_records) == 1:
        method_note = "One valid official LHMP scenario rating is available; no averaging was required."
    elif valid_records:
        method_note = "StayReady combines the official Oakland LHMP scenario ratings available for this plan area using an unweighted category average. This combined category is a StayReady summary of official scenario results, not a new official Oakland rating."
    else:
        method_note = "No valid official Low, Medium, or High scenario rating is available to average."

    return {
        "scenario_ratings": [
            {
                "scenario_name": record.get("scenario_name") or record.get("scenario") or "Official scenario",
                "official_rating": record.get("official_rating") or UNKNOWN,
                "official_numeric_risk_ranking_value": record.get("official_numeric_risk_ranking_value"),
                "source_plan": record.get("source_plan") or "",
                "plan_version": record.get("plan_version") or "",
                "document_status": record.get("document_status") or "",
                "source_chapter": record.get("source_chapter") or "",
                "source_page": record.get("source_page"),
                "source_table": record.get("source_table") or "",
                "methodology_notes": record.get("methodology_notes") or "",
                "limitations": record.get("limitations") or "",
            }
            for record in scenario_records
        ],
        "valid_scenario_count": len(valid_records),
        "excluded_scenario_count": len(excluded_records),
        "combined_scenario_score_internal": score,
        "combined_area_rating": combined,
        "combination_method": "unweighted average of official LHMP scenario categories",
        "combination_explanation": method_note,
    }


def _record_is_verified(record: Dict) -> bool:
    return (
        record.get("displayed_level_status") == "verified"
        or record.get("verification_status") == "verified"
        or record.get("visually_verified") is True
    ) and bool(record.get("last_verified"))


def _unsupported_source_trace(record: Dict, reason: str) -> Dict:
    return {
        "source_document": record.get("source_plan") or record.get("source_document") or "",
        "source_status": record.get("document_status") or "",
        "source_page": record.get("source_page"),
        "source_table": record.get("source_table") or "",
        "source_row": record.get("source_row") or "",
        "raw_value": record.get("raw_value") or record.get("official_rating") or "",
        "calculation_steps": [reason],
        "fallback_used": False,
        "last_verified": record.get("last_verified") or "",
        "displayed_level_status": "unsupported",
    }


def _scenario_records_for_area(jurisdiction: str, plan_area: str, hazard: str, status_preference: str = "adopted") -> List[Dict]:
    if not plan_area or plan_area == UNKNOWN:
        return []
    records = [
        record for record in load_priority_data().get("area_scenario_ratings", {}).get("records", [])
        if _slug(record.get("jurisdiction")) == _slug(jurisdiction)
        and _normalize_sub_area_name(record.get("plan_area")) == _normalize_sub_area_name(plan_area)
        and _canonical_hazard(record.get("hazard")) == _canonical_hazard(hazard)
    ]
    preferred = [record for record in records if record.get("document_status") == status_preference]
    return preferred or records


def official_area_rating_for_hazard(jurisdiction: str, plan_area: str, hazard: str) -> Dict:
    records = _scenario_records_for_area(jurisdiction, plan_area, hazard)
    verified_records = [record for record in records if _record_is_verified(record)]
    unsupported_records = [record for record in records if not _record_is_verified(record)]
    combined = combine_scenario_ratings(verified_records)
    source_statuses = _dedupe([record.get("document_status") for record in records])
    source_plans = _dedupe([record.get("source_plan") for record in records])
    unsupported_reason = ""
    if unsupported_records and not verified_records:
        unsupported_reason = "Official area/scenario records exist in local JSON, but they lack required source-row/raw-value/visual-verification provenance and are fail-closed to Unknown."
    elif unsupported_records:
        unsupported_reason = "Some local scenario records were excluded because they lack required verification provenance."
    return {
        **combined,
        "official_lhmp_area_rating": combined["combined_area_rating"],
        "displayed_level_status": "verified" if combined["valid_scenario_count"] else ("unsupported" if unsupported_records else "unavailable"),
        "unsupported_reason": unsupported_reason,
        "unsupported_source_records": [_unsupported_source_trace(record, unsupported_reason) for record in unsupported_records],
        "source_statuses": source_statuses,
        "source_plans": source_plans,
        "sources": [
            {
                "name": record.get("source_plan") or "Oakland LHMP area scenario rating",
                "url": record.get("source_url") or "",
                "reference": f"{record.get('source_table') or 'Source table'}; page {record.get('source_page')}",
                "document_status": record.get("document_status") or "",
                "status_label": (record.get("document_status") or "unknown").title(),
                "publication_date": record.get("plan_version") or "",
            }
            for record in verified_records
        ],
    }


def local_exposure_for_hazard(hazard: str, address_gis_results) -> Dict:
    hazard = _canonical_hazard(hazard)
    rules = load_priority_data().get("layer_rules", {}).get("rules", [])
    rule = next((item for item in rules if item.get("hazard") == hazard), {})
    direct_matches = []
    proximity_matches = []
    checked_nonmatches = []
    unavailable = []
    direct_match_terms = []
    proximity_match_terms = []
    checked_nonmatch_terms = []
    checked_layers = []
    successful_layers = []
    last_checked = []

    for record in _iter_gis_records(address_gis_results):
        if not rule or not _record_applies_to_hazard(record, hazard, rule):
            continue
        layer_name = (
            record.get("dataset_name")
            or record.get("name")
            or record.get("result_label")
            or record.get("source_agency")
            or "Official checked layer"
        )
        checked_layers.append(layer_name)
        record_terms = _terms(record, rule.get("terminology_fields", []))
        if record.get("checked_at"):
            last_checked.append(str(record["checked_at"]))
        if hazard == "earthquake" and _record_proximity_match(record):
            proximity_matches.append(layer_name)
            successful_layers.append(layer_name)
            proximity_match_terms.extend(record_terms)
        elif _record_direct_match(record):
            direct_matches.append(layer_name)
            successful_layers.append(layer_name)
            direct_match_terms.extend(record_terms)
            if hazard in {"flood", "wildfire"}:
                for nested_key in ("matched_layers", "layers"):
                    for nested_record in record.get(nested_key) or []:
                        if isinstance(nested_record, dict):
                            direct_match_terms.extend(
                                _terms(nested_record, rule.get("terminology_fields", []))
                            )
        elif _record_unavailable(record):
            unavailable.append(layer_name)
        elif _record_checked(record):
            checked_nonmatches.append(layer_name)
            checked_nonmatch_terms.extend(record_terms)

    if direct_matches:
        status = rule.get("match_status") or LOCAL_DIRECT
    elif proximity_matches:
        status = LOCAL_ADDITIONAL
    elif unavailable and not checked_nonmatches and not successful_layers:
        status = LOCAL_UNAVAILABLE
    elif checked_nonmatches:
        status = rule.get("nonmatch_status") or LOCAL_NONE
    elif hazard == "earthquake":
        status = LOCAL_GENERAL
    else:
        status = LOCAL_NOT_APPLICABLE

    if direct_matches:
        terms = direct_match_terms
    elif proximity_matches:
        terms = proximity_match_terms
    elif checked_nonmatches:
        terms = checked_nonmatch_terms
    else:
        terms = []

    if status == LOCAL_DIRECT:
        basis = "Official polygon intersection found in checked layer."
    elif status == LOCAL_ADDITIONAL:
        basis = rule.get("basis_template") or "Official mapped concern applies locally."
    elif status == LOCAL_NONE:
        basis = "No mapped exposure identified in the checked layer. This does not prove the location has no hazard risk."
    elif status == LOCAL_UNAVAILABLE:
        basis = "One or more official GIS layers were unavailable or failed, so local mapped exposure could not be determined."
    elif status == LOCAL_GENERAL:
        basis = "No local earthquake layer match changed the display context; citywide earthquake exposure still applies."
    else:
        basis = "No address-specific official GIS result was supplied for this hazard."

    limitations = []
    if status in {LOCAL_NONE, LOCAL_GENERAL}:
        limitations.append("Being outside one checked mapped layer does not mean the hazard cannot affect this location.")
    if status == LOCAL_UNAVAILABLE:
        limitations.append("Unavailable GIS data was not used to lower or raise the citywide hazard priority.")
    if hazard == "earthquake":
        limitations.append("Earthquake local layers describe ground-failure or proximity concerns and do not rewrite citywide Probability or Impact.")

    return {
        "local_exposure_status": status,
        "local_exposure_basis": basis,
        "official_layers_checked": _dedupe(checked_layers),
        "successful_layers": _dedupe(successful_layers),
        "unavailable_layers": _dedupe(unavailable),
        "polygon_intersections": _dedupe(direct_matches),
        "proximity_results": _dedupe(proximity_matches),
        "source_specific_terminology": _dedupe(terms),
        "confidence": "Medium" if successful_layers or checked_nonmatches else "Low",
        "limitations": limitations,
        "data_last_checked": max(last_checked) if last_checked else "",
    }


def _coordinate_pair(coordinates: Optional[Dict]) -> Optional[Tuple[float, float]]:
    if not coordinates:
        return None
    lat = coordinates.get("lat") or coordinates.get("latitude")
    lon = coordinates.get("lon") or coordinates.get("lng") or coordinates.get("longitude")
    try:
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def _sub_area_features(jurisdiction_key: str) -> List[Dict]:
    config = (load_priority_data().get("sub_areas", {}).get("jurisdictions") or {}).get(jurisdiction_key, {})
    dataset = config.get("boundary_dataset")
    if not dataset:
        return []
    path = dataset if os.path.isabs(dataset) else os.path.join(BASE_DIR, dataset)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as source:
        geojson = json.load(source)
    return geojson.get("features") or []


def _normalize_sub_area_name(value: str) -> str:
    return re.sub(r"\s*/\s*", "/", " ".join(str(value or "").split()))


def _sub_area_context(
    jurisdiction: str,
    optional_sub_area_match: Optional[Dict] = None,
    coordinates: Optional[Dict] = None,
) -> Dict:
    if optional_sub_area_match and optional_sub_area_match.get("sub_area"):
        return {
            "sub_area": optional_sub_area_match.get("sub_area"),
            "sub_area_status": optional_sub_area_match.get("sub_area_status") or "Matched reliable boundary",
            "sub_area_match_status": optional_sub_area_match.get("sub_area_match_status") or optional_sub_area_match.get("sub_area_status") or "Matched reliable boundary",
            "sub_area_basis": optional_sub_area_match.get("basis") or "",
            "sub_area_source": optional_sub_area_match.get("sub_area_source") or "",
            "sub_area_source_url": optional_sub_area_match.get("sub_area_source_url") or "",
            "polygon_version": optional_sub_area_match.get("polygon_version") or "",
            "data_last_checked": optional_sub_area_match.get("data_last_checked") or "",
            "sub_area_limitations": optional_sub_area_match.get("limitations") or [],
            "boundary_distance_m": optional_sub_area_match.get("boundary_distance_m"),
            "boundary_warning": optional_sub_area_match.get("boundary_warning") or "",
        }
    jurisdiction_key = _slug(jurisdiction)
    config = (load_priority_data().get("sub_areas", {}).get("jurisdictions") or {}).get(jurisdiction_key, {})
    base = {
        "sub_area": UNKNOWN,
        "sub_area_status": config.get("sub_area_source_status") or "Boundary data unavailable",
        "sub_area_match_status": config.get("sub_area_source_status") or "Boundary data unavailable",
        "sub_area_basis": "",
        "sub_area_source": config.get("source_name") or "",
        "sub_area_source_url": config.get("source_url") or "",
        "polygon_version": config.get("polygon_version") or "",
        "data_last_checked": config.get("data_last_checked") or "",
        "sub_area_limitations": config.get("limitations") or ["No reliable boundary dataset is available for sub-area assignment."],
        "boundary_distance_m": None,
        "boundary_warning": "",
    }

    pair = _coordinate_pair(coordinates)
    if not pair:
        base["sub_area_match_status"] = "Coordinates unavailable; boundary not evaluated"
        base["sub_area_status"] = base["sub_area_match_status"]
        base["sub_area_limitations"] = _dedupe(base["sub_area_limitations"] + ["Sub-area assignment requires address coordinates; ZIP, street, or neighborhood text is not used."])
        return base

    features = _sub_area_features(jurisdiction_key)
    if not features:
        return base

    valid_names = {_normalize_sub_area_name(item) for item in config.get("sub_areas") or []}
    source_field = config.get("match_field") or "OAKLAND_PE"
    point = Point(pair[1], pair[0])
    matches = []
    non_lhmp_matches = []
    nearest_boundary_distance_m = None
    for feature in features:
        try:
            polygon = shape(feature.get("geometry"))
        except Exception:
            continue
        if not polygon.is_empty:
            try:
                _, nearest_on_boundary = nearest_points(point, polygon.boundary)
                distance_m = point.distance(nearest_on_boundary) * 111_139
                nearest_boundary_distance_m = (
                    distance_m
                    if nearest_boundary_distance_m is None
                    else min(nearest_boundary_distance_m, distance_m)
                )
            except Exception:
                pass
        if not polygon.contains(point) and not polygon.touches(point):
            continue
        source_term = _normalize_sub_area_name((feature.get("properties") or {}).get(source_field) or "")
        if source_term in valid_names:
            matches.append(source_term)
        elif source_term:
            non_lhmp_matches.append(source_term)

    if len(matches) == 1:
        base.update({
            "sub_area": matches[0],
            "sub_area_status": "Matched official Oakland plan-area polygon",
            "sub_area_match_status": "Matched official Oakland plan-area polygon",
            "sub_area_basis": f"Point-in-polygon match against {config.get('source_name') or 'official boundary dataset'}.",
        })
        if nearest_boundary_distance_m is not None:
            base["boundary_distance_m"] = round(nearest_boundary_distance_m, 1)
            if nearest_boundary_distance_m <= 100:
                base["boundary_warning"] = "This location may be near a plan-area boundary."
    elif len(matches) > 1:
        base["sub_area_status"] = "Ambiguous official polygon match"
        base["sub_area_match_status"] = "Ambiguous official polygon match"
        base["sub_area_limitations"] = _dedupe(base["sub_area_limitations"] + ["The address point intersected more than one assignable sub-area polygon, so no sub-area was selected."])
        base["boundary_warning"] = "This location may be near a plan-area boundary."
    elif non_lhmp_matches:
        base["sub_area_status"] = "Matched official polygon outside configured LHMP sub-area list"
        base["sub_area_match_status"] = base["sub_area_status"]
        base["sub_area_basis"] = f"Matched official source term: {', '.join(non_lhmp_matches)}."
        base["sub_area_limitations"] = _dedupe(base["sub_area_limitations"] + ["This official source term is not one of the nine configured LHMP sub-areas, so the LHMP sub-area remains Unknown."])
    else:
        base["sub_area_status"] = "No official sub-area polygon match"
        base["sub_area_match_status"] = base["sub_area_status"]
    return {
        **base,
    }


def _community_context_for_hazard(jurisdiction: str, sub_area: str, hazard: str) -> Dict:
    if not sub_area or sub_area == UNKNOWN:
        return {
            "status": "Unavailable",
            "summary": "Sub-area community context is unavailable until an official sub-area polygon match is made.",
            "records": [],
            "limitations": ["Sub-area EPC and exposure tables are community context only and are not used to assign a personal hazard priority."],
        }

    records = [
        record for record in load_priority_data().get("sub_area_evidence", {}).get("records", [])
        if _slug(record.get("jurisdiction")) == _slug(jurisdiction)
        and _canonical_hazard(record.get("hazard")) == hazard
        and _normalize_sub_area_name(record.get("sub_area")) == _normalize_sub_area_name(sub_area)
    ]
    if not records:
        return {
            "status": "Unavailable",
            "summary": "No extracted LHMP sub-area table record is available for this hazard and sub-area.",
            "records": [],
            "limitations": ["Missing sub-area table data is not replaced with an inferred value."],
        }
    summaries = []
    for record in records[:3]:
        metric = record.get("metric_name") or "Sub-area metric"
        value = record.get("value")
        unit = record.get("unit") or ""
        summaries.append(f"{metric}: {value} {unit}".strip())
    return {
        "status": "Available",
        "summary": "; ".join(summaries) + ". These values are community context only and do not change Probability, Impact, or local mapped concern.",
        "records": records,
        "limitations": _dedupe([record.get("limitations") for record in records] + ["EPC exposure values describe community vulnerability context only. Zero EPC exposure is not a zero-hazard finding."]),
    }


def _official_zone_category(local: Dict) -> str:
    terms = local.get("source_specific_terminology") or []
    return ", ".join(terms) if terms else UNKNOWN


def _local_evidence_summary(local: Dict) -> str:
    if local.get("local_exposure_status") == LOCAL_UNAVAILABLE:
        return "Address-specific map information is temporarily unavailable."
    terms = _official_zone_category(local)
    if terms != UNKNOWN:
        return f"{local.get('local_exposure_status')}: {terms}."
    return f"{local.get('local_exposure_status')}. {local.get('local_exposure_basis')}"


def _source_name_for_local(hazard: str, local: Dict) -> str:
    layers = local.get("successful_layers") or local.get("official_layers_checked") or []
    if layers:
        return "; ".join(_dedupe(layers[:3]))
    return {
        "earthquake": "California Geological Survey earthquake hazard layers",
        "wildfire": "CAL FIRE Fire Hazard Severity Zone layer",
        "flood": "FEMA National Flood Hazard Layer snapshot",
        "landslide": "CGS and local landslide layers",
        "tsunami": "California Geological Survey tsunami hazard area layer",
    }.get(hazard, "Official mapped hazard layer")


def _has_any_text(local: Dict, *tokens: str) -> bool:
    text = " ".join(
        str(item or "")
        for item in (
            (local.get("source_specific_terminology") or [])
            + (local.get("official_layers_checked") or [])
            + (local.get("successful_layers") or [])
            + (local.get("polygon_intersections") or [])
            + (local.get("proximity_results") or [])
        )
    ).lower()
    return any(token.lower() in text for token in tokens)


def _mapped_finding_for_hazard(hazard: str, local: Dict) -> Dict:
    status = local.get("local_exposure_status")
    source_name = _source_name_for_local(hazard, local)
    terms = local.get("source_specific_terminology") or []
    terms_text = " ".join(terms).lower()
    unavailable = local.get("unavailable_layers") or []
    checked_layers = local.get("official_layers_checked") or []
    successful_layers = local.get("successful_layers") or []
    confidence = "Medium" if checked_layers or successful_layers else "Low"

    if status == LOCAL_UNAVAILABLE or (unavailable and not checked_layers and not successful_layers):
        return {
            "mapped_finding_status": MAPPED_UNAVAILABLE,
            "mapped_finding_category": "Map information unavailable",
            "mapped_finding_summary": "Address-specific map information is temporarily unavailable.",
            "mapped_finding_interpretation": "This failed check was kept separate and was not used to lower or raise a hazard priority.",
            "mapped_finding_source_name": source_name,
            "mapped_finding_confidence": "Low",
        }

    if status in {LOCAL_NOT_APPLICABLE, LOCAL_GENERAL} and not checked_layers and not successful_layers:
        return {
            "mapped_finding_status": MAPPED_NOT_SUPPORTED,
            "mapped_finding_category": "Map information unavailable",
            "mapped_finding_summary": "No address-specific official map result is available for this hazard.",
            "mapped_finding_interpretation": "StayReady does not infer a mapped concern where address-level source evidence is missing.",
            "mapped_finding_source_name": source_name,
            "mapped_finding_confidence": "Low",
        }

    if hazard == "flood":
        if status == LOCAL_DIRECT and _has_any_text(local, "special flood hazard", "zone ae", "zone a", "sfha", "1%"):
            summary = terms[0] if terms else "Mapped Special Flood Hazard Area finding."
            return {
                "mapped_finding_status": MAPPED_SIGNIFICANT,
                "mapped_finding_category": "Official mapped finding",
                "mapped_finding_summary": summary,
                "mapped_finding_interpretation": "The checked flood layer indicates Special Flood Hazard Area membership for the address point.",
                "mapped_finding_source_name": source_name,
                "mapped_finding_confidence": confidence,
            }
        zone = next((term for term in terms if "zone x" in term.lower() or term.strip().lower() == "x"), "")
        summary = "FEMA Zone X, not identified as a Special Flood Hazard Area." if zone else "No Special Flood Hazard Area match found in the checked layer."
        return {
            "mapped_finding_status": MAPPED_NO_SIGNIFICANT,
            "mapped_finding_category": "Other hazard checked",
            "mapped_finding_summary": summary,
            "mapped_finding_interpretation": "This is a checked map result, not a guarantee that flooding or drainage impacts cannot occur.",
            "mapped_finding_source_name": source_name,
            "mapped_finding_confidence": confidence,
        }

    if hazard == "wildfire":
        if status == LOCAL_DIRECT:
            summary = next((term for term in terms if "fire hazard severity" in term.lower()), terms[0] if terms else "Mapped Fire Hazard Severity Zone finding.")
            return {
                "mapped_finding_status": MAPPED_SIGNIFICANT,
                "mapped_finding_category": "Official mapped finding",
                "mapped_finding_summary": summary,
                "mapped_finding_interpretation": "The address point intersects the checked Fire Hazard Severity Zone layer. This is mapped concern language, not an exact personal risk prediction.",
                "mapped_finding_source_name": source_name,
                "mapped_finding_confidence": confidence,
            }

    if hazard == "earthquake":
        if local.get("polygon_intersections") and _has_any_text(
            local,
            "alquist-priolo",
            "liquefaction zone",
            "earthquake-induced landslide zone",
        ):
            ap_match = next((term for term in terms if "alquist-priolo" in term.lower()), "")
            summary = ap_match or next((term for term in terms if "mapped" in term.lower()), terms[0] if terms else "Official earthquake map finding.")
            interpretation = "This official mapped finding applies to the address point. Fault proximity, if shown, is separate context and is not the same as an Alquist-Priolo polygon match."
            return {
                "mapped_finding_status": MAPPED_SIGNIFICANT,
                "mapped_finding_category": "Official mapped finding",
                "mapped_finding_summary": summary,
                "mapped_finding_interpretation": interpretation,
                "mapped_finding_source_name": source_name,
                "mapped_finding_confidence": confidence,
            }
        if status == LOCAL_ADDITIONAL:
            return {
                "mapped_finding_status": MAPPED_NO_SIGNIFICANT,
                "mapped_finding_category": "Other hazard checked",
                "mapped_finding_summary": "Fault proximity context returned by the checked data.",
                "mapped_finding_interpretation": "This is mapped context for preparedness and is not presented as a personal damage prediction.",
                "mapped_finding_source_name": source_name,
                "mapped_finding_confidence": confidence,
            }

    if hazard == "landslide":
        if status in {LOCAL_DIRECT, LOCAL_ADDITIONAL} and not _has_any_text(local, "none", "no mapped match"):
            summary = next((term for term in terms if "landslide" in term.lower()), terms[0] if terms else "Official landslide map finding.")
            return {
                "mapped_finding_status": MAPPED_SIGNIFICANT,
                "mapped_finding_category": "Official mapped finding",
                "mapped_finding_summary": summary,
                "mapped_finding_interpretation": "The checked landslide layer returned a mapped concern for the address point.",
                "mapped_finding_source_name": source_name,
                "mapped_finding_confidence": confidence,
            }
        return {
            "mapped_finding_status": MAPPED_NO_SIGNIFICANT,
            "mapped_finding_category": "Other hazard checked",
            "mapped_finding_summary": "No mapped landslide-zone match found in the checked layers.",
            "mapped_finding_interpretation": "This result only describes the checked layers and does not replace slope, drainage, or geotechnical review.",
            "mapped_finding_source_name": source_name,
            "mapped_finding_confidence": confidence,
        }

    if hazard == "tsunami":
        if status == LOCAL_DIRECT and "outside hazard area" not in terms_text and "no mapped match" not in terms_text:
            summary = next((term for term in terms if "tsunami" in term.lower()), terms[0] if terms else "Official tsunami hazard-area finding.")
            return {
                "mapped_finding_status": MAPPED_SIGNIFICANT,
                "mapped_finding_category": "Official mapped finding",
                "mapped_finding_summary": summary,
                "mapped_finding_interpretation": "The checked tsunami hazard-area layer applies to the address point for evacuation and response planning.",
                "mapped_finding_source_name": source_name,
                "mapped_finding_confidence": confidence,
            }
        return {
            "mapped_finding_status": MAPPED_NO_SIGNIFICANT,
            "mapped_finding_category": "Other hazard checked",
            "mapped_finding_summary": "Outside the official tsunami hazard area.",
            "mapped_finding_interpretation": "This checked-layer result is not a guarantee of safety and does not replace official instructions during an event.",
            "mapped_finding_source_name": source_name,
            "mapped_finding_confidence": confidence,
        }

    return {
        "mapped_finding_status": MAPPED_NO_SIGNIFICANT,
        "mapped_finding_category": "Other hazard checked",
        "mapped_finding_summary": "No mapped match found in the checked layer.",
        "mapped_finding_interpretation": "This result only describes the checked official map layer.",
        "mapped_finding_source_name": source_name,
        "mapped_finding_confidence": confidence,
    }


def _legacy_local_hazard_level(hazard: str, citywide_priority: str, local: Dict, has_citywide_record: bool) -> Tuple[str, str]:
    status = local.get("local_exposure_status")
    terms = " ".join(local.get("source_specific_terminology") or []).lower()
    if not has_citywide_record:
        return UNKNOWN, "No source-backed citywide LHMP record is available, so the local hazard priority is Unknown."
    if status == LOCAL_UNAVAILABLE:
        return UNKNOWN, "Official GIS evidence was unavailable for this hazard, so StayReady does not assign a local hazard priority."
    if citywide_priority == UNKNOWN:
        return UNKNOWN, "The citywide LHMP Probability or Impact value is Unknown, so StayReady does not assign a local hazard priority."
    if status == LOCAL_DIRECT:
        if hazard == "flood":
            if any(token in terms for token in [" ae", "zone ae", " a ", "1%", "special flood hazard", "sfha"]):
                return "High", "FEMA/local flood mapping directly applies and indicates a higher mapped concern."
            if "0.2" in terms or "500" in terms or " x" in terms:
                return _max_rating(citywide_priority, "Medium"), "FEMA/local flood mapping directly applies; the source category is retained without treating it as a personal prediction."
        if hazard == "wildfire":
            if "very high" in terms or "high" in terms:
                return "High", "CAL FIRE/official FHSZ mapping directly applies; severity-zone language is mapped concern, not annual probability."
            if "moderate" in terms:
                return _max_rating(citywide_priority, "Medium"), "CAL FIRE/official FHSZ mapping directly applies; severity-zone language is mapped concern, not annual probability."
        if hazard == "tsunami":
            return _max_rating(citywide_priority, "Medium"), "Official tsunami hazard-area mapping directly applies at the address point."
        return _max_rating(citywide_priority, "Medium"), "Official mapped evidence directly applies at the address point."
    if status == LOCAL_ADDITIONAL:
        if hazard == "earthquake":
            return _max_rating(citywide_priority, "High"), "Official ground-failure or fault-proximity context adds mapped concern while preserving citywide earthquake Probability and Impact."
        return _max_rating(citywide_priority, "Medium"), "Official mapped evidence adds local concern while preserving citywide Probability and Impact."
    return citywide_priority, "No official address-specific layer increased the citywide LHMP hazard priority."


def _rank_sort_key(item: Dict) -> Tuple[int, int, int, int, int]:
    rules = load_priority_data().get("top_four", {})
    priority_order = rules.get("priority_order", {})
    local_order = rules.get("location_evidence_order", {})
    confidence_order = rules.get("confidence_order", {})
    stable_order = rules.get("stable_hazard_order") or SUPPORTED_HAZARDS
    final_value = priority_order.get(item.get("displayed_hazard_level") or item.get("final_rating"), 0)
    citywide_value = priority_order.get(item["calculated_citywide_priority"], 0)
    local_value = local_order.get(item["local_exposure"]["local_exposure_status"], 0)
    confidence_value = confidence_order.get(item.get("confidence"), 0)
    stable_value = len(stable_order) - stable_order.index(item["slug"]) if item["slug"] in stable_order else 0
    evidence_strength = 2 if item["sources_used"] else 0
    return (
        final_value + local_value,
        final_value,
        citywide_value,
        local_value,
        evidence_strength + confidence_value + stable_value,
    )


def build_hazard_priority_results(
    jurisdiction: str,
    address_gis_results,
    *,
    coordinates: Optional[Dict] = None,
    citywide_lhmp_data: Optional[List[Dict]] = None,
    optional_sub_area_match: Optional[Dict] = None,
    display_limit: Optional[int] = None,
) -> List[Dict]:
    methodology = load_priority_data().get("methodology", {})
    method_name = methodology.get("methodology_name", "StayReady's simplified LHMP-based hazard-priority methodology")
    sub_area = _sub_area_context(jurisdiction, optional_sub_area_match, coordinates)
    results = []

    override_records = citywide_lhmp_data or []
    for hazard in SUPPORTED_HAZARDS:
        record = next(
            (
                item for item in override_records
                if _slug(item.get("jurisdiction")) == _slug(jurisdiction)
                and _canonical_hazard(item.get("hazard")) == hazard
            ),
            None,
        ) or _ranking_record(jurisdiction, hazard)

        probability = (record or {}).get("probability") or UNKNOWN
        impact = (record or {}).get("impact") or UNKNOWN
        calculated = calculate_citywide_priority(probability, impact)
        official = (record or {}).get("official_lhmp_priority") or ""
        official_differs = bool(official and calculated != UNKNOWN and official != calculated)
        local = local_exposure_for_hazard(hazard, address_gis_results)
        official_area = official_area_rating_for_hazard(jurisdiction, sub_area["sub_area"], hazard)
        community_context = _community_context_for_hazard(jurisdiction, sub_area["sub_area"], hazard)
        mapped_finding = _mapped_finding_for_hazard(hazard, local)
        fallback_allowed = not (_slug(jurisdiction) == "oakland" and sub_area["sub_area"] == UNKNOWN)
        fallback_rating = calculated if calculated != UNKNOWN and fallback_allowed else UNKNOWN
        if official_area["valid_scenario_count"]:
            displayed_level = official_area["combined_area_rating"]
            displayed_basis = official_area["combination_explanation"]
            displayed_status = "verified"
        elif official_area["displayed_level_status"] == "unsupported":
            displayed_level = UNKNOWN
            displayed_basis = official_area["unsupported_reason"]
            displayed_status = "unsupported"
        elif fallback_rating != UNKNOWN:
            displayed_level = fallback_rating
            displayed_basis = f"No official LHMP area scenario rating is available; using documented StayReady Probability + Impact fallback ({probability} Probability and {impact} Impact)."
            displayed_status = "fallback"
        else:
            displayed_level = UNKNOWN
            displayed_basis = "No official LHMP area scenario rating is available for the matched plan area, and no defensible fallback is available."
            displayed_status = "unavailable"
        sources = official_area["sources"] or ([_source_payload(record)] if record else [])
        confidence = (record or {}).get("confidence") or "Low"
        if official_area["valid_scenario_count"]:
            confidence = "High" if "adopted" in official_area["source_statuses"] else "Medium"
        if not record:
            why = "No structured Probability and Impact record is available for this jurisdiction and hazard, so citywide hazard priority is Unknown."
        elif calculated == UNKNOWN:
            why = "Probability or Impact is Unknown, so StayReady does not calculate a citywide hazard priority."
        else:
            why = f"{method_name} calculates {calculated} from {probability} Probability and {impact} Impact."
        if official_differs:
            why += f" The official LHMP priority is {official}; it is preserved separately because the official method uses additional factors."

        ranking_basis = [
            f"Displayed hazard level: {displayed_level}",
            f"Oakland LHMP area rating: {official_area['combined_area_rating']}",
            f"Address-specific maps: {local['local_exposure_status']}",
            f"Confidence: {confidence}",
        ]
        if local["local_exposure_status"] in {LOCAL_DIRECT, LOCAL_ADDITIONAL}:
            ranking_basis.append("Official GIS evidence is shown separately and does not automatically replace the Oakland LHMP area rating.")
        if local["local_exposure_status"] == LOCAL_UNAVAILABLE:
            ranking_basis.append("Failed GIS data does not erase a valid LHMP area rating.")
        if not record:
            ranking_basis.append("Missing source-backed citywide values were not guessed.")
        if community_context["status"] == "Available":
            ranking_basis.append("Sub-area EPC/exposure context is shown only as community context.")

        results.append({
            "hazard": _hazard_label(hazard),
            "slug": hazard,
            "jurisdiction": _slug(jurisdiction),
            "methodology": method_name,
            "probability": probability if probability in {"Low", "Medium", "High"} else UNKNOWN,
            "probability_basis": (record or {}).get("probability_basis") or "No source-backed Probability value is available.",
            "impact": impact if impact in {"Low", "Medium", "High"} else UNKNOWN,
            "impact_basis": (record or {}).get("impact_basis") or "No source-backed Impact value is available.",
            "calculated_citywide_priority": calculated,
            "official_lhmp_priority": official or UNKNOWN,
            "official_lhmp_priority_basis": (record or {}).get("official_lhmp_priority_basis") or "",
            "official_priority_differs": official_differs,
            "official_lhmp_area_rating": official_area["official_lhmp_area_rating"],
            "stayready_fallback_rating": fallback_rating,
            "scenario_ratings": official_area["scenario_ratings"],
            "valid_scenario_count": official_area["valid_scenario_count"],
            "excluded_scenario_count": official_area["excluded_scenario_count"],
            "combined_scenario_score_internal": official_area["combined_scenario_score_internal"],
            "combined_area_rating": official_area["combined_area_rating"],
            "combination_method": official_area["combination_method"],
            "combination_explanation": official_area["combination_explanation"],
            "displayed_hazard_level": displayed_level,
            "displayed_level": displayed_level,
            "displayed_level_status": displayed_status,
            "displayed_level_basis": displayed_basis,
            "source_records": official_area["unsupported_source_records"] if displayed_status == "unsupported" else official_area["scenario_ratings"],
            "successful_queries": local["successful_layers"],
            "failed_queries": local["unavailable_layers"],
            "fallback_used": displayed_status == "fallback",
            **mapped_finding,
            "successful_sources": _dedupe(local["official_layers_checked"] + local["successful_layers"]),
            "failed_sources": local["unavailable_layers"],
            "area_rating_status": displayed_status,
            "final_rating": displayed_level,
            "lhmp_rating": calculated,
            "why_this_rating": f"{displayed_basis} {_local_evidence_summary(local)}",
            "ranking_basis": ranking_basis,
            "local_exposure": local,
            "local_exposure_status": local["local_exposure_status"],
            "physical_exposure_status": local["local_exposure_status"],
            "official_zone_category": _official_zone_category(local),
            "local_hazard_level": displayed_level,
            "local_hazard_level_basis": displayed_basis,
            "local_evidence_summary": _local_evidence_summary(local),
            "address_specific_notes": local["local_exposure_basis"],
            "sub_area_context": sub_area,
            "plan_area": sub_area["sub_area"],
            "boundary_warning": sub_area.get("boundary_warning") or "",
            "boundary_distance_m": sub_area.get("boundary_distance_m"),
            "community_context": community_context,
            "area_context": sub_area["sub_area"] if sub_area["sub_area"] != UNKNOWN else "Not currently available",
            "sources_used": sources,
            "source_document": (record or {}).get("source_document") or "",
            "source_page": (record or {}).get("source_page"),
            "source_table": (record or {}).get("source_table") or "",
            "publication_date": (record or {}).get("publication_date") or "",
            "document_status": (record or {}).get("document_status") or "",
            "confidence": confidence if confidence in {"High", "Medium", "Low"} else "Low",
            "limitations": _dedupe(
                ((record or {}).get("limitations") or [])
                + local["limitations"]
                + sub_area["sub_area_limitations"]
                + community_context["limitations"]
                + ["This is not an exact personal risk or property-damage prediction."]
            ),
            "evidence_summary": f"{displayed_basis} {_local_evidence_summary(local)}",
        })

    ranked = sorted(results, key=_rank_sort_key, reverse=True)
    for index, item in enumerate(ranked, start=1):
        item["display_rank"] = index
        item["display_group"] = {
            "High": "High hazards",
            "Medium": "Medium hazards",
            "Low": "Lower hazards identified",
            UNKNOWN: "Information unavailable",
        }.get(item.get("displayed_hazard_level"), "Information unavailable")
        item["is_top_four"] = item.get("displayed_hazard_level") in {"High", "Medium"}
    return ranked


def rank_hazards_for_risk_summary(
    city: str,
    address_gis_results,
    *,
    limit: int = 4,
    coordinates: Optional[Dict] = None,
) -> List[Dict]:
    """Compatibility wrapper for the Risk Summary page.

    Returns all supported hazards. `is_top_four` is retained as a compatibility
    flag for prominent High/Medium hazards, not as a fixed display limit.
    """
    return build_hazard_priority_results(city, address_gis_results, coordinates=coordinates, display_limit=None)
