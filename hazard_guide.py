from typing import Dict, List


HAZARD_ORDER = ("earthquake", "flood", "wildfire")

HAZARD_GUIDES = {
    "earthquake": {
        "name": "Earthquake",
        "icon": "fa-house-crack",
        "definition": (
            "Earthquakes can cause strong shaking, falling objects, utility disruption, "
            "building damage, and recovery challenges across Alameda County."
        ),
        "local_relevance": (
            "Alameda County is part of an active earthquake region. Shaking can affect every "
            "community, while mapped fault, liquefaction, and landslide conditions vary by location."
        ),
        "check_summary": (
            "Regional shaking context, nearby mapped faults, and three California Geological "
            "Survey zone layers."
        ),
        "source_ids": (
            "usgs_faults",
            "cgs_alquist_priolo",
            "cgs_liquefaction",
            "cgs_earthquake_landslide",
            "ready_earthquakes",
        ),
    },
    "flood": {
        "name": "Flood",
        "icon": "fa-house-flood-water",
        "definition": (
            "Flooding can damage homes and belongings, interrupt roads and utilities, and "
            "create health and recovery challenges."
        ),
        "local_relevance": (
            "Creeks, shorelines, drainage systems, low-lying areas, and severe storms can create "
            "different flood concerns across Alameda County."
        ),
        "check_summary": "Regional flood preparedness context and the FEMA flood layer at an address point.",
        "source_ids": ("fema_nfhl", "ready_floods"),
    },
    "wildfire": {
        "name": "Wildfire",
        "icon": "fa-fire-flame-curved",
        "definition": (
            "Wildfire can affect communities through fire, smoke, evacuation, road closures, "
            "power disruption, and difficult recovery."
        ),
        "local_relevance": (
            "Hillside vegetation, dry weather, wind, smoke, and constrained evacuation routes "
            "make wildfire preparedness relevant across Alameda County."
        ),
        "check_summary": (
            "Regional wildfire, smoke, and evacuation context plus the CAL FIRE Fire Hazard "
            "Severity Zone layer at an address point."
        ),
        "source_ids": ("calfire_fhsz", "berkeley_fire_evacuation"),
    },
}

SECONDARY_HAZARDS = {
    "landslide": "Landslide",
    "tsunami": "Tsunami",
    "dam_failure": "Dam failure",
    "drought": "Drought",
    "extreme_heat": "Extreme heat",
    "poor_air_quality": "Smoke and poor air quality",
    "utility_disruption": "Power and utility disruption",
    "severe_weather": "Severe storms",
    "sea_level_rise": "Sea level rise",
}

CHECK_DEFINITIONS = {
    "regional_earthquake": {
        "name": "Regional earthquake shaking context",
        "plain_meaning": "Strong earthquake shaking is a regional preparedness concern across Alameda County.",
        "does_not_mean": "Regional context does not predict when an earthquake will happen or how a specific building will perform.",
        "source_ids": ("ready_earthquakes", "alameda_county_lhmp"),
    },
    "fault_proximity": {
        "name": "Nearby fault context",
        "plain_meaning": "StayReady checks whether a loaded mapped fault trace is within about two kilometres of the address.",
        "does_not_mean": "Distance to a mapped fault is not regulatory zone membership or a building-damage prediction.",
        "source_ids": ("usgs_faults",),
    },
    "cgs_alquist_priolo_remote": {
        "name": "CGS Alquist-Priolo fault rupture zones",
        "plain_meaning": "This official layer maps regulatory zones for potential surface fault rupture.",
        "does_not_mean": "A match is not a general shaking or building-safety determination.",
        "source_ids": ("cgs_alquist_priolo",),
    },
    "cgs_liquefaction_remote": {
        "name": "CGS liquefaction zones",
        "plain_meaning": "This official layer maps areas where loose, wet soil may lose strength during strong shaking.",
        "does_not_mean": "A match does not guarantee liquefaction or property damage.",
        "source_ids": ("cgs_liquefaction",),
    },
    "cgs_earthquake_landslide_remote": {
        "name": "CGS earthquake-induced landslide zones",
        "plain_meaning": "This official layer maps areas where strong shaking may trigger landslides.",
        "does_not_mean": "This is not a general landslide forecast or site-specific engineering assessment.",
        "source_ids": ("cgs_earthquake_landslide",),
    },
    "regional_flood": {
        "name": "Regional flood preparedness context",
        "plain_meaning": "Flooding and storm-related disruption remain regional preparedness concerns.",
        "does_not_mean": "Regional context does not establish flood-zone membership for an address.",
        "source_ids": ("alameda_county_lhmp", "ready_floods"),
    },
    "fema_nfhl_local": {
        "name": "FEMA flood-zone address check",
        "plain_meaning": "StayReady checks the saved address point against its provisional local FEMA flood-layer snapshot.",
        "does_not_mean": "This is not a parcel determination and does not capture every drainage or storm condition.",
        "source_ids": ("fema_nfhl",),
    },
    "regional_wildfire": {
        "name": "Regional wildfire, smoke, and evacuation context",
        "plain_meaning": "Wildfire can affect residents through fire, smoke, evacuation, and service disruption.",
        "does_not_mean": "Regional context does not establish Fire Hazard Severity Zone membership.",
        "source_ids": ("alameda_county_lhmp",),
    },
    "calfire_fhsz_local": {
        "name": "CAL FIRE severity-zone address check",
        "plain_meaning": "StayReady checks the saved address point against its provisional local CAL FIRE layer snapshot.",
        "does_not_mean": "The layer maps fire hazard, not individual building safety, smoke exposure, or evacuation difficulty.",
        "source_ids": ("calfire_fhsz",),
    },
}


def _source_status(source: Dict) -> str:
    if source.get("review_status") == "reviewed" and source.get("confidence") == "source_backed":
        return "Reviewed official guidance"
    if source.get("review_status") in {"reviewed", "draft_reviewed"}:
        return "Official source; StayReady integration under review"
    return "Source review in progress"


def _source_support(source: Dict) -> str:
    return source.get("use_in_app") or source.get("notes") or "Official hazard information and guidance."


def _public_sources(hazard: Dict, source_ids=()) -> List[Dict]:
    requested = set(source_ids)
    sources = []
    seen = set()
    for source in hazard.get("sources") or []:
        source_id = source.get("source_id")
        if requested and source_id not in requested:
            continue
        key = source_id or source.get("url") or source.get("name")
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append({
            "source_id": source_id,
            "name": source.get("name") or "Official source",
            "agency": source.get("agency") or "",
            "url": source.get("url") or "",
            "supports": _source_support(source),
            "status_label": _source_status(source),
        })
    return sources


def _has_location(location_context: Dict) -> bool:
    return bool(location_context.get("address") or location_context.get("zip_code"))


def _has_address(location_context: Dict) -> bool:
    return bool(location_context.get("has_precise_location"))


def _status_from_evidence(evidence: Dict, has_address: bool) -> str:
    if not has_address:
        return "Add an address to check this layer"
    if not evidence:
        return "Data unavailable"
    if evidence.get("data_available") is False or evidence.get("available") is False:
        return "Data unavailable"
    if evidence.get("matched") is True:
        return "Mapped match found"
    if evidence.get("checked") is True or evidence.get("matched") is False:
        return "Checked: no address-level match found"
    return "Data unavailable"


def _status_tone(status: str) -> str:
    if status == "Mapped match found":
        return "match"
    if status == "Nearby hazard context found":
        return "nearby"
    if status == "Checked: no address-level match found":
        return "checked"
    if status == "Regional preparedness context":
        return "regional"
    if status == "Add an address to check this layer":
        return "address"
    return "unavailable"


def _evidence_by_id(hazard: Dict) -> Dict[str, Dict]:
    evidence = {}
    for item in hazard.get("additional_geospatial_evidence") or []:
        dataset_id = item.get("dataset_id")
        if dataset_id:
            evidence[dataset_id] = item
    for item in hazard.get("normalized_mapped_evidence") or []:
        dataset_id = item.get("dataset_id") or item.get("source_id")
        if dataset_id:
            evidence.setdefault(dataset_id, item)
    return evidence


def _check_row(key: str, status: str, hazard: Dict, evidence: Dict = None) -> Dict:
    definition = CHECK_DEFINITIONS[key]
    evidence = evidence or {}
    sources = _public_sources(hazard, definition["source_ids"])
    limitations = list(evidence.get("limitations") or [])
    return {
        "key": key,
        "name": definition["name"],
        "status": status,
        "tone": _status_tone(status),
        "plain_meaning": evidence.get("plain_definition") or definition["plain_meaning"],
        "what_this_means": evidence.get("what_this_means") or definition["plain_meaning"],
        "does_not_mean": evidence.get("what_this_does_not_mean") or definition["does_not_mean"],
        "source_summary": evidence.get("source_summary") or (sources[0]["supports"] if sources else ""),
        "sources": sources,
        "limitations": limitations,
        "provisional": any(source["status_label"].endswith("under review") for source in sources),
    }


def _earthquake_checks(hazard: Dict, location_context: Dict) -> List[Dict]:
    has_address = _has_address(location_context)
    evidence = _evidence_by_id(hazard)
    fault_near = hazard.get("match_type") in {"near_fault", "fault_proximity_context"} and bool(
        hazard.get("matched_layers")
    )
    fault_status = (
        "Add an address to check this layer"
        if not has_address
        else "Nearby hazard context found"
        if fault_near
        else "Checked: no address-level match found"
    )
    rows = [
        _check_row("regional_earthquake", "Regional preparedness context", hazard),
        _check_row("fault_proximity", fault_status, hazard, hazard.get("geospatial_evidence") or {}),
    ]
    for dataset_id in (
        "cgs_alquist_priolo_remote",
        "cgs_liquefaction_remote",
        "cgs_earthquake_landslide_remote",
    ):
        item = evidence.get(dataset_id, {})
        rows.append(_check_row(dataset_id, _status_from_evidence(item, has_address), hazard, item))
    return rows


def _single_layer_check(hazard: Dict, location_context: Dict, regional_key: str, layer_key: str) -> List[Dict]:
    has_address = _has_address(location_context)
    layer_evidence = hazard.get("geospatial_evidence") or {}
    if not layer_evidence:
        normalized = hazard.get("normalized_mapped_evidence") or []
        layer_evidence = normalized[0] if normalized else {}
    status = _status_from_evidence(layer_evidence, has_address)
    return [
        _check_row(regional_key, "Regional preparedness context", hazard),
        _check_row(layer_key, status, hazard, layer_evidence),
    ]


def _checks_for_hazard(hazard: Dict, location_context: Dict) -> List[Dict]:
    slug = hazard.get("slug")
    if slug == "earthquake":
        return _earthquake_checks(hazard, location_context)
    if slug == "flood":
        return _single_layer_check(hazard, location_context, "regional_flood", "fema_nfhl_local")
    return _single_layer_check(hazard, location_context, "regional_wildfire", "calfire_fhsz_local")


def _under_review_checks(hazard: Dict, location_context: Dict) -> List[Dict]:
    has_address = _has_address(location_context)
    rows = [{
        "key": "regional_context",
        "name": "Regional preparedness context",
        "status": "Regional preparedness context",
        "tone": "regional",
        "plain_meaning": (
            "This hazard appears in local or regional preparedness planning, but StayReady "
            "does not yet provide a complete address-level guide for it."
        ),
        "what_this_means": hazard.get("summary") or hazard.get("why_shown") or "",
        "does_not_mean": "This planning context is not an address-level hazard determination.",
        "source_summary": "",
        "sources": _public_sources(hazard),
        "limitations": [],
        "provisional": True,
    }]
    for index, evidence in enumerate(hazard.get("additional_geospatial_evidence") or []):
        status = _status_from_evidence(evidence, has_address)
        rows.append({
            "key": evidence.get("dataset_id") or f"mapped_check_{index}",
            "name": evidence.get("dataset_name") or "Official mapped layer",
            "status": status,
            "tone": _status_tone(status),
            "plain_meaning": evidence.get("plain_definition") or "StayReady checks an official mapped layer when an address is available.",
            "what_this_means": evidence.get("result_label") or evidence.get("what_this_means") or "",
            "does_not_mean": evidence.get("what_this_does_not_mean") or "This is not a property-specific determination.",
            "source_summary": evidence.get("source_summary") or "",
            "sources": _public_sources(hazard, (evidence.get("source_id"),)),
            "limitations": list(evidence.get("limitations") or []),
            "provisional": True,
        })
    return rows


def _location_context(hazards: List[Dict], location_context: Dict) -> Dict:
    has_location = _has_location(location_context)
    if not has_location:
        return {
            "active": False,
            "display_name": "",
            "message": "Add an Alameda County address to check available map layers for your location.",
        }
    matched = []
    nearby = []
    for hazard in hazards:
        checks = (
            _checks_for_hazard(hazard, location_context)
            if hazard.get("slug") in HAZARD_GUIDES
            else _under_review_checks(hazard, location_context)
        )
        for check in checks:
            if check["status"] == "Mapped match found":
                matched.append(check["name"])
            elif check["status"] == "Nearby hazard context found":
                nearby.append(check["name"])
    if matched:
        message = f"StayReady found address-level mapped context for {len(matched)} checked layer"
        message += "" if len(matched) == 1 else "s"
        message += "."
    elif nearby:
        message = "StayReady found nearby hazard context for this location."
    elif _has_address(location_context):
        message = "StayReady checked available address-level layers for this location."
    else:
        message = "A ZIP provides regional context; add an address for available map-layer checks."
    return {
        "active": True,
        "has_address": _has_address(location_context),
        "display_name": location_context.get("display_name") or "",
        "message": message,
    }


def _library_entry(hazard: Dict, location_context: Dict) -> Dict:
    slug = hazard.get("slug")
    guide = HAZARD_GUIDES[slug]
    sources = _public_sources(hazard, guide["source_ids"])
    return {
        "slug": slug,
        **guide,
        "sources": sources[:5],
        "has_location": _has_location(location_context),
    }


def _secondary_hazards(items: List[Dict], has_location: bool) -> List[Dict]:
    if items:
        return [{
            "hazard_id": item.get("hazard_id"),
            "label": SECONDARY_HAZARDS.get(item.get("hazard_id"), item.get("label")),
            "message": (
                "This hazard appears in local or regional preparedness planning, but StayReady "
                "does not yet provide a full address-level guide for it."
            ),
            "plan_name": item.get("plan_name") or "",
            "plan_url": item.get("plan_url") or "",
        } for item in items]
    return [{
        "hazard_id": hazard_id,
        "label": label,
        "message": "StayReady is still reviewing local source coverage for this hazard.",
        "plan_name": "",
        "plan_url": "",
    } for hazard_id, label in SECONDARY_HAZARDS.items()]


def build_hazard_library(
    hazards: List[Dict],
    location_context: Dict,
    additional_hazards: List[Dict] = None,
) -> Dict:
    by_slug = {hazard.get("slug"): hazard for hazard in hazards}
    ordered = [by_slug[slug] for slug in HAZARD_ORDER if slug in by_slug]
    return {
        "hazards": [_library_entry(hazard, location_context) for hazard in ordered],
        "location": _location_context(ordered, location_context),
        "secondary_hazards": _secondary_hazards(
            additional_hazards or [],
            _has_location(location_context),
        ),
    }


def build_hazard_detail_guide(hazard: Dict, location_context: Dict) -> Dict:
    slug = hazard.get("slug")
    guide = HAZARD_GUIDES.get(slug)
    under_review = guide is None
    if under_review:
        guide = {
            "name": hazard.get("name") or hazard.get("label") or slug.replace("-", " ").title(),
            "icon": "fa-triangle-exclamation",
            "definition": hazard.get("summary") or (
                "This hazard appears in local or regional preparedness planning. "
                "StayReady is still reviewing source coverage for a complete guide."
            ),
            "local_relevance": hazard.get("summary") or hazard.get("why_shown") or (
                "StayReady is still reviewing local source coverage for this hazard."
            ),
            "check_summary": "Regional planning context and any registered official layer currently available.",
            "source_ids": (),
        }
    city_context = (hazard.get("specialized_guidance") or {}).get("city_context") or []
    return {
        "slug": slug,
        **guide,
        "has_location": _has_location(location_context),
        "has_address": _has_address(location_context),
        "location": _location_context([hazard], location_context),
        "city_context": city_context[:2],
        "checks": (
            _under_review_checks(hazard, location_context)
            if under_review
            else _checks_for_hazard(hazard, location_context)
        ),
        "under_review": under_review,
        "actions": list(hazard.get("action_steps") or []),
        "recovery_questions": list(hazard.get("recovery_questions") or []),
        "sources": _public_sources(hazard),
        "limitations": list(hazard.get("limitations") or []),
    }
