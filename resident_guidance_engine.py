import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

from action_library_service import select_actions
from hazard_priority import rank_hazards_for_risk_summary
from risk_summary_view_model import build_canonical_risk_summary
from pydantic_models import LHMPLocationFact


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWED_STATUSES = {"reviewed", "draft_reviewed"}
EVIDENCE_TIER_LABELS = {
    "address_specific": "Address-specific source",
    "area_based": "Area-based source",
    "citywide": "Citywide source",
    "general": "General guidance",
}


def _load_json(filename: str, default):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _norm(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _dedupe(items: List[str]) -> List[str]:
    output = []
    seen = set()
    for item in items or []:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _listify(value) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@lru_cache(maxsize=1)
def load_local_hazard_facts() -> List[Dict]:
    location_facts = [
        LHMPLocationFact(**item).model_dump()
        for item in _load_json("lhmp_location_facts.json", [])
    ]
    return _load_json("hazard_facts.json", []) + location_facts


def load_supabase_hazard_facts(city: Optional[str] = None, hazard: Optional[str] = None) -> List[Dict]:
    try:
        from supabase_repository import fetch_hazard_facts
    except Exception:
        return []

    facts = fetch_hazard_facts(city=city, hazard=hazard)
    return facts or []


def load_hazard_facts(city: Optional[str] = None, hazard: Optional[str] = None) -> List[Dict]:
    local_facts = load_local_hazard_facts()
    supabase_facts = load_supabase_hazard_facts(city=city, hazard=hazard)
    combined = list(local_facts)
    existing = {item.get("id") for item in combined if item.get("id")}
    for fact in supabase_facts:
        fact_id = fact.get("id")
        if fact_id and fact_id in existing:
            continue
        combined.append(fact)
        if fact_id:
            existing.add(fact_id)
    return combined


def get_household_context(session_data: Dict) -> Dict:
    tags = set(session_data.get("household_tags") or [])
    special_needs = (session_data.get("special_needs") or "").strip()
    special_lower = special_needs.lower()

    inferred_keywords = {
        "medical": ("medication", "medicine", "medical", "asthma", "diabetes", "oxygen", "device", "cpap", "prescription"),
        "access_needs": ("wheelchair", "walker", "mobility", "disabled", "disability", "accessible", "caregiver"),
        "children": ("child", "children", "baby", "infant", "school"),
        "older_adults": ("elder", "senior", "older", "aging"),
        "pets": ("pet", "dog", "cat", "animal"),
        "no_car": ("no car", "transit", "bus", "bart", "ride"),
    }
    for tag, keywords in inferred_keywords.items():
        if any(re.search(rf"\b{re.escape(keyword)}\b", special_lower) for keyword in keywords):
            tags.add(tag)

    household = (session_data.get("household") or "").strip()
    preparedness = (session_data.get("preparedness") or "").strip()

    labels = {
        "medical": "medical or medication needs",
        "pets": "pets",
        "no_car": "no reliable car access",
        "renter": "renter",
        "homeowner": "homeowner",
        "children": "children or school continuity needs",
        "older_adults": "older adults",
        "access_needs": "disability, mobility, or access needs",
    }

    return {
        "household_size": household,
        "preparedness": preparedness,
        "special_needs": special_needs,
        "tags": sorted(tags),
        "tag_labels": [labels[tag] for tag in sorted(tags) if tag in labels],
        "has_context": bool(household or preparedness or special_needs or tags),
    }


def _action_payloads(actions) -> List[Dict]:
    return [action.model_dump(mode="json") for action in actions]


def household_guidance(
    household_context: Dict,
    *,
    city: str = "",
    county: str = "",
    hazards: Optional[List[str]] = None,
) -> List[Dict]:
    return _action_payloads(select_actions(
        hazards=hazards or ["all"],
        household_factors=household_context.get("tags") or [],
        time_buckets=["today", "this_week", "this_month", "before", "recovery"],
        city=city,
        county=county,
        trigger_types=["household"],
        limit=8,
    ))


def _coordinate_rule_matches(rule: Dict, location_context: Dict) -> bool:
    required = {"min_lat", "max_lat", "min_lon", "max_lon"}
    if not rule or set(rule) != required:
        return False
    try:
        lat = float(location_context.get("location_result", {}).get("lat"))
        lon = float(location_context.get("location_result", {}).get("lon"))
    except (TypeError, ValueError):
        return False

    return (
        rule["min_lat"] <= lat <= rule["max_lat"]
        and rule["min_lon"] <= lon <= rule["max_lon"]
    )


def _normalized_place(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _named_area_match(alias: str, location_context: Dict) -> Optional[Dict]:
    normalized_alias = _normalized_place(alias)
    if not normalized_alias:
        return None

    location_result = location_context.get("location_result") or {}
    structured_values = [
        location_result.get("neighborhood"),
        location_result.get("suburb"),
        location_result.get("district"),
    ]
    for value in structured_values:
        if value and _normalized_place(str(value)) == normalized_alias:
            return {
                "method": "structured_named_area",
                "reason": f"The geocoder identified the address area as “{alias}”.",
                "rank": 3,
            }

    formatted_values = [
        location_context.get("address"),
        location_context.get("display_name"),
        location_result.get("formatted_address"),
    ]
    components = {
        _normalized_place(component)
        for value in formatted_values
        for component in str(value or "").split(",")
        if _normalized_place(component)
    }
    if normalized_alias in components:
        return {
            "method": "resolved_address_component",
            "reason": f"The resolved address contains the reviewed named area “{alias}”.",
            "rank": 2,
        }

    # Multi-word street, corridor, park, and district names may appear inside a
    # resolved address component. Single generic words require an exact component.
    if len(normalized_alias.split()) >= 2:
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])"
        if any(re.search(pattern, _normalized_place(str(value or ""))) for value in formatted_values):
            return {
                "method": "reviewed_address_phrase",
                "reason": f"The resolved address includes the reviewed place name “{alias}”.",
                "rank": 1,
            }
    return None


def _fact_location_match(fact: Dict, location_context: Dict) -> Optional[Dict]:
    if fact.get("evidence_tier") != "area_based":
        return None

    for alias in fact.get("location_aliases") or []:
        match = _named_area_match(alias, location_context)
        if match:
            return match

    if _coordinate_rule_matches(fact.get("coordinate_rule") or {}, location_context):
        return {
            "method": "reviewed_coordinate_bounds",
            "reason": (
                f"The geocoded point falls within reviewed bounded geography for "
                f"{fact.get('location_cue') or 'the named LHMP area'}."
            ),
            "rank": 2,
        }
    return None


def _match_facts(city: str, hazard: str, location_context: Optional[Dict] = None) -> Dict[str, List[Dict]]:
    city_key = _norm(city)
    hazard_key = _norm(hazard)
    area = []
    citywide = []
    county = []
    needs_review = False
    location_context = location_context or {}

    for fact in load_hazard_facts(city=city, hazard=hazard):
        fact_hazard = _norm(fact.get("hazard"))
        if fact_hazard != hazard_key:
            continue
        review_status = fact.get("review_status", "")
        jurisdiction_key = _norm(fact.get("jurisdiction"))
        applies_to = {_norm(value) for value in fact.get("applies_to_jurisdictions") or []}
        is_local = jurisdiction_key == city_key or city_key in applies_to
        if is_local and review_status in REVIEWED_STATUSES:
            if fact.get("evidence_tier") == "area_based":
                match = _fact_location_match(fact, location_context)
                if match:
                    area.append({
                        **fact,
                        "location_match_method": match["method"],
                        "location_match_reason": match["reason"],
                        "location_match_rank": match["rank"],
                    })
            else:
                citywide.append(fact)
        elif is_local:
            needs_review = True
        elif jurisdiction_key in {"alameda_county", "unincorporated_alameda_county"} and review_status in REVIEWED_STATUSES:
            county.append(fact)

    ambiguous_area_matches = []
    if len(area) > 1:
        best_rank = max(item.get("location_match_rank", 0) for item in area)
        best_matches = [item for item in area if item.get("location_match_rank", 0) == best_rank]
        if len(best_matches) == 1:
            area = best_matches
        else:
            ambiguous_area_matches = area
            area = []

    return {
        "area": area,
        "ambiguous_area_matches": ambiguous_area_matches,
        "citywide": citywide,
        "county": county,
        "needs_review": needs_review,
    }


def _guidance_from_facts(facts: List[Dict], field: str) -> List[str]:
    values = []
    for fact in facts:
        values.extend(_listify(fact.get(field)))
    return _dedupe(values)


def _source_rows(hazard: Dict, facts: List[Dict]) -> List[Dict]:
    rows = []
    for fact in facts:
        if fact.get("source_name") or fact.get("source_url"):
            rows.append({
                "name": fact.get("source_name") or "Source",
                "url": fact.get("source_url") or "",
                "basis": (fact.get("evidence_tier") or fact.get("evidence_type", "")).replace("_", " ").title(),
                "review_status": fact.get("review_status", "").replace("_", " ").title(),
                "page": fact.get("source_page"),
                "document": fact.get("source_document", ""),
            })
    for source in hazard.get("sources", [])[:4]:
        rows.append({
            "name": source.get("name") or source.get("agency") or "Source",
            "url": source.get("url") or "",
            "basis": source.get("claim_type", source.get("source_type", "")).replace("_", " ").title(),
            "review_status": source.get("review_status", "").replace("_", " ").title(),
            "page": None,
            "document": "",
        })

    deduped = []
    seen = set()
    for row in rows:
        key = ((row.get("url") or "").lower(), (row.get("name") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:5]


def _what_was_checked(hazard: Dict) -> Dict:
    slug = hazard.get("slug") or hazard.get("hazard_id")
    scope_label = hazard.get("scope_label") or "County fallback"
    data_status = hazard.get("data_status") or ""
    checked = []
    not_checked = []

    if hazard.get("scope") == "address_level":
        if slug == "flood":
            checked.append("Saved address point checked against the loaded FEMA flood layer.")
        elif slug == "wildfire":
            checked.append("Saved address point checked against the loaded CAL FIRE fire hazard layer.")
        elif slug == "earthquake":
            checked.append("Saved address point checked for proximity to loaded mapped fault lines; this is not hazard-zone membership.")
    else:
        checked.append(f"{scope_label} used for this result.")

    if data_status == "data_unavailable":
        not_checked.append("Official layer unavailable — not checked.")
    if data_status == "not_in_layer":
        checked.append(
            "This layer did not find an address-level match. "
            "This is informational and does not establish overall location risk."
        )
    if slug == "flood" and hazard.get("scope") != "address_level":
        not_checked.append("FEMA address-point flood overlay was not completed for this result.")
    if slug == "wildfire" and hazard.get("scope") != "address_level":
        not_checked.append("Address-level fire hazard zone membership is not available for this result.")
    if slug == "earthquake":
        cgs_checks = hazard.get("additional_geospatial_evidence") or []
        if not cgs_checks:
            not_checked.append("CGS regulatory-zone layers were not checked.")
        for evidence in cgs_checks:
            dataset_name = evidence.get("dataset_name") or "CGS mapped layer"
            if evidence.get("checked"):
                checked.append(
                    f"{dataset_name}: {evidence.get('status_label') or 'Official source checked'}."
                )
            else:
                not_checked.append(f"{dataset_name}: official layer unavailable — not checked.")
        not_checked.append("Building retrofit, structural condition, and parcel-level seismic risk were not checked.")
    if slug in {"flood", "wildfire"}:
        not_checked.append("Parcel-level property conditions and building-specific vulnerability were not checked.")

    return {"checked": _dedupe(checked), "not_checked": _dedupe(not_checked)}


def _selected_actions_for_phase(
    hazard: str,
    phase: str,
    location_context: Dict,
    household_context: Dict,
    limit: int,
) -> List[Dict]:
    trigger_types = ["general", "hazard_result", "location"]
    if phase == "recovery":
        trigger_types.append("household")
    return _action_payloads(select_actions(
        hazards=[hazard],
        household_factors=household_context.get("tags") or [],
        time_buckets=[phase],
        city=location_context.get("city") or "",
        county=location_context.get("county") or "",
        trigger_types=trigger_types,
        limit=limit,
    ))


def _hazard_plan(hazard: Dict, location_context: Dict, household_context: Dict, order: int) -> Dict:
    city = location_context.get("city") or ""
    slug = hazard.get("slug") or hazard.get("hazard_id")
    fact_matches = _match_facts(city, slug, location_context)
    local_facts = fact_matches["area"] + fact_matches["citywide"]
    selected_facts = local_facts or fact_matches["county"]
    fallback_facts = fact_matches["county"] if local_facts else []
    specialized = hazard.get("specialized_guidance") or {}
    guidance_source_status = specialized.get("guidance_source_status")
    has_local_guidance = guidance_source_status == "local_reviewed"
    local_status = "local_reviewed" if local_facts or has_local_guidance else "county_fallback"
    if not local_facts and city and not has_local_guidance:
        local_status = "local_not_reviewed"

    fact_meaning = _dedupe([fact.get("resident_meaning", "") for fact in selected_facts])
    fact_cues = _dedupe([fact.get("location_cue", "") for fact in selected_facts])
    checked_payload = _what_was_checked(hazard)

    before = _selected_actions_for_phase(slug, "before", location_context, household_context, 5)
    during = _selected_actions_for_phase(slug, "during", location_context, household_context, 4)
    after = _selected_actions_for_phase(slug, "after", location_context, household_context, 4)
    recovery = _selected_actions_for_phase(slug, "recovery", location_context, household_context, 6)

    city_context = specialized.get("city_context") or []
    location_specific_context = specialized.get("location_specific_context") or []
    why_parts = [hazard.get("why_shown", "")]
    why_parts.extend(fact_meaning[:1])
    why_parts.extend(fact_cues[:1])
    if has_local_guidance and not local_facts:
        why_parts.extend(_dedupe(location_specific_context + city_context)[:2])
    why = " ".join(_dedupe(why_parts))
    if not local_facts and city and not has_local_guidance:
        why += f" Local plan facts for {city} are not fully reviewed in the structured facts layer yet, so county-level guidance is included."

    if fact_matches["area"]:
        evidence_tier = "area_based"
    elif local_facts or has_local_guidance:
        evidence_tier = "citywide"
    else:
        evidence_tier = "general"
    if hazard.get("match_type") in {"near_fault", "fault_proximity_context"}:
        primary_evidence_badge = "Fault proximity context"
    elif hazard.get("scope") == "address_level":
        primary_evidence_badge = "Address point checked"
    else:
        primary_evidence_badge = hazard.get("scope_label") or "County fallback"
    evidence_badges = [
        primary_evidence_badge,
        hazard.get("data_status_label") or "Needs review",
        EVIDENCE_TIER_LABELS[evidence_tier],
    ]
    public_claim_status = hazard.get("public_claim_status")
    if public_claim_status:
        evidence_badges.append(public_claim_status.replace("_", " ").title())
    location_matches = [
        {
            "label": fact.get("location_cue") or ", ".join(fact.get("named_areas") or []),
            "method": fact.get("location_match_method"),
            "reason": fact.get("location_match_reason"),
            "precision_limitations": fact.get("precision_limitations") or [],
            "source_name": fact.get("source_name"),
            "source_document": fact.get("source_document"),
            "source_page": fact.get("source_page"),
            "source_url": fact.get("source_url"),
        }
        for fact in fact_matches["area"]
    ]
    fact_limitations = _guidance_from_facts(selected_facts, "precision_limitations")
    if fact_matches["ambiguous_area_matches"]:
        fact_limitations.append(
            "Multiple local-plan areas matched with equal confidence, so StayReady did not apply any area-specific claim automatically."
        )

    exposure_labels = {
        "mapped_match": "Mapped local exposure found",
        "proximity_context": "Nearby mapped context found",
        "no_mapped_match": "This layer did not find an address-level match",
        "not_checked": "Official layer unavailable or not checked",
        "regional_context": "Regional hazard context",
    }
    priority_reasons = hazard.get("priority_reasons") or []
    exposure_statement = exposure_labels.get(
        hazard.get("hazard_exposure"),
        hazard.get("evidence_status_label", ""),
    ).rstrip(".") + "."
    concise_parts = [
        exposure_statement,
        priority_reasons[0] if priority_reasons else "",
    ]
    concise_summary = " ".join(_dedupe(concise_parts)).strip()

    return {
        "hazard": hazard.get("name") or hazard.get("label") or slug.title(),
        "slug": slug,
        "priority": order,
        "exposure_level": hazard.get("exposure_level") or "Unknown",
        "evidence_status_label": hazard.get("evidence_status_label") or hazard.get("data_status_label") or "Not determined from checked map data",
        "priority_label": hazard.get("priority_label") or "Regional preparedness priority",
        "hazard_exposure": hazard.get("hazard_exposure") or "regional_context",
        "hazard_importance": hazard.get("hazard_importance") or "general",
        "action_priority": hazard.get("action_priority") or "keep_in_plan",
        "source_confidence": hazard.get("source_confidence") or "needs_review",
        "priority_reasons": priority_reasons,
        "concise_summary": concise_summary or why,
        "why_it_matters": why,
        "evidence_badges": _dedupe(evidence_badges),
        "local_guidance_status": local_status,
        "evidence_tier": evidence_tier,
        "evidence_tier_label": EVIDENCE_TIER_LABELS[evidence_tier],
        "location_cues": fact_cues,
        "location_matches": location_matches,
        "official_mapped_evidence": hazard.get("additional_geospatial_evidence") or [],
        "checked": checked_payload["checked"],
        "not_checked": checked_payload["not_checked"],
        "before_actions": before,
        "during_actions": during,
        "after_actions": after,
        "recovery_steps": recovery,
        "household_actions": household_guidance(
            household_context,
            city=location_context.get("city") or "",
            county=location_context.get("county") or "",
            hazards=[slug],
        )[:4],
        "sources": _source_rows(hazard, selected_facts),
        "limitations": _dedupe(hazard.get("limitations", []) + fact_limitations)[:6],
    }


def _recovery_plan(
    hazard_plans: List[Dict],
    household_context: Dict,
    location_context: Dict,
) -> List[Dict]:
    hazards = [plan.get("slug") for plan in hazard_plans if plan.get("slug")]
    actions = select_actions(
        hazards=hazards or ["all"],
        household_factors=household_context.get("tags") or [],
        time_buckets=["recovery"],
        city=location_context.get("city") or "",
        county=location_context.get("county") or "",
        trigger_types=["general", "hazard_result", "location", "household"],
        limit=10,
    )
    category_labels = {
        "life_safety": "Safety after the event",
        "medical": "Health and medication",
        "evacuation": "Temporary housing and evacuation",
        "communication": "Household continuity",
        "property": "Property",
        "recovery": "Documents, insurance, and finances",
        "supplies": "Recovery supplies",
        "official_alerts": "Official information",
    }
    grouped = {}
    for action in actions:
        category = category_labels.get(action.priority_category, "Recovery preparation")
        grouped.setdefault(category, []).append(action.model_dump(mode="json"))
    return [{"category": category, "items": items} for category, items in grouped.items()]


def _summary_text(location_context: Dict, hazard_plans: List[Dict]) -> str:
    city = location_context.get("city") or "Alameda County"
    precision = location_context.get("precision_label") or "location"
    hazard_names = ", ".join(plan["hazard"] for plan in hazard_plans[:3])
    return (
        f"For {location_context.get('display_name')}, StayReady used {precision.lower()} context to organize "
        f"{hazard_names or 'local hazards'} into practical preparedness and recovery steps. "
        "The page separates checked address evidence from local-plan context and fallback guidance."
    ).replace("  ", " ")


def _check_summary(hazard_plans: List[Dict]) -> Dict:
    checked = []
    not_checked = []
    for plan in hazard_plans:
        checked.extend([f"{plan['hazard']}: {item}" for item in plan.get("checked", [])])
        not_checked.extend([f"{plan['hazard']}: {item}" for item in plan.get("not_checked", [])])
    return {"checked": _dedupe(checked), "not_checked": _dedupe(not_checked)}


def build_resident_plan(location_context: Dict, structured_hazards: List[Dict], additional_local_hazards: Optional[List[Dict]] = None, session_data: Optional[Dict] = None) -> Dict:
    session_data = session_data or {}
    household_context = get_household_context(session_data)
    hazard_slugs = [
        hazard.get("slug") or hazard.get("hazard_id")
        for hazard in structured_hazards[:4]
        if hazard.get("slug") or hazard.get("hazard_id")
    ]
    household_actions = household_guidance(
        household_context,
        city=location_context.get("city") or "",
        county=location_context.get("county") or "",
        hazards=hazard_slugs,
    )
    hazard_plans = [
        _hazard_plan(hazard, location_context, household_context, index + 1)
        for index, hazard in enumerate(structured_hazards[:4])
    ]
    hazard_priorities = rank_hazards_for_risk_summary(
        location_context.get("city") or "",
        structured_hazards,
        limit=4,
        coordinates=location_context.get("location_result") or {},
    )
    check_summary = _check_summary(hazard_plans)

    what_to_do_now = []
    for plan in hazard_plans:
        if plan.get("before_actions"):
            what_to_do_now.append({
                "hazard": plan["hazard"],
                "action": plan["before_actions"][0],
            })
    for action in household_actions[:2]:
        what_to_do_now.append({"hazard": "Household priority", "action": action})

    deduped_now = []
    seen_action_ids = set()
    for item in what_to_do_now:
        action_id = item["action"].get("action_id")
        if not action_id or action_id in seen_action_ids:
            continue
        seen_action_ids.add(action_id)
        deduped_now.append(item)

    sources = []
    for plan in hazard_plans:
        sources.extend(plan.get("sources", []))
    source_seen = set()
    source_rows = []
    for row in sources:
        key = ((row.get("url") or "").lower(), (row.get("name") or "").lower())
        if key in source_seen:
            continue
        source_seen.add(key)
        source_rows.append(row)

    return {
        "address_summary": {
            "display_name": location_context.get("display_name") or "Selected location",
            "city": location_context.get("city") or "City not verified",
            "county": location_context.get("county") or "Alameda County",
            "zip_code": location_context.get("zip_code") or "",
            "precision_label": location_context.get("precision_label") or "Unknown",
            "gis_status": location_context.get("gis_status") or "Not checked yet",
            "summary": _summary_text(location_context, hazard_plans),
        },
        "hazards": hazard_plans,
        "hazard_priorities": hazard_priorities,
        "canonical_summary": build_canonical_risk_summary(
            location_context,
            structured_hazards,
            hazard_priorities,
        ),
        "household_context": household_context,
        "household_priorities": household_actions,
        "what_to_do_now": deduped_now[:7],
        "recovery_plan": _recovery_plan(hazard_plans, household_context, location_context),
        "checks": check_summary,
        "sources": source_rows[:8],
        "additional_local_hazards": additional_local_hazards or [],
        "limits": [
            "StayReady is educational guidance and does not replace official emergency alerts, evacuation orders, inspections, insurance advice, or 911.",
            "Address checks are point checks against layers currently loaded by the app; they are not parcel determinations.",
            "A layer not matching the address does not mean the hazard cannot affect the household.",
            "Local-plan context is used only when reviewed or draft-reviewed source data exists; otherwise county fallback language is shown.",
        ],
    }
