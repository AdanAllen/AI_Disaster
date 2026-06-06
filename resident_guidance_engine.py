import json
import os
from functools import lru_cache
from typing import Dict, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REVIEWED_STATUSES = {"reviewed", "draft_reviewed"}


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
    return _load_json("hazard_facts.json", [])


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
        if any(keyword in special_lower for keyword in keywords):
            tags.add(tag)

    if special_needs:
        tags.add("medical")

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


def household_guidance(household_context: Dict) -> List[Dict]:
    tags = set(household_context.get("tags") or [])
    notes = []

    if household_context.get("household_size"):
        notes.append({
            "label": "Household size",
            "text": f"Plan water, food, medications, chargers, transportation, and temporary housing around {household_context['household_size']} household member(s).",
        })
    if household_context.get("preparedness"):
        notes.append({
            "label": "Preparedness level",
            "text": f"Because preparedness was marked as {household_context['preparedness']}, the plan emphasizes practical first steps and recovery basics.",
        })
    if "medical" in tags:
        notes.append({
            "label": "Medical and medication needs",
            "text": "Keep prescription copies, refill timing, doctor/pharmacy contacts, medical-device instructions, and backup power or cooling plans ready.",
        })
    if "pets" in tags:
        notes.append({
            "label": "Pets",
            "text": "Prepare pet food, water, carrier, leash, medications, vaccination records, ID, and pet-friendly evacuation or temporary housing options.",
        })
    if "no_car" in tags:
        notes.append({
            "label": "No reliable car access",
            "text": "Plan transit, rides, neighbors, family contacts, and earlier evacuation timing. Do not wait until the last minute if officials warn of evacuation.",
        })
    if "renter" in tags:
        notes.append({
            "label": "Renter recovery",
            "text": "Keep renter insurance details, landlord contact information, lease copies, belongings photos, and temporary housing options accessible.",
        })
    if "homeowner" in tags:
        notes.append({
            "label": "Homeowner recovery",
            "text": "Review insurance, know utility shutoff basics, photograph property and belongings, and keep repair/documentation contacts ready.",
        })
    if "children" in tags:
        notes.append({
            "label": "Children and school continuity",
            "text": "Keep school contacts, reunification plans, comfort items, medications, and backup caregiver arrangements ready.",
        })
    if "older_adults" in tags or "access_needs" in tags:
        notes.append({
            "label": "Access and functional needs",
            "text": "Plan mobility support, assistive devices, caregiver contacts, backup supplies, accessible transportation, and charging needs.",
        })
    if household_context.get("special_needs"):
        notes.append({
            "label": "Reported needs",
            "text": f"Reported household note: {household_context['special_needs']}",
        })

    return notes


def _match_facts(city: str, hazard: str) -> Dict[str, List[Dict]]:
    city_key = _norm(city)
    hazard_key = _norm(hazard)
    local = []
    county = []
    needs_review = False

    for fact in load_hazard_facts(city=city, hazard=hazard):
        fact_hazard = _norm(fact.get("hazard"))
        if fact_hazard != hazard_key:
            continue
        review_status = fact.get("review_status", "")
        jurisdiction_key = _norm(fact.get("jurisdiction"))
        if jurisdiction_key == city_key and review_status in REVIEWED_STATUSES:
            local.append(fact)
        elif jurisdiction_key == city_key:
            needs_review = True
        elif jurisdiction_key in {"alameda_county", "unincorporated_alameda_county"} and review_status in REVIEWED_STATUSES:
            county.append(fact)

    return {"local": local, "county": county, "needs_review": needs_review}


def _guidance_from_chunks(hazard: Dict, phase: str) -> List[str]:
    guidance = ((hazard.get("specialized_guidance") or {}).get("resident_guidance") or {})
    items = guidance.get(phase) or []
    values = []
    for item in items:
        values.append(item.get("recommended_action") or item.get("recovery_question") or item.get("plain_language"))
    return _dedupe(values)


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
                "basis": fact.get("evidence_type", "").replace("_", " ").title(),
                "review_status": fact.get("review_status", "").replace("_", " ").title(),
            })
    for source in hazard.get("sources", [])[:4]:
        rows.append({
            "name": source.get("name") or source.get("agency") or "Source",
            "url": source.get("url") or "",
            "basis": source.get("claim_type", source.get("source_type", "")).replace("_", " ").title(),
            "review_status": source.get("review_status", "").replace("_", " ").title(),
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
            checked.append("Saved address point checked for approximate proximity to loaded mapped fault lines.")
    else:
        checked.append(f"{scope_label} used for this result.")

    if data_status == "not_in_layer":
        checked.append("The checked layer did not match the address as a hazard-zone membership result; this does not mean no risk.")
    if slug == "flood" and hazard.get("scope") != "address_level":
        not_checked.append("FEMA address-point flood overlay was not completed for this result.")
    if slug == "wildfire" and hazard.get("scope") != "address_level":
        not_checked.append("Address-level fire hazard zone membership is not available for this result.")
    if slug == "earthquake":
        not_checked.extend([
            "Liquefaction polygons were not checked.",
            "Building retrofit, structural condition, and parcel-level seismic risk were not checked.",
        ])
    if slug in {"flood", "wildfire"}:
        not_checked.append("Parcel-level property conditions and building-specific vulnerability were not checked.")

    return {"checked": _dedupe(checked), "not_checked": _dedupe(not_checked)}


def _hazard_plan(hazard: Dict, location_context: Dict, household_notes: List[Dict], order: int) -> Dict:
    city = location_context.get("city") or ""
    slug = hazard.get("slug") or hazard.get("hazard_id")
    fact_matches = _match_facts(city, slug)
    selected_facts = fact_matches["local"] or fact_matches["county"]
    specialized = hazard.get("specialized_guidance") or {}
    guidance_source_status = specialized.get("guidance_source_status")
    has_local_guidance = guidance_source_status == "local_reviewed"
    local_status = "local_reviewed" if fact_matches["local"] or has_local_guidance else "county_fallback"
    if not fact_matches["local"] and city and not has_local_guidance:
        local_status = "local_not_reviewed"

    fact_meaning = _dedupe([fact.get("resident_meaning", "") for fact in selected_facts])
    fact_cues = _dedupe([fact.get("location_cue", "") for fact in selected_facts])
    checked_payload = _what_was_checked(hazard)

    before = _dedupe(
        _guidance_from_chunks(hazard, "before")
        + _guidance_from_facts(selected_facts, "before_actions")
        + [item.get("label") for item in hazard.get("recommended_actions", [])]
    )[:5]
    during = _dedupe(
        _guidance_from_chunks(hazard, "during")
        + _guidance_from_facts(selected_facts, "during_actions")
    )[:4]
    after = _dedupe(
        _guidance_from_chunks(hazard, "after")
        + _guidance_from_facts(selected_facts, "after_actions")
    )[:4]
    recovery = _dedupe(
        _guidance_from_chunks(hazard, "recovery")
        + _guidance_from_facts(selected_facts, "recovery_steps")
        + ((hazard.get("specialized_guidance") or {}).get("recovery_needs") or [])
    )[:6]

    why_parts = _dedupe([
        hazard.get("why_shown", ""),
        *(fact_meaning[:1]),
        *(fact_cues[:1]),
    ])
    why = " ".join(why_parts)
    city_context = specialized.get("city_context") or []
    location_specific_context = specialized.get("location_specific_context") or []
    if has_local_guidance and not fact_matches["local"]:
        why_parts.extend(_dedupe(location_specific_context + city_context)[:2])
    if not fact_matches["local"] and city and not has_local_guidance:
        why += f" Local plan facts for {city} are not fully reviewed in the structured facts layer yet, so county-level guidance is included."

    evidence_badges = [
        hazard.get("scope_label") or "County fallback",
        hazard.get("data_status_label") or "Needs review",
        "Local plan context" if local_status == "local_reviewed" else "County fallback",
    ]

    return {
        "hazard": hazard.get("name") or hazard.get("label") or slug.title(),
        "slug": slug,
        "priority": order,
        "exposure_level": hazard.get("exposure_level") or "Unknown",
        "why_it_matters": why,
        "evidence_badges": _dedupe(evidence_badges),
        "local_guidance_status": local_status,
        "location_cues": fact_cues,
        "checked": checked_payload["checked"],
        "not_checked": checked_payload["not_checked"],
        "before_actions": before,
        "during_actions": during,
        "after_actions": after,
        "recovery_steps": recovery,
        "household_notes": household_notes[:4],
        "sources": _source_rows(hazard, selected_facts),
        "limitations": _dedupe(hazard.get("limitations", []))[:5],
    }


def _recovery_plan(hazard_plans: List[Dict], household_notes: List[Dict]) -> List[Dict]:
    base = [
        {
            "category": "Documents",
            "items": [
                "Save copies of IDs, insurance, lease or mortgage documents, medical records, and emergency contacts.",
                "Photograph important belongings before a disaster so claims and replacement are easier.",
            ],
        },
        {
            "category": "Housing",
            "items": [
                "Identify at least two places your household could stay if home is unsafe.",
                "Include pet-friendly or accessible options if your household needs them.",
            ],
        },
        {
            "category": "Health and medication",
            "items": [
                "Keep medication lists, refill plans, doctor/pharmacy contacts, and backup charging plans accessible.",
            ],
        },
        {
            "category": "Money and continuity",
            "items": [
                "Plan for deductibles, lost work time, transportation disruption, school/work continuity, and urgent repairs.",
            ],
        },
    ]
    if household_notes:
        base.append({
            "category": "Household-specific",
            "items": [note["text"] for note in household_notes[:4]],
        })
    return base


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
    household_notes = household_guidance(household_context)
    hazard_plans = [
        _hazard_plan(hazard, location_context, household_notes, index + 1)
        for index, hazard in enumerate(structured_hazards[:4])
    ]
    check_summary = _check_summary(hazard_plans)

    what_to_do_now = []
    for plan in hazard_plans:
        if plan.get("before_actions"):
            what_to_do_now.append({
                "hazard": plan["hazard"],
                "action": plan["before_actions"][0],
            })
    for note in household_notes[:2]:
        what_to_do_now.append({"hazard": note["label"], "action": note["text"]})

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
        "household_context": household_context,
        "household_priorities": household_notes,
        "what_to_do_now": what_to_do_now[:7],
        "recovery_plan": _recovery_plan(hazard_plans, household_notes),
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
