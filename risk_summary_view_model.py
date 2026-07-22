"""Canonical, evidence-scoped view model for the live Risk Summary.

This module intentionally does not calculate address-level risk. It translates
existing structured evidence into display-safe regional, address, vulnerability,
attention, action, and claim-citation scopes.
"""

from typing import Dict, Iterable, List, Optional

from hazard_priority import reviewed_subarea_context_for_hazard


ADDRESS_OUTCOMES = {
    "mapped_match",
    "proximity_context",
    "checked_no_match",
    "data_unavailable",
    "not_checked",
    "not_applicable",
}

DESCRIPTORS = {
    "earthquake": "Shaking, faults, and ground failure",
    "flood": "Floodplains, rivers, and drainage",
    "wildfire": "Fire hazard, smoke, and evacuation",
    "tsunami": "Coastal wave and evacuation hazard",
    "tsunami-seiche": "Coastal wave and evacuation hazard",
    "landslide": "Slope and ground-movement hazard",
    "dam_failure": "Dam-inundation planning scenarios",
    "dam-failure": "Dam-inundation planning scenarios",
}

REVIEWED_STATUSES = {"reviewed", "draft_reviewed", "adopted"}


def _dedupe(values: Iterable[str]) -> List[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        output.append(text)
    return output


def _slug(hazard: Dict) -> str:
    return str(hazard.get("slug") or hazard.get("hazard_id") or hazard.get("hazard_type") or "").strip().lower()


def _claim(*, claim: str, source: Dict, scope: str, review_status: str, evidence: Optional[Dict] = None) -> Dict:
    evidence = evidence or {}
    return {
        "claim": claim,
        "agency": evidence.get("source_agency") or source.get("agency") or source.get("name") or "",
        "dataset_or_document": evidence.get("dataset_name") or source.get("name") or source.get("source_document") or "",
        "reference": evidence.get("source_detail") or source.get("reference") or source.get("source_section") or "",
        "geographic_scope": scope,
        "review_status": review_status or source.get("review_status") or source.get("document_status") or "unknown",
        "source_id": evidence.get("source_id") or evidence.get("dataset_id") or source.get("source_id") or "",
        "url": evidence.get("source_url") or source.get("url") or source.get("source_url") or "",
    }


def _regional_context(hazard: Dict, priority: Dict, location_context: Dict) -> Dict:
    jurisdiction = location_context.get("city") or location_context.get("county") or "Alameda County"
    sources = priority.get("sources_used") or []
    source = sources[0] if sources else {}
    document_status = priority.get("document_status") or source.get("document_status") or "unavailable"
    official_priority = priority.get("official_lhmp_priority")
    probability = priority.get("probability")
    impact = priority.get("impact")
    has_exact_source = bool(source.get("url") and (priority.get("source_page") or source.get("reference")))
    has_record = bool(priority.get("source_document") and document_status)

    # Draft categories may be displayed only as exact, visibly draft citywide
    # source statements. They are never promoted into address ratings.
    show_categories = has_record and has_exact_source and document_status in {
        "draft", "reviewed", "draft_reviewed", "adopted"
    }
    probability_label = probability if show_categories and probability in {"Low", "Medium", "High"} else ""
    impact_label = impact if show_categories and impact in {"Low", "Medium", "High"} else ""
    priority_label = official_priority if show_categories and official_priority in {"Low", "Medium", "High"} else ""

    if priority_label:
        status = "available"
        scope_label = f"{document_status.title()} {jurisdiction} citywide priority"
        explanation = f"{scope_label}: {priority_label}. This is citywide planning context, not an address-risk rating."
        claim_text = explanation
    elif probability_label or impact_label:
        status = "available"
        scope_label = f"{document_status.title()} {jurisdiction} citywide context"
        parts = []
        if probability_label:
            parts.append(f"probability {probability_label}")
        if impact_label:
            parts.append(f"impact {impact_label}")
        explanation = f"{scope_label}: {', '.join(parts)}. These categories do not rate the address."
        claim_text = explanation
    else:
        status = "unavailable"
        scope_label = f"{jurisdiction} regional context"
        explanation = "No eligible, exactly cited jurisdiction rating is available for this hazard."
        claim_text = ""

    claims = []
    if claim_text:
        claims.append(_claim(
            claim=claim_text,
            source=source,
            scope=f"{jurisdiction} citywide",
            review_status=document_status,
        ))
    return {
        "status": status,
        "jurisdiction": jurisdiction,
        "scope_label": scope_label,
        "probability_label": probability_label,
        "impact_label": impact_label,
        "priority_label": priority_label,
        "source_status": document_status,
        "explanation": explanation,
        "claims": claims,
        "limitations": _dedupe((priority.get("limitations") or []) + [
            "Jurisdiction categories describe planning context and do not determine risk for an individual address."
        ]),
    }


def _subarea_context(hazard: Dict, priority: Dict, location_context: Dict, matched_area: Dict) -> Dict:
    jurisdiction = location_context.get("city") or ""
    subarea_name = matched_area.get("sub_area") or ""
    if jurisdiction.strip().lower() != "oakland" or not subarea_name or subarea_name == "Unknown":
        return {
            "status": "not_applicable" if jurisdiction.strip().lower() != "oakland" else "unavailable",
            "subarea_name": "",
            "source_status": "",
            "summary": "Oakland sub-area context does not apply to this location." if jurisdiction.strip().lower() != "oakland" else "No reliable Oakland sub-area match is available.",
            "findings": [],
            "claims": [],
            "limitations": [],
        }

    context = reviewed_subarea_context_for_hazard(jurisdiction, subarea_name, _slug(hazard))
    records = context.get("records") or []
    findings = []
    claims = []
    for record in records:
        text = record.get("display_text") or record.get("source_claim") or ""
        if not text:
            continue
        findings.append({
            "text": text,
            "metric_type": record.get("metric_type") or "",
            "scenario": record.get("scenario") or "",
            "unit": record.get("unit") or "",
            "draft": record.get("document_status") == "draft",
            "reference": f"{record.get('table')}; PDF page {record.get('page')}",
        })
        claims.append({
            "claim": text,
            "agency": "City of Oakland",
            "dataset_or_document": record.get("source_document") or "",
            "reference": f"{record.get('table')}; PDF page {record.get('page')}",
            "geographic_scope": f"Oakland sub-area: {subarea_name}",
            "review_status": record.get("review_status") or "unknown",
            "source_id": "oakland_draft_lhmp_subarea",
            "url": "https://www.oaklandca.gov/topics/local-hazard-mitigation-plan",
        })
    status = "available" if findings else "unavailable"
    return {
        "status": status,
        "subarea_name": subarea_name,
        "source_status": "draft" if findings else "",
        "boundary_warning": matched_area.get("boundary_warning") or "",
        "summary": context.get("summary") or "No reviewed sub-area metric is available for this hazard.",
        "findings": findings,
        "claims": claims,
        "limitations": context.get("limitations") or [],
    }


def _positive_records(hazard: Dict) -> List[Dict]:
    records = []
    seen = set()
    for item in (hazard.get("normalized_mapped_evidence") or []) + (hazard.get("additional_geospatial_evidence") or []):
        if not isinstance(item, dict):
            continue
        if item.get("matched") is True or item.get("is_in_hazard_zone") is True or item.get("hazard_exposure") == "mapped_match":
            key = (
                item.get("dataset_id") or item.get("dataset_name") or item.get("source_url"),
                item.get("result_label") or item.get("status_label") or item.get("name"),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(item)
    return records


def _evidence_dataset(record: Dict) -> str:
    return str(
        record.get("dataset_name")
        or record.get("label")
        or record.get("dataset_id")
        or "Official mapped layer"
    ).strip()


def _evidence_status(record: Dict) -> str:
    raw_status = str(
        record.get("exposure")
        or record.get("status")
        or record.get("data_status")
        or ""
    ).strip().lower()
    if record.get("data_available") is False or raw_status in {"data_unavailable", "unavailable"}:
        return "data_unavailable"
    if record.get("claim_type") == "proximity":
        return "proximity_context"
    if record.get("matched") is True:
        return "mapped_match"
    if record.get("matched") is False and (
        record.get("checked") is True
        or raw_status in {"checked", "not_in_layer", "no_mapped_match", ""}
    ):
        return "checked_no_match"
    if raw_status in {"not_checked", "unknown", ""}:
        return "not_checked"
    if raw_status in {"not_in_layer", "no_mapped_match"}:
        return "checked_no_match"
    return "not_checked"


def _layer_names(record: Dict) -> List[str]:
    details = record.get("details") or {}
    features = (
        record.get("matched_features")
        or details.get("matched_features")
        or details.get("matched_layers")
        or []
    )
    names = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        name = (
            feature.get("name")
            or feature.get("display_name")
            or feature.get("result_label")
        )
        if name:
            names.append(str(name))
    return _dedupe(names)


def _evidence_finding(record: Dict, status: str) -> str:
    explicit = str(record.get("result_label") or "").strip()
    if explicit and status in {"mapped_match", "proximity_context", "data_unavailable"}:
        return explicit
    dataset = _evidence_dataset(record)
    names = _layer_names(record)
    if status == "mapped_match" and names:
        return "; ".join(names)
    if status == "proximity_context" and names:
        nearest = names[0]
        details = (record.get("matched_features") or (record.get("details") or {}).get("matched_layers") or [{}])[0]
        distance_km = details.get("distance_km") if isinstance(details, dict) else None
        if distance_km is not None:
            return f"Approximately {float(distance_km) * 0.621371:.1f} miles from {nearest}."
        return f"Proximity to {nearest} was calculated separately from zone membership."
    if status == "checked_no_match" and names:
        return f"Returned official category: {'; '.join(names)}. No qualifying mapped concern was identified by this check."
    if status == "checked_no_match" and explicit and any(
        phrase in explicit.lower() for phrase in ("no mapped match", "outside", "not in")
    ):
        return explicit
    if status == "checked_no_match":
        return f"No mapped match found in {dataset}."
    if status == "data_unavailable":
        return f"{dataset} was unavailable and was not checked."
    if status == "proximity_context":
        return f"A proximity check was returned by {dataset}; it is not polygon membership."
    return f"{dataset} was not checked."


def _evidence_items(hazard: Dict, *, address_mode: bool) -> List[Dict]:
    if not address_mode:
        return []
    # Rich source records come first; normalized compatibility records fill
    # gaps. One dataset/claim pair is shown once so CGS application does not
    # duplicate the same source result.
    candidates = list(hazard.get("additional_geospatial_evidence") or [])
    candidates.extend(hazard.get("normalized_mapped_evidence") or [])
    items = []
    seen = set()
    for record in candidates:
        if not isinstance(record, dict):
            continue
        dataset_id = str(record.get("dataset_id") or _evidence_dataset(record)).strip()
        claim_type = str(record.get("claim_type") or "mapped_layer").strip()
        key = (dataset_id.lower(), claim_type.lower())
        if key in seen:
            continue
        seen.add(key)
        status = _evidence_status(record)
        details = record.get("details") if isinstance(record.get("details"), dict) else {}
        review_status = (
            record.get("public_claim_status")
            or record.get("review_status")
            or details.get("public_claim_status")
            or details.get("review_status")
            or "unknown"
        )
        checked_at = record.get("checked_at") or details.get("checked_at") or ""
        source = {
            "name": _evidence_dataset(record),
            "url": record.get("source_url") or "",
        }
        finding = _evidence_finding(record, status)
        scope = "Near address" if status == "proximity_context" else "Address point"
        items.append({
            "dataset_id": dataset_id,
            "dataset_name": _evidence_dataset(record),
            "status": status,
            "status_label": {
                "mapped_match": "Official mapped match found",
                "proximity_context": "Near a mapped feature" if record.get("near") else "Proximity calculated",
                "checked_no_match": "Checked, no mapped match found",
                "data_unavailable": "Data unavailable",
                "not_checked": "Not checked",
            }[status],
            "finding": finding,
            "geographic_scope": scope,
            "claim_type": claim_type,
            "source_agency": record.get("source_agency") or "",
            "source_url": record.get("source_url") or "",
            "checked_at": checked_at,
            "review_status": review_status,
            "limitations": record.get("limitations") or [],
            "near": record.get("near"),
            "claim": _claim(
                claim=finding,
                source=source,
                scope=scope.lower(),
                review_status=review_status,
                evidence=record,
            ),
        })
    return items


def _address_outcome(hazard: Dict, *, address_mode: bool) -> str:
    if not address_mode:
        return "not_checked"
    scope = hazard.get("scope") or (hazard.get("structured_result") or {}).get("scope")
    if scope != "address_level":
        return "not_checked"
    exposure = hazard.get("hazard_exposure") or (hazard.get("structured_result") or {}).get("hazard_exposure")
    status = hazard.get("data_status") or (hazard.get("structured_result") or {}).get("data_status")
    if exposure == "mapped_match" and _positive_records(hazard):
        return "mapped_match"
    if exposure == "proximity_context":
        return "proximity_context"
    if status == "data_unavailable" or exposure == "not_checked":
        return "data_unavailable"
    if exposure == "no_mapped_match" or status == "not_in_layer":
        return "checked_no_match"
    if status in {"not_applicable", "outside_coverage"}:
        return "not_applicable"
    return "not_checked"


def _address_evidence(hazard: Dict, location_context: Dict) -> Dict:
    address_mode = location_context.get("location_mode") == "address" and location_context.get("has_precise_location") is not False
    records = _evidence_items(hazard, address_mode=address_mode)
    mapped = [item for item in records if item["status"] == "mapped_match" and item["claim_type"] != "proximity"]
    proximity = [item for item in records if item["status"] == "proximity_context"]
    nearby = [item for item in proximity if item.get("near") is True]
    nonmatches = [item for item in records if item["status"] == "checked_no_match"]
    unavailable = [item for item in records if item["status"] == "data_unavailable"]

    parent_outcome = _address_outcome(hazard, address_mode=address_mode)
    if mapped:
        outcome = "mapped_match"
    elif nearby or parent_outcome == "proximity_context":
        outcome = "proximity_context"
    elif nonmatches:
        outcome = "checked_no_match"
    elif unavailable:
        outcome = "data_unavailable"
    else:
        outcome = parent_outcome if parent_outcome in ADDRESS_OUTCOMES else "not_checked"

    primary_records = mapped if outcome == "mapped_match" else (nearby or proximity) if outcome == "proximity_context" else []
    finding_labels = _dedupe(item["finding"] for item in primary_records)
    claims = [item["claim"] for item in records]

    labels = {
        "mapped_match": "Mapped match found",
        "proximity_context": "Nearby mapped feature",
        "checked_no_match": "No mapped match found",
        "data_unavailable": "Data unavailable",
        "not_checked": "Not checked",
        "not_applicable": "Not applicable",
    }
    explanations = {
        "mapped_match": "A positive structured address-level intersection was returned by the checked official layer.",
        "proximity_context": "A mapped feature is nearby. Proximity is not polygon or hazard-zone membership.",
        "checked_no_match": "The checked layer did not return a mapped match. This does not mean the hazard cannot affect the location.",
        "data_unavailable": "The address-level source could not be checked. Missing data was not converted into a lower rating.",
        "not_checked": "No eligible address-level map result is available for this hazard.",
        "not_applicable": "The checked official source does not apply to this location.",
    }
    hazard_id = _slug(hazard)
    if outcome == "mapped_match" and hazard_id == "wildfire":
        explanations["mapped_match"] = "The checked official layer returned a mapped fire-hazard-severity category. It does not estimate personal annual wildfire probability or loss."
    elif outcome == "mapped_match" and hazard_id == "flood":
        explanations["mapped_match"] = "The checked official layer returned a mapped FEMA flood category. It is not a universal High, Medium, or Low address-risk score."
    return {
        "outcome": outcome,
        "label": labels[outcome],
        "findings": finding_labels,
        "records": records,
        "mapped_matches": mapped,
        "proximity": proximity,
        "nonmatches": nonmatches,
        "unavailable": unavailable,
        "has_source_failures": bool(unavailable),
        "explanation": explanations[outcome],
        "claims": claims,
        "limitations": _dedupe(hazard.get("limitations") or []),
    }


def _actions(hazard: Dict) -> Dict:
    groups = {"before": [], "during": [], "recovery": []}
    for action in hazard.get("recommended_actions") or hazard.get("action_steps") or []:
        buckets = action.get("time_buckets") or []
        if "during" in buckets:
            groups["during"].append(action)
        elif "after" in buckets or "recovery" in buckets:
            groups["recovery"].append(action)
        else:
            groups["before"].append(action)
    return {key: value[:3] for key, value in groups.items()}


def _attention(hazard: Dict, regional: Dict, subarea: Dict, address: Dict) -> Dict:
    if address["outcome"] == "mapped_match":
        category, label = "direct_mapped_concern", "Direct mapped concern"
    elif address["outcome"] == "proximity_context":
        category, label = "near_mapped_feature", "Near a mapped feature"
    elif address["outcome"] == "checked_no_match" and (
        subarea["status"] == "available"
        or regional["status"] == "available"
        or hazard.get("hazard_importance") == "major_regional"
    ):
        category, label = "prepare_despite_no_match", "Prepare despite no mapped match"
    elif address["outcome"] == "data_unavailable":
        category, label = "information_unavailable", "Information unavailable"
    elif subarea["status"] == "available" or regional["status"] == "available":
        category, label = "important_local_hazard", "Important local hazard"
    else:
        category, label = "general_awareness", "General awareness"
    based_on = _dedupe(
        [claim.get("claim") for claim in address["claims"]]
        + [claim.get("claim") for claim in subarea["claims"]]
        + [claim.get("claim") for claim in regional["claims"]]
    )
    return {
        "category": category,
        "label": label,
        "explanation": hazard.get("priority_reasons", [""])[0] if hazard.get("priority_reasons") else hazard.get("why_shown") or address["explanation"],
        "based_on": based_on,
    }


def _interpretation(hazard: Dict, address: Dict, regional: Dict, subarea: Dict) -> str:
    local_context = ""
    if subarea.get("status") == "available" and subarea.get("findings"):
        local_context = subarea["findings"][0]["text"]
    elif regional.get("status") == "available":
        local_context = regional.get("explanation") or ""

    if address["outcome"] == "mapped_match":
        lead = address["findings"][0] if address["findings"] else "A checked official map returned an address-point match."
    elif address["outcome"] == "proximity_context":
        lead = address["findings"][0] if address["findings"] else "A mapped feature is near the selected point; this is not zone membership."
    elif address["outcome"] == "checked_no_match":
        lead = "The checked address layers did not return a qualifying mapped match. This does not rule out indirect, unmapped, or future impacts."
    elif address["outcome"] == "data_unavailable":
        lead = "One or more address-level source checks were unavailable. Missing data was not treated as a safer result."
    else:
        lead = "No eligible address-level map result is available for this hazard."

    parts = [lead]
    if local_context:
        parts.append(local_context)
    if address.get("has_source_failures") and address["outcome"] != "data_unavailable":
        parts.append("Another source check was unavailable, so valid returned findings remain visible while that gap stays unknown.")
    parts.append("Building condition and household vulnerability remain unknown.")
    return " ".join(_dedupe(parts))


def build_canonical_risk_summary(location_context: Dict, hazards: List[Dict], hazard_priorities: List[Dict]) -> Dict:
    priorities = {item.get("slug"): item for item in hazard_priorities or []}
    matched_area = next(
        (
            item.get("sub_area_context") for item in hazard_priorities or []
            if isinstance(item.get("sub_area_context"), dict)
            and item["sub_area_context"].get("sub_area") not in (None, "", "Unknown")
        ),
        {},
    )
    normalized = []
    for raw_hazard in hazards or []:
        hazard = {**(raw_hazard.get("structured_result") or {}), **raw_hazard}
        hazard_id = _slug(hazard)
        if not hazard_id:
            continue
        priority = priorities.get(hazard_id) or priorities.get(hazard_id.replace("-seiche", "")) or {}
        regional = _regional_context(hazard, priority, location_context)
        subarea = _subarea_context(hazard, priority, location_context, matched_area)
        address = _address_evidence(hazard, location_context)
        vulnerability = {
            "status": "unknown",
            "label": "Unknown",
            "explanation": "StayReady has not assessed building construction, retrofit status, insurance, or household-specific vulnerability.",
        }
        actions = _actions(hazard)
        attention = _attention(hazard, regional, subarea, address)
        normalized.append({
            "hazard_id": hazard_id,
            "hazard_name": hazard.get("name") or hazard.get("label") or hazard_id.replace("-", " ").title(),
            "descriptor": DESCRIPTORS.get(hazard_id, "Preparedness and response context"),
            "detail_url": f"/hazards/{hazard_id}",
            "regional_context": regional,
            "subarea_context": subarea,
            "address_evidence": address,
            "vulnerability": vulnerability,
            "attention": attention,
            "interpretation": _interpretation(hazard, address, regional, subarea),
            "actions": actions,
        })
    unique_checks = {}
    for hazard in normalized:
        for item in hazard["address_evidence"].get("records", []):
            unique_checks[(item.get("dataset_id"), item.get("claim_type"))] = item
    check_records = list(unique_checks.values())
    unavailable_count = sum(item.get("status") == "data_unavailable" for item in check_records)
    completed_count = sum(item.get("status") in {"mapped_match", "checked_no_match", "proximity_context"} for item in check_records)
    address_mode = location_context.get("location_mode") == "address" and location_context.get("has_precise_location") is not False
    return {
        "location_mode": location_context.get("location_mode") or "unknown",
        "location_label": location_context.get("display_name") or "Selected location",
        "hazards": normalized,
        "result_quality": {
            "is_address_level": bool(address_mode),
            "precision_label": location_context.get("precision_label") or "Unknown",
            "geocoding_note": (
                "The location was geocoded to an address point. Rooftop, parcel, and boundary precision are not guaranteed."
                if address_mode else
                "ZIP fallback provides broad area context only; no address-point intersections were inferred."
            ),
            "completed_check_count": completed_count,
            "unavailable_check_count": unavailable_count,
            "status_label": (
                "Some source checks unavailable"
                if unavailable_count else
                "Address-level checks returned"
                if address_mode and completed_count else
                "ZIP fallback only"
                if not address_mode else
                "No address checks returned"
            ),
        },
        "unknowns": [
            "Building construction and condition",
            "Foundation, roof, vegetation, and floor elevation",
            "Retrofit status",
            "Insurance, mobility, medical, and household vulnerability",
            "Exact future hazard intensity or damage probability",
            "Live emergency conditions",
        ],
    }
