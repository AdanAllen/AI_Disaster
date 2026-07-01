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
    outcome = _address_outcome(hazard, address_mode=address_mode)
    positive = _positive_records(hazard) if outcome == "mapped_match" else []
    normalized = hazard.get("normalized_mapped_evidence") or []
    sources = hazard.get("sources") or []
    source = sources[0] if sources else {}

    finding_labels = []
    claims = []
    for evidence in positive:
        label = evidence.get("result_label") or evidence.get("status_label") or evidence.get("dataset_name")
        if label:
            finding_labels.append(label)
            claims.append(_claim(
                claim=label,
                source=source,
                scope="address point",
                review_status=evidence.get("public_claim_status") or hazard.get("review_status") or "unknown",
                evidence=evidence,
            ))

    if outcome == "mapped_match" and not positive:
        # Fail closed on contradictory parent flags.
        outcome = "not_checked"
    if outcome == "proximity_context":
        proximity_records = [
            item for item in normalized
            if isinstance(item, dict) and (item.get("claim_type") == "proximity" or item.get("status") == "proximity_context")
        ]
        finding_labels = _dedupe([
            item.get("result_label") or item.get("status_label") or item.get("name")
            for item in proximity_records
        ])
        if not finding_labels:
            finding_labels = ["Nearby mapped fault context"]
        for index, label in enumerate(finding_labels):
            claims.append(_claim(
                claim=label,
                source=source,
                scope="address-point proximity",
                review_status=hazard.get("review_status") or "unknown",
                evidence=proximity_records[index] if index < len(proximity_records) else {},
            ))

    if outcome == "checked_no_match":
        nonmatches = [
            item for item in (hazard.get("additional_geospatial_evidence") or []) + normalized
            if isinstance(item, dict)
            and item.get("matched") is False
            and item.get("data_available") is not False
            and (item.get("checked") is True or item.get("exposure") == "no_mapped_match")
        ]
        seen_datasets = set()
        for item in nonmatches:
            dataset_key = item.get("dataset_id") or item.get("dataset_name")
            if not dataset_key or dataset_key in seen_datasets:
                continue
            seen_datasets.add(dataset_key)
            claims.append(_claim(
                claim=f"No mapped match found in {item.get('dataset_name') or 'the checked layer'}.",
                source=source,
                scope="address point",
                review_status=item.get("public_claim_status") or hazard.get("review_status") or "unknown",
                evidence=item,
            ))

    if outcome == "data_unavailable":
        unavailable_records = [
            item for item in (hazard.get("additional_geospatial_evidence") or []) + normalized
            if isinstance(item, dict) and (item.get("data_available") is False or item.get("data_status") == "data_unavailable")
        ]
        for item in unavailable_records[:3]:
            claims.append(_claim(
                claim=f"{item.get('dataset_name') or 'Address-level layer'} was unavailable.",
                source=source,
                scope="address point",
                review_status=item.get("public_claim_status") or hazard.get("review_status") or "unknown",
                evidence=item,
            ))

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
        category, label = "address_finding", "Address finding"
    elif address["outcome"] == "proximity_context":
        category, label = "address_finding", "Nearby mapped feature"
    elif subarea["status"] == "available":
        category, label = "subarea_context", "Sub-area context"
    elif regional["status"] == "available":
        category, label = "regional_priority", "Regional priority"
    elif address["outcome"] in {"data_unavailable", "not_checked"}:
        category, label = "insufficient_evidence", "Insufficient evidence"
    else:
        category, label = "preparedness_context", "Preparedness context"
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
            "actions": actions,
        })
    return {
        "location_mode": location_context.get("location_mode") or "unknown",
        "location_label": location_context.get("display_name") or "Selected location",
        "hazards": normalized,
        "unknowns": [
            "Building construction and condition",
            "Retrofit status",
            "Insurance and household vulnerability",
            "Live emergency conditions",
        ],
    }
