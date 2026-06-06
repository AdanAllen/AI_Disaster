import json
import logging
import os
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

from pydantic import ValidationError

from pydantic_models import ActionRecord, PreparednessAction


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGGER = logging.getLogger(__name__)
DISPLAYABLE_REVIEW_STATUSES = {"reviewed", "draft_reviewed"}
PRIORITY_ORDER = {
    "life_safety": 0,
    "official_alerts": 1,
    "medical": 2,
    "evacuation": 3,
    "communication": 4,
    "supplies": 5,
    "property": 6,
    "recovery": 7,
}


def _normalize(value: str) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_set(values: Optional[Iterable[str]]) -> set:
    return {_normalize(value) for value in values or [] if _normalize(value)}


@lru_cache(maxsize=1)
def load_action_library() -> List[ActionRecord]:
    path = os.path.join(BASE_DIR, "preparedness_actions.json")
    try:
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError):
        LOGGER.exception("Preparedness action library could not be loaded.")
        return []

    records = []
    seen = set()
    for index, item in enumerate(payload):
        try:
            record = ActionRecord(**item)
        except ValidationError:
            LOGGER.exception("Preparedness action at index %s failed validation.", index)
            continue
        if record.action_id in seen:
            LOGGER.error("Duplicate preparedness action_id ignored: %s", record.action_id)
            continue
        seen.add(record.action_id)
        records.append(record)
    return records


def publishable_actions() -> List[ActionRecord]:
    return [record for record in load_action_library() if record.displayable]


def get_action_record(action_id: str) -> Optional[ActionRecord]:
    normalized_id = _normalize(action_id)
    for record in publishable_actions():
        if _normalize(record.action_id) == normalized_id:
            return record
    return None


def _jurisdiction_matches(record: ActionRecord, jurisdictions: set) -> bool:
    if not record.applicable_jurisdictions:
        return True
    return bool(_normalized_set(record.applicable_jurisdictions) & jurisdictions)


def _hazard_matches(record: ActionRecord, hazards: set) -> bool:
    record_hazards = _normalized_set(record.hazards)
    return "all" in record_hazards or bool(record_hazards & hazards)


def _selection_reason(
    record: ActionRecord,
    matched_hazards: Sequence[str],
    matched_household_factors: Sequence[str],
    jurisdictions: set,
) -> str:
    if matched_household_factors:
        labels = ", ".join(value.replace("_", " ") for value in matched_household_factors)
        return f"Shown because you reported: {labels}."
    if record.trigger_type == "location" and jurisdictions:
        return "Shown because the resolved location is in Alameda County."
    if matched_hazards:
        labels = ", ".join(value.replace("_", " ").title() for value in matched_hazards)
        return f"Shown because this plan includes {labels}."
    return "Shown as reviewed general household preparedness guidance."


def select_actions(
    *,
    hazards: Optional[Iterable[str]] = None,
    household_factors: Optional[Iterable[str]] = None,
    time_buckets: Optional[Iterable[str]] = None,
    city: str = "",
    county: str = "",
    trigger_types: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[PreparednessAction]:
    hazard_set = _normalized_set(hazards) or {"all"}
    factor_set = _normalized_set(household_factors)
    bucket_set = _normalized_set(time_buckets)
    jurisdiction_set = _normalized_set([city, county])
    trigger_set = _normalized_set(trigger_types)
    selected = []

    for record in publishable_actions():
        if trigger_set and _normalize(record.trigger_type) not in trigger_set:
            continue
        if bucket_set and not (_normalized_set(record.time_buckets) & bucket_set):
            continue
        if not _hazard_matches(record, hazard_set):
            continue
        if not _jurisdiction_matches(record, jurisdiction_set):
            continue

        required_factors = _normalized_set(record.required_household_factors)
        excluded_factors = _normalized_set(record.excluded_household_factors)
        if required_factors and not required_factors.issubset(factor_set):
            continue
        if excluded_factors & factor_set:
            continue

        matched_hazards = sorted(
            value for value in _normalized_set(record.hazards) & hazard_set if value != "all"
        )
        matched_factors = sorted(required_factors & factor_set)
        selected.append(
            PreparednessAction(
                **record.model_dump(),
                why_shown=_selection_reason(
                    record,
                    matched_hazards,
                    matched_factors,
                    jurisdiction_set,
                ),
                matched_hazards=matched_hazards,
                matched_household_factors=matched_factors,
            )
        )

    selected.sort(
        key=lambda action: (
            0 if action.guidance_scope == "hazard_specific" else 1,
            0 if action.trigger_type == "location" else 1,
            PRIORITY_ORDER.get(action.priority_category, 99),
            action.action_id,
        )
    )

    deduped = []
    seen = set()
    for action in selected:
        if action.action_id in seen:
            continue
        seen.add(action.action_id)
        deduped.append(action)
    return deduped[:limit] if limit is not None else deduped


def select_action_ids(
    action_ids: Iterable[str],
    *,
    reason: str = "Shown in this reviewed preparedness checklist.",
) -> List[PreparednessAction]:
    selected = []
    seen = set()
    for action_id in action_ids:
        record = get_action_record(action_id)
        if not record or record.action_id in seen:
            continue
        seen.add(record.action_id)
        selected.append(
            PreparednessAction(
                **record.model_dump(),
                why_shown=reason,
                matched_hazards=[],
                matched_household_factors=[],
            )
        )
    return selected
