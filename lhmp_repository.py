"""Read-only access to manually reviewed LHMP records.

This module intentionally has no fallback to candidate, Markdown, or raw data.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from lhmp_models import (
    ReviewedDataSource,
    ReviewedFact,
    ReviewedPlanManifest,
    ReviewedVisual,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REVIEWED_ROOT = PROJECT_ROOT / "data" / "lhmp" / "reviewed"
CITY_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _city_directory(city: str) -> Optional[Path]:
    city_slug = (city or "").strip().lower()
    if not CITY_SLUG_PATTERN.fullmatch(city_slug):
        return None
    root = REVIEWED_ROOT.resolve()
    candidate = (root / city_slug).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def _read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _load_reviewed_list(city: str, filename: str, model: Type[BaseModel]) -> List[Dict]:
    city_directory = _city_directory(city)
    if city_directory is None:
        return []
    payload = _read_json(city_directory / filename)
    if not isinstance(payload, list):
        return []
    try:
        records = [model.model_validate(item) for item in payload]
    except (ValidationError, TypeError):
        return []
    return [record.model_dump(mode="json") for record in records]


def get_reviewed_plan(city: str) -> Dict:
    city_directory = _city_directory(city)
    if city_directory is None:
        return {}
    payload = _read_json(city_directory / "plan_manifest.json")
    if not isinstance(payload, dict):
        return {}
    try:
        record = ReviewedPlanManifest.model_validate(payload)
    except ValidationError:
        return {}
    return record.model_dump(mode="json")


def _filter_records(
    records: List[Dict],
    *,
    hazard: Optional[str] = None,
    display_location: Optional[str] = None,
) -> List[Dict]:
    hazard_key = (hazard or "").strip().lower()
    output = []
    for record in records:
        if hazard_key and record.get("hazard", "").strip().lower() != hazard_key:
            continue
        if display_location and display_location not in record.get("display_locations", []):
            continue
        output.append(record)
    return output


def get_reviewed_facts(
    city: str,
    hazard: Optional[str] = None,
    display_location: Optional[str] = None,
) -> List[Dict]:
    records = _load_reviewed_list(city, "facts.json", ReviewedFact)
    return _filter_records(records, hazard=hazard, display_location=display_location)


def get_reviewed_visuals(
    city: str,
    hazard: Optional[str] = None,
    display_location: Optional[str] = None,
) -> List[Dict]:
    records = _load_reviewed_list(city, "visuals.json", ReviewedVisual)
    return _filter_records(records, hazard=hazard, display_location=display_location)


def get_reviewed_data_sources(
    city: str,
    hazard: Optional[str] = None,
) -> List[Dict]:
    records = _load_reviewed_list(city, "data_sources.json", ReviewedDataSource)
    return _filter_records(records, hazard=hazard)
