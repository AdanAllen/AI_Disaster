import json
import os
from functools import lru_cache
from typing import Dict, List

from pydantic_models import RAGChunk, SourceRecord


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json_file(filename, default):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


@lru_cache(maxsize=1)
def load_source_records() -> List[SourceRecord]:
    return [SourceRecord(**item) for item in _load_json_file("sources.json", [])]


@lru_cache(maxsize=1)
def load_hazard_registry() -> List[Dict]:
    return _load_json_file("hazards.json", [])


@lru_cache(maxsize=1)
def load_jurisdictions() -> List[Dict]:
    return _load_json_file("jurisdictions.json", [])


@lru_cache(maxsize=1)
def load_official_chunks() -> List[RAGChunk]:
    return [RAGChunk(**item) for item in _load_json_file("official_chunks.json", [])]


@lru_cache(maxsize=1)
def load_local_plans() -> List[Dict]:
    return _load_json_file("local_plans.json", [])


@lru_cache(maxsize=1)
def load_city_hazard_chunks() -> List[Dict]:
    return _load_json_file("city_hazard_chunks.json", [])


def normalize_jurisdiction_name(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


def get_local_plan_for_city(city: str) -> Dict:
    city_key = normalize_jurisdiction_name(city)
    for plan in load_local_plans():
        if normalize_jurisdiction_name(plan.get("name", "")) == city_key:
            return plan
    return {}


def get_city_chunks(city: str, hazard_type: str) -> List[Dict]:
    city_key = (city or "").strip().lower()
    hazard_key = (hazard_type or "").strip().lower()
    if not city_key:
        return []

    chunks = []
    for chunk in load_city_hazard_chunks():
        if chunk.get("jurisdiction", "").strip().lower() != city_key:
            continue
        if chunk.get("review_status") not in {"reviewed", "draft_reviewed"}:
            continue
        chunk_hazard = chunk.get("hazard_type", "").strip().lower()
        if chunk_hazard in {hazard_key, "all"}:
            chunks.append(chunk)
    return chunks


def get_source(source_id: str) -> SourceRecord:
    for source in load_source_records():
        if source.source_id == source_id:
            return source
    return SourceRecord(
        source_id=source_id or "unknown",
        name="Source under review",
        agency="StayReady",
        source_type="registry_placeholder",
        confidence="needs_review",
        review_status="draft",
        notes="This source has not been fully registered yet.",
    )


def get_sources_for_hazard(hazard_type: str) -> List[SourceRecord]:
    hazard_key = (hazard_type or "").strip().lower()
    return [
        source
        for source in load_source_records()
        if hazard_key in {item.lower() for item in source.hazards}
        and source.source_type != "local hazard mitigation plan"
    ]


def source_records_payload() -> List[Dict]:
    return [source.model_dump() for source in load_source_records()]
