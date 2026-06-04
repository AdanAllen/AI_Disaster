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
    ]


def source_records_payload() -> List[Dict]:
    return [source.model_dump() for source in load_source_records()]
