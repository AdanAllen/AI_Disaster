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


@lru_cache(maxsize=1)
def load_resident_guidance_chunks() -> List[Dict]:
    return _load_json_file("resident_guidance_chunks.json", [])


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


def get_resident_guidance(city: str, hazard_type: str, limit: int = 12) -> List[Dict]:
    city_key = normalize_jurisdiction_name(city)
    hazard_key = (hazard_type or "").strip().lower()
    reviewed_statuses = {"reviewed", "draft_reviewed"}
    plan = get_local_plan_for_city(city)
    local_allowed = bool(plan and plan.get("review_status") in reviewed_statuses)

    matches = []
    if city_key and local_allowed:
        for chunk in load_resident_guidance_chunks():
            if normalize_jurisdiction_name(chunk.get("jurisdiction", "")) != city_key:
                continue
            if chunk.get("review_status") not in reviewed_statuses:
                continue
            if chunk.get("hazard_type", "").strip().lower() == hazard_key:
                matches.append(chunk)

    if matches:
        return matches[:limit]

    fallback = []
    for chunk in load_resident_guidance_chunks():
        if normalize_jurisdiction_name(chunk.get("jurisdiction", "")) not in {
            "alameda_county",
            "unincorporated_alameda_county",
        }:
            continue
        if chunk.get("review_status") not in reviewed_statuses:
            continue
        if chunk.get("hazard_type", "").strip().lower() == hazard_key:
            fallback.append(chunk)
    return fallback[:limit]


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
    local_sources = [source.model_dump() for source in load_source_records()]
    supabase_sources = _supabase_source_records_payload()
    if not supabase_sources:
        return local_sources

    existing_keys = {
        ((item.get("url") or "").strip().lower(), (item.get("name") or "").strip().lower())
        for item in local_sources
    }
    merged = list(local_sources)
    for item in supabase_sources:
        key = ((item.get("url") or "").strip().lower(), (item.get("name") or "").strip().lower())
        if key not in existing_keys:
            merged.append(item)
            existing_keys.add(key)
    return merged


def _supabase_source_records_payload() -> List[Dict]:
    try:
        from supabase_repository import fetch_sources
    except Exception:
        return []

    records = []
    for item in fetch_sources():
        source_id = item.get("id")
        title = item.get("title")
        if not source_id or not title:
            continue
        hazard_type = item.get("hazard_type") or "all"
        scope = ", ".join(part for part in [item.get("city"), item.get("county")] if part)
        trust_level = (item.get("trust_level") or "").strip().lower()
        records.append(SourceRecord(
            source_id=source_id,
            name=title,
            agency=item.get("agency") or "",
            source_type="official source",
            hazards=[hazard_type],
            geographic_scope=scope,
            claim_type="official_source",
            use_in_app="Supabase official source registry.",
            confidence="source_backed" if trust_level == "official" else "mixed_support",
            review_status="reviewed" if trust_level == "official" else "draft_reviewed",
            url=item.get("url") or "",
            notes="Loaded from Supabase Phase 1 official data table.",
            last_verified="",
        ).model_dump())
    return records
