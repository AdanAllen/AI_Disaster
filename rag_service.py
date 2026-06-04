from typing import List

from pydantic_models import RAGChunk
from source_registry import load_official_chunks


def retrieve_chunks(hazard_type: str, jurisdiction: str = "", scope: str = "", limit: int = 4) -> List[RAGChunk]:
    hazard_key = (hazard_type or "").strip().lower()
    jurisdiction_key = (jurisdiction or "").strip().lower()
    scope_key = (scope or "").strip().lower()

    scored = []
    for chunk in load_official_chunks():
        if hazard_key and chunk.hazard_type.lower() not in {hazard_key, "all"}:
            continue

        score = 0
        if hazard_key and chunk.hazard_type.lower() == hazard_key:
            score += 3
        if jurisdiction_key and jurisdiction_key in chunk.jurisdiction.lower():
            score += 2
        if scope_key and scope_key in chunk.geographic_scope.lower():
            score += 1
        if chunk.review_status == "reviewed":
            score += 1
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:limit] if score > 0]
