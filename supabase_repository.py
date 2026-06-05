from typing import Dict, List, Optional

from supabase_client import get_supabase_client, get_supabase_config_status


TABLES = ("cities", "sources", "hazards", "document_chunks")


def _execute_select(table: str, filters: Optional[Dict] = None, limit: int = 500) -> List[Dict]:
    client = get_supabase_client()
    if client is None:
        return []

    try:
        query = client.table(table).select("*").limit(limit)
        for key, value in (filters or {}).items():
            if value not in {None, ""}:
                query = query.eq(key, value)
        response = query.execute()
    except Exception:
        return []

    return response.data or []


def fetch_sources(hazard_type: Optional[str] = None, city: Optional[str] = None) -> List[Dict]:
    filters = {}
    if hazard_type:
        filters["hazard_type"] = hazard_type
    if city:
        filters["city"] = city
    return _execute_select("sources", filters)


def fetch_cities(county: Optional[str] = None) -> List[Dict]:
    filters = {"county": county} if county else {}
    return _execute_select("cities", filters)


def fetch_hazards(city_id: Optional[str] = None, hazard_type: Optional[str] = None) -> List[Dict]:
    filters = {}
    if city_id:
        filters["city_id"] = city_id
    if hazard_type:
        filters["hazard_type"] = hazard_type
    return _execute_select("hazards", filters)


def fetch_document_chunks(
    city: Optional[str] = None,
    hazard_type: Optional[str] = None,
    review_status: Optional[str] = None,
) -> List[Dict]:
    filters = {}
    if city:
        filters["city"] = city
    if hazard_type:
        filters["hazard_type"] = hazard_type
    if review_status:
        filters["review_status"] = review_status
    return _execute_select("document_chunks", filters)


def _safe_count(table: str) -> Dict:
    client = get_supabase_client()
    if client is None:
        return {"ok": False, "count": None}

    try:
        response = client.table(table).select("id", count="exact").limit(1).execute()
        return {"ok": True, "count": response.count}
    except Exception:
        return {"ok": False, "count": None}


def supabase_health() -> Dict:
    status = get_supabase_config_status()
    tables = {table: {"ok": False, "count": None} for table in TABLES}
    if not status["enabled"]:
        return {
            **status,
            "connected": False,
            "tables": tables,
            "error": "Supabase is disabled.",
        }
    if not status["configured"]:
        return {
            **status,
            "connected": False,
            "tables": tables,
            "error": "Supabase is not configured.",
        }

    tables = {table: _safe_count(table) for table in TABLES}
    connected = all(value["ok"] for value in tables.values())
    return {
        **status,
        "connected": connected,
        "tables": tables,
        "error": None if connected else "Supabase connection check failed.",
    }
