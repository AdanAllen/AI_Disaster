import os
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse


def supabase_enabled() -> bool:
    return (os.getenv("SUPABASE_ENABLED") or "false").strip().lower() == "true"


def get_supabase_config_status() -> dict:
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    return {
        "enabled": supabase_enabled(),
        "configured": bool(url and key and _is_valid_project_url(url)),
    }


def _is_valid_project_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False
    if not parsed.netloc.endswith(".supabase.co"):
        return False
    if parsed.path not in {"", "/"}:
        return False
    return True


@lru_cache(maxsize=1)
def get_supabase_client() -> Optional[object]:
    if not supabase_enabled():
        return None

    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    if not url or not key or not _is_valid_project_url(url):
        return None

    try:
        from supabase import create_client
    except Exception:
        return None

    try:
        return create_client(url, key)
    except Exception:
        return None
