import os
from functools import lru_cache
from typing import Dict, Optional

from supabase import create_client

from supabase_client import is_valid_project_url


@lru_cache(maxsize=1)
def get_submission_client() -> Optional[object]:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    secret_key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()
    if not url or not secret_key or not is_valid_project_url(url):
        return None

    try:
        return create_client(url, secret_key)
    except Exception:
        return None


def save_email_interest(payload: Dict) -> bool:
    client = get_submission_client()
    if client is None:
        return False

    try:
        response = (
            client.table("email_interests")
            .upsert(payload, on_conflict="email")
            .execute()
        )
    except Exception:
        return False
    return response.data is not None


def save_feedback_submission(payload: Dict) -> bool:
    client = get_submission_client()
    if client is None:
        return False

    try:
        response = client.table("feedback_submissions").insert(payload).execute()
    except Exception:
        return False
    return response.data is not None
