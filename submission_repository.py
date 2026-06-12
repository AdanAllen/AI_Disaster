import os
import base64
import json
from functools import lru_cache
from typing import Dict, Optional

from supabase import create_client

from supabase_client import is_valid_project_url


def submission_key_type(value: str) -> str:
    key = str(value or "").strip()
    if key.startswith("sb_secret_"):
        return "secret"
    if key.startswith("sb_publishable_"):
        return "publishable"
    if key.count(".") == 2:
        try:
            payload = key.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            role = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8")).get("role")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return "invalid"
        if role == "service_role":
            return "service_role"
        if role == "anon":
            return "anon"
    return "invalid"


def get_submission_config_status() -> Dict:
    url = (os.getenv("SUPABASE_URL") or "").strip()
    secret_key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()
    key_type = submission_key_type(secret_key)
    return {
        "configured": bool(
            url
            and is_valid_project_url(url)
            and key_type in {"secret", "service_role"}
        ),
        "project_url_configured": bool(url and is_valid_project_url(url)),
        "secret_key_configured": bool(secret_key),
        "credential_type": key_type,
    }


@lru_cache(maxsize=1)
def get_submission_client() -> Optional[object]:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    secret_key = (os.getenv("SUPABASE_SECRET_KEY") or "").strip()
    if not get_submission_config_status()["configured"]:
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
