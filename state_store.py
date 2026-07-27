"""Ephemeral shared state selection for resident results and rate limits."""

from __future__ import annotations

import os
import tempfile

from cachelib import FileSystemCache, RedisCache


def create_ephemeral_store(default_timeout: int):
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    if redis_url:
        try:
            import redis

            client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
            return RedisCache(
                host=client,
                default_timeout=default_timeout,
                key_prefix="stayready:",
            )
        except Exception as error:
            if (os.getenv("FLASK_ENV") or "").lower() == "production" or os.getenv("RENDER"):
                raise RuntimeError("REDIS_URL is configured but shared state could not start.") from error

    return FileSystemCache(
        cache_dir=os.path.join(tempfile.gettempdir(), "stayready-resident-state"),
        threshold=1000,
        default_timeout=default_timeout,
    )


def store_backend_name(store) -> str:
    return "redis" if isinstance(store, RedisCache) else "filesystem"


def store_is_ready(store) -> bool:
    probe_key = "readiness-probe"
    try:
        store.set(probe_key, {"ok": True}, timeout=5)
        ready = store.get(probe_key) == {"ok": True}
        store.delete(probe_key)
        return ready
    except Exception:
        return False
