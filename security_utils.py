"""Small, dependency-free security helpers for public StayReady routes."""

from collections import defaultdict, deque
from functools import wraps
import hashlib
import ipaddress
import os
import re
import threading
import time
from urllib.parse import urlparse

from flask import jsonify, redirect, request, url_for


ADDRESS_MAX_LENGTH = 180
QUERY_MAX_LENGTH = 80
ALLOWED_HAZARD_SLUGS = {
    "dam-failure",
    "drought",
    "earthquake",
    "extreme-heat",
    "flood",
    "high-wind",
    "landslide",
    "poor-air-quality",
    "sea-level-rise",
    "severe-weather",
    "tsunami",
    "tsunami-seiche",
    "utility-disruption",
    "wildfire",
}
ALLOWED_CGS_LAYER_KEYS = {
    "alquist-priolo",
    "liquefaction",
    "earthquake-landslide",
    "tsunami",
}
ALLOWED_REMOTE_HOSTS = {
    "earthquake.usgs.gov",
    "geocode-api.arcgis.com",
    "geocode.arcgis.com",
    "gis.conservation.ca.gov",
    "services.arcgis.com",
    "services2.arcgis.com",
    "services5.arcgis.com",
    "services7.arcgis.com",
}

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_ADDRESS_CHARACTERS = re.compile(r"^[A-Za-z0-9\s,.'#;/()-]+$")
_rate_buckets = defaultdict(deque)
_rate_lock = threading.Lock()
_redis_rate_client = None


def _shared_rate_client():
    global _redis_rate_client
    if _redis_rate_client is not None:
        return _redis_rate_client
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    if not redis_url:
        return None
    try:
        import redis

        _redis_rate_client = redis.Redis.from_url(
            redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        return _redis_rate_client
    except Exception:
        return None


def valid_zip(value):
    return bool(re.fullmatch(r"\d{5}", str(value or "").strip()))


def valid_address(value):
    text = str(value or "").strip()
    return (
        5 <= len(text) <= ADDRESS_MAX_LENGTH
        and not _CONTROL_CHARACTERS.search(text)
        and bool(_ADDRESS_CHARACTERS.fullmatch(text))
    )


def valid_short_query(value):
    text = str(value or "").strip()
    return (
        len(text) <= QUERY_MAX_LENGTH
        and not _CONTROL_CHARACTERS.search(text)
    )


def valid_coordinate_pair(lat, lon, bounds):
    try:
        lat_value = float(lat)
        lon_value = float(lon)
    except (TypeError, ValueError):
        return None
    if not (
        bounds["min_lat"] <= lat_value <= bounds["max_lat"]
        and bounds["min_lon"] <= lon_value <= bounds["max_lon"]
    ):
        return None
    return lat_value, lon_value


def allowed_remote_url(url):
    parsed = urlparse(str(url or ""))
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        return False
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname not in ALLOWED_REMOTE_HOSTS:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def safe_next_url(value, fallback_endpoint):
    parsed = urlparse(str(value or ""))
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return url_for(fallback_endpoint)
    return parsed.path


def client_key():
    # Render/Cloudflare should be configured to replace, not append, trusted
    # forwarding headers. The right-most value is closest to this application.
    forwarded = request.headers.get("X-Forwarded-For", "")
    candidate = (
        forwarded.split(",")[-1].strip()
        if forwarded and os.getenv("RENDER")
        else request.remote_addr
    )
    try:
        return str(ipaddress.ip_address(candidate or ""))
    except ValueError:
        return "unknown"


def _consume_shared_limit(name, client, limit, window_seconds):
    redis_client = _shared_rate_client()
    if redis_client is None:
        return None
    bucket = int(time.time() // window_seconds)
    opaque_client = hashlib.sha256(client.encode()).hexdigest()[:24]
    key = f"stayready:rate:{name}:{opaque_client}:{bucket}"
    try:
        current = redis_client.incr(key)
        if current == 1:
            redis_client.expire(key, window_seconds + 2)
        return current <= limit
    except Exception:
        return None


def rate_limit(limit, window_seconds):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            now = time.monotonic()
            client = client_key()
            shared_allowed = _consume_shared_limit(
                function.__name__,
                client,
                limit,
                window_seconds,
            )
            if shared_allowed is False:
                response = jsonify({
                    "error": "Too many requests. Please wait and try again.",
                    "retry_after_seconds": window_seconds,
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(window_seconds)
                return response
            key = (function.__name__, client)
            if shared_allowed is True:
                return function(*args, **kwargs)
            with _rate_lock:
                bucket = _rate_buckets[key]
                while bucket and bucket[0] <= now - window_seconds:
                    bucket.popleft()
                if len(bucket) >= limit:
                    retry_after = max(1, int(window_seconds - (now - bucket[0])))
                    response = jsonify({
                        "error": "Too many requests. Please wait and try again.",
                        "retry_after_seconds": retry_after,
                    })
                    response.status_code = 429
                    response.headers["Retry-After"] = str(retry_after)
                    return response
                bucket.append(now)
            return function(*args, **kwargs)
        return wrapped
    return decorator


def form_rate_limit(limit, window_seconds, endpoint, status_value, anchor=""):
    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            now = time.monotonic()
            client = client_key()
            shared_allowed = _consume_shared_limit(
                function.__name__,
                client,
                limit,
                window_seconds,
            )
            if shared_allowed is False:
                location = url_for(endpoint, **status_value)
                response = redirect(f"{location}{anchor}")
                response.status_code = 302
                response.headers["Retry-After"] = str(window_seconds)
                return response
            if shared_allowed is True:
                return function(*args, **kwargs)
            key = (function.__name__, client)
            with _rate_lock:
                bucket = _rate_buckets[key]
                while bucket and bucket[0] <= now - window_seconds:
                    bucket.popleft()
                if len(bucket) >= limit:
                    retry_after = max(1, int(window_seconds - (now - bucket[0])))
                    location = url_for(endpoint, **status_value)
                    response = redirect(f"{location}{anchor}")
                    response.status_code = 302
                    response.headers["Retry-After"] = str(retry_after)
                    return response
                bucket.append(now)
            return function(*args, **kwargs)
        return wrapped
    return decorator


def reset_rate_limits():
    with _rate_lock:
        _rate_buckets.clear()
