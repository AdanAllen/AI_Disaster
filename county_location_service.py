"""County-authoritative address resolution for StayReady.

The public Alameda County Address Points service is the primary address
authority. Jurisdiction is assigned only by official city and unincorporated
community polygons. ArcGIS World Geocoding is an authenticated fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import math
import os
import re
import secrets
from typing import Any

import requests

from pydantic_models import ResolvedLocation


ADDRESS_LAYER_URL = (
    "https://services5.arcgis.com/ROBnTHSNjoZ2Wm1P/arcgis/rest/services/"
    "Address_Points/FeatureServer/0"
)
CITY_BOUNDARY_LAYER_URL = (
    "https://services5.arcgis.com/ROBnTHSNjoZ2Wm1P/arcgis/rest/services/"
    "City_Boundaries/FeatureServer/0"
)
UNINCORPORATED_BOUNDARY_LAYER_URL = (
    "https://services7.arcgis.com/6Yz0DcWQc1LtdRPC/arcgis/rest/services/"
    "Unincorporated_Area_Boundaries/FeatureServer/0"
)
WORLD_GEOCODER_URL = (
    "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer"
)
ALAMEDA_SEARCH_EXTENT = "-122.38,37.45,-121.46,37.91"
REQUEST_TIMEOUT = (3.05, 8)
ACCEPTED_WORLD_MATCH_TYPES = {"PointAddress", "Subaddress", "StreetAddress"}
PO_BOX_RE = re.compile(r"\bP(?:OST(?:AL)?)?\.?\s*O(?:FFICE)?\.?\s*BOX\b", re.I)
UNIT_RE = re.compile(
    r"\s+(?:APT|APARTMENT|UNIT|STE|SUITE|#)\s*[A-Z0-9-]+\s*(?=,|$)",
    re.I,
)


class LocationServiceError(RuntimeError):
    """The location authority could not safely complete the request."""


class UnsupportedLocationInput(ValueError):
    """The input needs a documented fallback rather than address evidence."""


@dataclass(frozen=True)
class LocationCandidate:
    candidate_id: str
    display_label: str
    latitude: float
    longitude: float
    zip_code: str
    provider: str
    match_type: str
    match_score: float
    source_object_id: str = ""
    unit: str = ""
    limitations: tuple[str, ...] = ()

    def public_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_label": self.display_label,
            "match_type": self.match_type,
            "match_score": self.match_score,
            "provider": self.provider,
        }

    def stored_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "display_label": self.display_label,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "zip_code": self.zip_code,
            "provider": self.provider,
            "match_type": self.match_type,
            "match_score": self.match_score,
            "source_object_id": self.source_object_id,
            "unit": self.unit,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_stored_payload(cls, payload: dict[str, Any]) -> "LocationCandidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            display_label=str(payload["display_label"]),
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            zip_code=str(payload.get("zip_code") or ""),
            provider=str(payload["provider"]),
            match_type=str(payload["match_type"]),
            match_score=float(payload["match_score"]),
            source_object_id=str(payload.get("source_object_id") or ""),
            unit=str(payload.get("unit") or ""),
            limitations=tuple(payload.get("limitations") or ()),
        )


def _arcgis_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        url,
        params={"f": "json", **params},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "StayReadyNow/2.0 location-service"},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise LocationServiceError(payload["error"].get("message", "ArcGIS request failed"))
    return payload


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _normalize_lookup(value: str) -> str:
    value = UNIT_RE.sub("", str(value or "").strip())
    value = re.sub(r"[^A-Za-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().upper()
    return value


def _display_address(attributes: dict[str, Any]) -> str:
    address = str(attributes.get("ADDRESS") or "").strip()
    unit = str(attributes.get("UNIT") or "").strip()
    unit_type = str(attributes.get("UNIT_TYP") or "Unit").strip()
    if unit and unit.casefold() not in address.casefold():
        address = f"{address}, {unit_type} {unit}"
    city = str(attributes.get("CITY") or attributes.get("MUN") or "").strip()
    zip_code = str(attributes.get("ZIPCODE") or "").strip()
    parts = [part for part in (address, city, "CA", zip_code) if part]
    return ", ".join(parts)


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _county_candidates(query: str, limit: int) -> list[LocationCandidate]:
    lookup = _normalize_lookup(query)
    if not lookup:
        return []
    # ADDRESSLOOKUP is the county-maintained search field. A prefix match keeps
    # this endpoint deterministic and prevents broad table scans.
    payload = _arcgis_get(
        f"{ADDRESS_LAYER_URL}/query",
        {
            "where": f"ADDRESSLOOKUP LIKE '{_sql_literal(lookup)}%'",
            "outFields": "OBJECTID,ADDRESS,CITY,MUN,ZIPCODE,UNIT,UNIT_TYP,ADDRESSLOOKUP",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": min(max(limit * 4, 10), 50),
            "orderByFields": "ADDRESS ASC",
        },
    )
    results: list[LocationCandidate] = []
    seen: set[tuple[str, float, float]] = set()
    for feature in payload.get("features", []):
        attributes = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        if geometry.get("x") is None or geometry.get("y") is None:
            continue
        label = _display_address(attributes)
        key = (label.casefold(), round(float(geometry["y"]), 7), round(float(geometry["x"]), 7))
        if key in seen:
            continue
        seen.add(key)
        unit = str(attributes.get("UNIT") or "").strip()
        results.append(
            LocationCandidate(
                candidate_id=secrets.token_urlsafe(18),
                display_label=label,
                latitude=float(geometry["y"]),
                longitude=float(geometry["x"]),
                zip_code=str(attributes.get("ZIPCODE") or "").strip()[:5],
                provider="alameda_county_address_points",
                match_type="subaddress" if unit else "address_point",
                match_score=100.0 if _normalize_lookup(label).startswith(lookup) else 98.0,
                source_object_id=str(attributes.get("OBJECTID") or ""),
                unit=unit,
            )
        )
        if len(results) >= limit:
            break
    return results


def _world_candidates(query: str, limit: int) -> list[LocationCandidate]:
    token = (os.getenv("ARCGIS_GEOCODING_TOKEN") or "").strip()
    if not token:
        return []
    payload = _arcgis_get(
        f"{WORLD_GEOCODER_URL}/findAddressCandidates",
        {
            "token": token,
            "SingleLine": query,
            "category": "Address",
            "searchExtent": ALAMEDA_SEARCH_EXTENT,
            "outFields": "Addr_type,Postal,Match_addr",
            "maxLocations": min(limit, 5),
            "locationType": "rooftop",
            "forStorage": "false",
        },
    )
    candidates = []
    for item in payload.get("candidates", []):
        attributes = item.get("attributes") or {}
        location = item.get("location") or {}
        match_type = attributes.get("Addr_type")
        score = float(item.get("score") or 0)
        if match_type not in ACCEPTED_WORLD_MATCH_TYPES or score < 90:
            continue
        candidates.append(
            LocationCandidate(
                candidate_id=secrets.token_urlsafe(18),
                display_label=str(item.get("address") or attributes.get("Match_addr") or ""),
                latitude=float(location["y"]),
                longitude=float(location["x"]),
                zip_code=str(attributes.get("Postal") or "")[:5],
                provider="arcgis_world_geocoder",
                match_type={
                    "PointAddress": "address_point",
                    "Subaddress": "subaddress",
                    "StreetAddress": "street",
                }[match_type],
                match_score=score,
                limitations=(
                    "This address was not found in the county snapshot and used the authenticated "
                    "ArcGIS fallback. Confirm the displayed match before continuing.",
                ),
            )
        )
    return candidates


def suggest_locations(query: str, limit: int = 5) -> list[LocationCandidate]:
    if PO_BOX_RE.search(query or ""):
        raise UnsupportedLocationInput(
            "PO boxes cannot receive address-level mapped evidence. Use a ZIP code or community."
        )
    county = _county_candidates(query, limit)
    if county:
        return county
    return _world_candidates(query, limit)


def nearest_address_candidate(latitude: float, longitude: float) -> LocationCandidate | None:
    payload = _arcgis_get(
        f"{ADDRESS_LAYER_URL}/query",
        {
            "where": "1=1",
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "distance": 150,
            "units": "esriSRUnit_Meter",
            "outFields": "OBJECTID,ADDRESS,CITY,MUN,ZIPCODE,UNIT,UNIT_TYP",
            "returnGeometry": "true",
            "outSR": "4326",
            "resultRecordCount": 20,
        },
    )
    choices = []
    for feature in payload.get("features", []):
        attributes = feature.get("attributes") or {}
        geometry = feature.get("geometry") or {}
        if geometry.get("x") is None or geometry.get("y") is None:
            continue
        candidate = LocationCandidate(
            candidate_id=secrets.token_urlsafe(18),
            display_label=_display_address(attributes),
            latitude=float(geometry["y"]),
            longitude=float(geometry["x"]),
            zip_code=str(attributes.get("ZIPCODE") or "")[:5],
            provider="alameda_county_address_points",
            match_type="subaddress" if attributes.get("UNIT") else "address_point",
            match_score=100,
            source_object_id=str(attributes.get("OBJECTID") or ""),
            unit=str(attributes.get("UNIT") or ""),
        )
        choices.append(
            (
                _haversine_meters(
                    latitude,
                    longitude,
                    candidate.latitude,
                    candidate.longitude,
                ),
                candidate,
            )
        )
    if not choices:
        return None
    distance, nearest = min(choices, key=lambda item: item[0])
    if distance > 150:
        return None
    return nearest


def _point_boundary(
    layer_url: str,
    latitude: float,
    longitude: float,
    out_fields: str,
) -> list[dict[str, Any]]:
    payload = _arcgis_get(
        f"{layer_url}/query",
        {
            "where": "1=1",
            "geometry": f"{longitude},{latitude}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": out_fields,
            "returnGeometry": "false",
        },
    )
    return [item.get("attributes") or {} for item in payload.get("features", [])]


@lru_cache(maxsize=2)
def _layer_version(layer_url: str) -> str:
    payload = _arcgis_get(layer_url, {})
    edit_fields = payload.get("editingInfo") or {}
    timestamp = edit_fields.get("lastEditDate")
    if timestamp:
        updated = datetime.fromtimestamp(float(timestamp) / 1000, tz=timezone.utc)
        return updated.date().isoformat()
    return str(payload.get("currentVersion") or "version-unknown")


def resolve_jurisdiction(
    latitude: float,
    longitude: float,
) -> tuple[str, str, bool, str]:
    city_records = _point_boundary(
        CITY_BOUNDARY_LAYER_URL,
        latitude,
        longitude,
        "*",
    )
    city_names = []
    for record in city_records:
        for key in ("CITY", "City", "CITY_NAME", "CityName", "NAME", "Name"):
            if record.get(key):
                city_names.append(str(record[key]).strip())
                break
    incorporated_names = sorted(
        {name for name in city_names if name.casefold() != "alameda county"}
    )
    if len(incorporated_names) > 1:
        raise UnsupportedLocationInput(
            "The point lies on a shared city boundary. StayReady cannot safely assign "
            "a jurisdiction without human review."
        )
    incorporated = incorporated_names[0] if incorporated_names else ""
    is_unincorporated = bool(
        not incorporated
        and any(name.casefold() == "alameda county" for name in city_names)
    )
    community = ""
    if not incorporated:
        records = _point_boundary(
            UNINCORPORATED_BOUNDARY_LAYER_URL,
            latitude,
            longitude,
            "*",
        )
        communities = []
        for record in records:
            for key in ("Area_name", "AREA_NAME", "NAME", "Name"):
                if record.get(key):
                    communities.append(str(record[key]).strip())
                    break
        communities = sorted(set(communities))
        if len(communities) > 1:
            raise UnsupportedLocationInput(
                "The point lies on a shared unincorporated-community boundary. "
                "StayReady cannot safely choose one."
            )
        community = communities[0] if communities else ""
    if not incorporated and not is_unincorporated and not community and not city_records:
        raise UnsupportedLocationInput(
            "That point is outside Alameda County. No result was created."
        )
    version = (
        f"city:{_layer_version(CITY_BOUNDARY_LAYER_URL)};"
        f"unincorporated:{_layer_version(UNINCORPORATED_BOUNDARY_LAYER_URL)}"
    )
    return incorporated, community, is_unincorporated, version


def resolve_candidate(candidate: LocationCandidate) -> ResolvedLocation:
    incorporated, community, is_unincorporated, boundary_version = resolve_jurisdiction(
        candidate.latitude,
        candidate.longitude,
    )
    limitations = list(candidate.limitations)
    if candidate.match_type == "subaddress":
        limitations.append(
            "The unit helped confirm the address, but mapped hazards use the building address point."
        )
    return ResolvedLocation(
        match_id=candidate.candidate_id,
        normalized_display_address=candidate.display_label,
        latitude=candidate.latitude,
        longitude=candidate.longitude,
        zip_code=candidate.zip_code,
        incorporated_jurisdiction=incorporated,
        unincorporated_community=community,
        is_unincorporated=is_unincorporated,
        match_type=candidate.match_type,
        match_score=candidate.match_score,
        precision=(
            "address_point"
            if candidate.match_type in {"address_point", "subaddress"}
            else "unknown"
        ),
        provider=candidate.provider,
        boundary_version=boundary_version,
        limitations=limitations,
    )
