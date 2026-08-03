"""Memory- and CPU-bounded helpers for local GeoJSON point checks."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import ijson
from shapely.geometry import Point, shape


POLYGON_TYPES = {"Polygon", "MultiPolygon"}
INDEX_FILENAME = "address_point_index.json"
MAX_UNINDEXED_SCAN_BYTES = 4_000_000


@lru_cache(maxsize=2)
def _load_point_index(path_string: str, modified_ns: int, file_size: int):
    del modified_ns, file_size
    try:
        with Path(path_string).open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if payload.get("schema_version") != 1 or not isinstance(payload.get("layers"), dict):
        return {}
    return payload["layers"]


@lru_cache(maxsize=8)
def _source_matches_index(
    path_string: str,
    modified_ns: int,
    file_size: int,
    expected_sha256: str,
) -> bool:
    del modified_ns
    if file_size <= 0:
        return False
    digest = hashlib.sha256()
    try:
        with Path(path_string).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == expected_sha256


def _indexed_point_scan(path: Path, latitude: float, longitude: float):
    index_path = path.parent / "processed" / INDEX_FILENAME
    try:
        index_stat = index_path.stat()
        source_stat = path.stat()
    except OSError:
        return None
    layers = _load_point_index(
        str(index_path),
        index_stat.st_mtime_ns,
        index_stat.st_size,
    )
    layer = layers.get(path.name)
    if not isinstance(layer, dict):
        return None
    if source_stat.st_size != layer.get("file_size") or not _source_matches_index(
        str(path),
        source_stat.st_mtime_ns,
        source_stat.st_size,
        str(layer.get("sha256") or ""),
    ):
        return None

    matched_properties = []
    point = Point(float(longitude), float(latitude))
    try:
        with path.open("rb") as source:
            for min_lon, min_lat, max_lon, max_lat, offset, length in layer["entries"]:
                if not (
                    min_lon <= longitude <= max_lon
                    and min_lat <= latitude <= max_lat
                ):
                    continue
                source.seek(offset)
                feature = json.loads(source.read(length))
                geometry_shape = shape(feature.get("geometry"))
                if geometry_shape.contains(point) or geometry_shape.touches(point):
                    matched_properties.append(dict(feature.get("properties") or {}))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return {"error": "spatial_index"}
    return {
        "record_count": layer["record_count"],
        "valid_geometry_count": layer["valid_geometry_count"],
        "invalid_geometry_count": layer["invalid_geometry_count"],
        "matched_properties": matched_properties,
        "source_crs": layer.get("source_crs") or "",
        "indexed": True,
    }


def _first_item(path: Path, prefix: str):
    with path.open("rb") as source:
        return next(ijson.items(source, prefix, use_float=True), None)


def read_geojson_header(path: Path) -> Dict[str, Any]:
    """Read the small top-level fields without materializing ``features``."""
    path = Path(path)
    try:
        return {
            "type": _first_item(path, "type"),
            "crs": _first_item(path, "crs"),
        }
    except (OSError, UnicodeError, ValueError, ijson.JSONError):
        return {"error": "json"}


def scan_polygon_features_at_point(
    path: Path,
    *,
    latitude: float,
    longitude: float,
    permitted_geometry_types: Iterable[str] = POLYGON_TYPES,
) -> Dict[str, Any]:
    """Scan one feature at a time and retain only properties that match a point.

    Loading the repository's full FEMA and CAL FIRE layers with ``json.load``
    can expand tens of megabytes of JSON into hundreds of megabytes of Python
    and GEOS objects.  This scanner keeps memory proportional to one feature.
    """
    path = Path(path)
    indexed = _indexed_point_scan(path, float(latitude), float(longitude))
    if indexed is not None:
        return indexed
    try:
        if path.stat().st_size > MAX_UNINDEXED_SCAN_BYTES:
            return {"error": "spatial_index"}
    except OSError:
        return {"error": "json"}

    header = read_geojson_header(path)
    if header.get("error"):
        return {"error": header["error"]}
    if header.get("type") != "FeatureCollection":
        return {"error": "schema"}

    allowed = set(permitted_geometry_types)
    point = Point(float(longitude), float(latitude))
    record_count = 0
    valid_geometry_count = 0
    invalid_geometry_count = 0
    matched_properties = []

    try:
        with path.open("rb") as source:
            for feature in ijson.items(
                source,
                "features.item",
                use_float=True,
            ):
                record_count += 1
                geometry = feature.get("geometry") if isinstance(feature, dict) else None
                if not geometry:
                    invalid_geometry_count += 1
                    continue
                try:
                    geometry_shape = shape(geometry)
                except Exception:
                    invalid_geometry_count += 1
                    continue
                if (
                    geometry_shape.is_empty
                    or not geometry_shape.is_valid
                    or geometry_shape.geom_type not in allowed
                ):
                    invalid_geometry_count += 1
                    continue
                valid_geometry_count += 1
                if geometry_shape.contains(point) or geometry_shape.touches(point):
                    matched_properties.append(dict(feature.get("properties") or {}))
    except (OSError, UnicodeError, ValueError, ijson.JSONError):
        return {"error": "json"}

    if record_count == 0:
        return {"error": "schema"}
    if valid_geometry_count == 0:
        return {"error": "geometry", "record_count": record_count}
    return {
        "record_count": record_count,
        "valid_geometry_count": valid_geometry_count,
        "invalid_geometry_count": invalid_geometry_count,
        "matched_properties": matched_properties,
        "source_crs": str(
            ((header.get("crs") or {}).get("properties") or {}).get("name")
            or ""
        ),
    }
