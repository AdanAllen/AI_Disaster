"""Memory-bounded helpers for address-point checks against local GeoJSON."""

from pathlib import Path
from typing import Any, Dict, Iterable

import ijson
from shapely.geometry import Point, shape


POLYGON_TYPES = {"Polygon", "MultiPolygon"}


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
