#!/usr/bin/env python3
"""Build byte-offset indexes for memory- and CPU-bounded point checks."""

import argparse
import hashlib
import json
from pathlib import Path

from shapely.geometry import shape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "static" / "processed" / "address_point_index.json"
LAYER_PATHS = (
    ROOT / "static" / "FldHaz.geojson",
    ROOT / "static" / "FireHaz.geojson",
    ROOT / "static" / "FireResponsibilityAreas.geojson",
    ROOT / "static" / "FireHazSRA.geojson",
    ROOT / "static" / "FireHazLRARecommended.geojson",
)


def _feature_byte_ranges(raw: bytes):
    marker = b'"features":['
    marker_at = raw.find(marker)
    if marker_at < 0:
        raise ValueError("GeoJSON does not contain a features array")
    cursor = marker_at + len(marker)
    size = len(raw)
    while cursor < size:
        while cursor < size and raw[cursor] in b" \t\r\n,":
            cursor += 1
        if cursor >= size or raw[cursor] == ord("]"):
            return
        if raw[cursor] != ord("{"):
            raise ValueError(f"Expected feature object at byte {cursor}")
        start = cursor
        depth = 0
        in_string = False
        escaped = False
        while cursor < size:
            value = raw[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif value == ord("\\"):
                    escaped = True
                elif value == ord('"'):
                    in_string = False
            elif value == ord('"'):
                in_string = True
            elif value in (ord("{"), ord("[")):
                depth += 1
            elif value in (ord("}"), ord("]")):
                depth -= 1
                if depth == 0:
                    cursor += 1
                    yield start, cursor - start
                    break
            cursor += 1
        else:
            raise ValueError("Unterminated feature object")


def build_index() -> dict:
    layers = {}
    for path in LAYER_PATHS:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("type") != "FeatureCollection":
            raise ValueError(f"{path} is not a GeoJSON FeatureCollection")
        source_crs = str(
            ((payload.get("crs") or {}).get("properties") or {}).get("name")
            or ""
        )
        entries = []
        invalid_geometry_count = 0
        for offset, length in _feature_byte_ranges(raw):
            feature = json.loads(raw[offset:offset + length])
            try:
                geometry = shape(feature.get("geometry"))
            except Exception:
                invalid_geometry_count += 1
                continue
            if (
                geometry.is_empty
                or not geometry.is_valid
                or geometry.geom_type not in {"Polygon", "MultiPolygon"}
            ):
                invalid_geometry_count += 1
                continue
            min_lon, min_lat, max_lon, max_lat = geometry.bounds
            entries.append([
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                offset,
                length,
            ])
        layers[path.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "file_size": len(raw),
            "record_count": len(payload.get("features") or []),
            "valid_geometry_count": len(entries),
            "invalid_geometry_count": invalid_geometry_count,
            "source_crs": source_crs,
            "entries": entries,
        }
    return {"schema_version": 1, "layers": layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = (json.dumps(build_index(), separators=(",", ":")) + "\n").encode()
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != encoded:
            raise SystemExit("Address-point index is missing or stale. Regenerate it.")
        print(f"Address-point index is current: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    print(f"Wrote address-point index: {args.output} ({len(encoded)} bytes)")


if __name__ == "__main__":
    main()
