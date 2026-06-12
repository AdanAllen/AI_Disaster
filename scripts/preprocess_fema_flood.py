#!/usr/bin/env python3
"""Build small, display-only FEMA flood GeoJSON files by ZIP code."""

import argparse
import json
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.strtree import STRtree


KEPT_PROPERTIES = (
    "DFIRM_ID",
    "VERSION_ID",
    "FLD_AR_ID",
    "STUDY_TYP",
    "FLD_ZONE",
    "ZONE_SUBTY",
    "SFHA_TF",
    "STATIC_BFE",
    "DEPTH",
    "SOURCE_CIT",
)


def read_feature_collection(path):
    with Path(path).open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="static/FldHaz.geojson")
    parser.add_argument("--zip-boundaries", default="static/zipbound.geojson")
    parser.add_argument("--output", default="static/processed/flood_by_zip")
    parser.add_argument("--tolerance", type=float, default=0.00012)
    args = parser.parse_args()

    flood = read_feature_collection(args.input)
    zip_boundaries = read_feature_collection(args.zip_boundaries)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_records = []
    for feature in zip_boundaries.get("features", []):
        zip_code = str((feature.get("properties") or {}).get("ZIP_CODE") or "")
        if not zip_code.isdigit() or len(zip_code) != 5:
            continue
        geometry = shape(feature.get("geometry"))
        if not geometry.is_empty and geometry.is_valid:
            zip_records.append((zip_code, geometry))

    tree = STRtree([record[1] for record in zip_records])
    features_by_zip = {zip_code: [] for zip_code, _ in zip_records}
    skipped = 0

    for feature in flood.get("features", []):
        try:
            geometry = shape(feature.get("geometry"))
        except Exception:
            skipped += 1
            continue
        if geometry.is_empty or not geometry.is_valid:
            skipped += 1
            continue
        properties = feature.get("properties") or {}
        kept = {key: properties.get(key) for key in KEPT_PROPERTIES if key in properties}
        kept["SOURCE"] = "FEMA National Flood Hazard Layer"

        for index in tree.query(geometry, predicate="intersects"):
            zip_code, zip_geometry = zip_records[int(index)]
            clipped = geometry.intersection(zip_geometry)
            if clipped.is_empty:
                continue
            simplified = clipped.simplify(args.tolerance, preserve_topology=True)
            if simplified.is_empty or not simplified.is_valid:
                simplified = clipped
            features_by_zip[zip_code].append({
                "type": "Feature",
                "properties": kept,
                "geometry": mapping(simplified),
            })

    manifest = {}
    for zip_code, features in features_by_zip.items():
        if not features:
            continue
        destination = output_dir / f"{zip_code}.geojson"
        destination.write_text(
            json.dumps({
                "type": "FeatureCollection",
                "features": features,
                "source": "FEMA National Flood Hazard Layer",
                "processing": {
                    "purpose": "display_only",
                    "simplified": True,
                    "clipped_to_zip": zip_code,
                    "tolerance_degrees": args.tolerance,
                },
            }, separators=(",", ":")),
            encoding="utf-8",
        )
        manifest[zip_code] = {
            "feature_count": len(features),
            "bytes": destination.stat().st_size,
        }

    (output_dir / "manifest.json").write_text(
        json.dumps({
            "source": str(args.input),
            "zip_count": len(manifest),
            "skipped_invalid_features": skipped,
            "files": manifest,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} ZIP files; skipped {skipped} invalid features.")


if __name__ == "__main__":
    main()
