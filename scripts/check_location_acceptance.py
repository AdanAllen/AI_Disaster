#!/usr/bin/env python3
"""Validate the public/civic countywide address acceptance matrix."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from county_location_service import resolve_candidate, suggest_locations  # noqa: E402


FIXTURE_PATH = ROOT / "data" / "location_acceptance_fixtures.json"
INCORPORATED_CITIES = {
    "Alameda",
    "Albany",
    "Berkeley",
    "Dublin",
    "Emeryville",
    "Fremont",
    "Hayward",
    "Livermore",
    "Newark",
    "Oakland",
    "Piedmont",
    "Pleasanton",
    "San Leandro",
    "Union City",
}
COMMUNITIES = {
    "Ashland",
    "Castro Valley",
    "Cherryland",
    "Fairview",
    "Hayward Acres",
    "San Lorenzo",
    "Sunol",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Query the official county services and verify every fixture.",
    )
    args = parser.parse_args()
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    cities = {item["jurisdiction"] for item in fixtures} & INCORPORATED_CITIES
    communities = {item["community"] for item in fixtures if item["community"]}
    errors = []
    if cities != INCORPORATED_CITIES:
        errors.append(f"incorporated city coverage differs: {sorted(INCORPORATED_CITIES - cities)}")
    if communities != COMMUNITIES:
        errors.append(f"community coverage differs: {sorted(COMMUNITIES - communities)}")

    if args.live:
        for item in fixtures:
            candidates = suggest_locations(item["address"], limit=5)
            resolved = [resolve_candidate(candidate) for candidate in candidates]
            if not any(
                result.jurisdiction == item["jurisdiction"]
                and result.unincorporated_community == item["community"]
                for result in resolved
            ):
                errors.append(f"fixture did not resolve as expected: {item['address']}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"Location acceptance matrix passed: {len(fixtures)} public/civic fixtures"
        + (" checked live." if args.live else " structurally checked.")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
