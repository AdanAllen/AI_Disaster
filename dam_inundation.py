"""Official DWR/DSOD dam-failure inundation boundary checks."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from geospatial.models import GeospatialEvidence
from geospatial.registry import DatasetRegistryError
from geospatial.service import GeospatialEvidenceService


DATASET_ID = "dwr_dsod_dam_inundation_remote"
SOURCE_AGENCY = "California Department of Water Resources, Division of Safety of Dams"
SOURCE_URL = (
    "https://services.arcgis.com/aa38u6OgfNoCkTJ6/arcgis/rest/services/"
    "Approved_InundationBoundaries_As_of_Oct01_2025/FeatureServer/100"
)
SOURCE_LANDING_URL = (
    "https://water.ca.gov/programs/all-programs/division-of-safety-of-dams/"
    "inundation-maps"
)
BASE_LIMITATIONS = [
    "Dam inundation boundaries show hypothetical dam or critical-structure failure scenarios for emergency planning; they do not predict that a failure will occur.",
    "A non-match does not establish that a location is safe from flooding, infrastructure failure, indirect impacts, or evacuation disruption.",
    "Published boundaries are approximate, may omit federal dams or redacted information, and may change as DSOD approves updated maps.",
    "Actual evacuation zones, routes, timing, and instructions come from local emergency officials.",
    "StayReady does not use LHMP or EOP screenshots for address-level dam inundation conclusions.",
]


def _published_date(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date().isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _scenario(attributes: Dict[str, Any]) -> Dict[str, Any]:
    dam_name = str(attributes.get("DamName") or "").strip()
    failed_structure = str(attributes.get("FailedStr") or "").strip()
    scenario = str(attributes.get("Scenario") or "").strip()
    loading_scenario = str(attributes.get("LoadingScn") or "").strip()
    parts = [part for part in (dam_name, failed_structure, scenario, loading_scenario) if part]
    return {
        "dam_name": dam_name or "Unnamed dam",
        "failed_structure": failed_structure or None,
        "scenario": scenario or None,
        "loading_scenario": loading_scenario or None,
        "display_name": " · ".join(parts) or "Unnamed dam scenario",
        "hazard_classification": str(attributes.get("HazardCl") or "").strip() or None,
        "published_date": _published_date(attributes.get("PubDate")),
        "national_inventory_id": str(attributes.get("NID") or "").strip() or None,
        "state_id": str(attributes.get("StateID") or "").strip() or None,
    }


def result_from_evidence(evidence: GeospatialEvidence) -> Dict[str, Any]:
    checked = evidence.evidence_status == "checked"
    scenarios = [
        _scenario(feature.get("attributes") or {})
        for feature in evidence.matched_features
    ]
    limitations = list(dict.fromkeys([*evidence.limitations, *BASE_LIMITATIONS]))
    return {
        "data_status": "checked" if checked else evidence.evidence_status,
        "inside_inundation_boundary": bool(scenarios) if checked else None,
        "matched_dam_scenarios": scenarios,
        "matched_dam_scenario_names": [item["display_name"] for item in scenarios],
        "source_agency": SOURCE_AGENCY,
        "source_url": SOURCE_URL,
        "source_landing_url": SOURCE_LANDING_URL,
        "limitations": limitations,
        "checked_at": evidence.checked_at.isoformat(),
        "effective_date": evidence.effective_date,
        "public_claim_status": evidence.public_claim_status,
    }


def unavailable_result(message: str) -> Dict[str, Any]:
    return {
        "data_status": "data_unavailable",
        "inside_inundation_boundary": None,
        "matched_dam_scenarios": [],
        "matched_dam_scenario_names": [],
        "source_agency": SOURCE_AGENCY,
        "source_url": SOURCE_URL,
        "source_landing_url": SOURCE_LANDING_URL,
        "limitations": list(dict.fromkeys([message, *BASE_LIMITATIONS])),
        "checked_at": None,
        "effective_date": "2025-10-01",
        "public_claim_status": "official_unavailable",
    }


def check_dam_inundation(
    lat: float,
    lon: float,
    *,
    project_root: Optional[Path] = None,
    service: Optional[GeospatialEvidenceService] = None,
) -> Dict[str, Any]:
    checker = service or GeospatialEvidenceService(
        project_root=project_root or Path(__file__).resolve().parent
    )
    try:
        evidence = checker.check_point(DATASET_ID, lat, lon)
    except (DatasetRegistryError, TypeError, ValueError):
        return unavailable_result(
            "The official DWR/DSOD dam inundation service was unavailable, so the address was not checked."
        )
    return result_from_evidence(evidence)
