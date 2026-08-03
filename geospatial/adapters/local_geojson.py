import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from geospatial.adapters.base import GeospatialAdapter
from geospatial.models import (
    DatasetProvenance,
    DatasetValidation,
    GeoPoint,
    GeospatialEvidence,
)
from geospatial.streaming_geojson import read_geojson_header, scan_polygon_features_at_point


PROVISIONAL_LIMIT = (
    "This official-source snapshot is provisional until a named human reviewer "
    "compares it with the agency's official viewer and approves this dataset version."
)
CAL_FIRE_SEVERITY_CLASSES = {"moderate", "high", "very high"}


@lru_cache(maxsize=8)
def _load_validated_layer(
    path_string: str,
    modified_ns: int,
    file_size: int,
    expected_sha256: str,
    expected_record_count: int,
    expected_crs: str,
) -> Dict[str, Any]:
    """Validate immutable file identity without retaining the full layer."""
    del modified_ns, file_size
    path = Path(path_string)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return {"error": "missing"}

    if digest.hexdigest() != expected_sha256:
        return {"error": "checksum"}

    header = read_geojson_header(path)
    if header.get("error"):
        return {"error": header["error"]}
    if header.get("type") != "FeatureCollection":
        return {"error": "schema"}
    source_crs = str(
        ((header.get("crs") or {}).get("properties") or {}).get("name")
        or ""
    )
    if not source_crs or source_crs != expected_crs:
        return {"error": "crs"}
    return {
        "expected_record_count": expected_record_count,
        "source_crs": source_crs,
    }


class LocalGeoJSONAdapter(GeospatialAdapter):
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def _path(self, dataset: DatasetProvenance) -> Path:
        return self.project_root / dataset.local_path

    def _validation(
        self,
        dataset: DatasetProvenance,
        *,
        status: str,
        checked_at: datetime,
        checksum_matches: Optional[bool] = None,
        record_count_matches: Optional[bool] = None,
        crs_matches: Optional[bool] = None,
        valid_geometry_count: int = 0,
        errors=None,
        warnings=None,
    ) -> DatasetValidation:
        return DatasetValidation(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            status=status,
            checked_at=checked_at,
            checksum_matches=checksum_matches,
            record_count_matches=record_count_matches,
            crs_matches=crs_matches,
            valid_geometry_count=valid_geometry_count,
            errors=errors or [],
            warnings=warnings or [],
        )

    def _unavailable(
        self,
        dataset: DatasetProvenance,
        checked_at: datetime,
        validation: DatasetValidation,
        limitation: str,
    ) -> GeospatialEvidence:
        public_status = "retired" if dataset.status == "retired" else "official_unavailable"
        return GeospatialEvidence(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            hazard_type=dataset.hazard_type,
            evidence_status="data_unavailable",
            matched=None,
            checked_at=checked_at,
            effective_date=dataset.effective_date,
            claim_type=self._claim_type(dataset),
            precision="address_point",
            public_claim_status=public_status,
            source_agency=dataset.agency,
            source_url=dataset.authoritative_landing_url,
            matched_features=[],
            limitations=[limitation, *dataset.prohibited_claims],
            provenance=dataset,
            validation=validation,
        )

    @staticmethod
    def _claim_type(dataset: DatasetProvenance) -> str:
        return dataset.claim_type

    @staticmethod
    def _crs_name(payload: Dict[str, Any]) -> str:
        return str(
            ((payload.get("crs") or {}).get("properties") or {}).get("name")
            or ""
        )

    @staticmethod
    def _feature_payload(dataset: DatasetProvenance, feature: Dict[str, Any]) -> Dict[str, Any]:
        properties = feature.get("properties") or {}
        if dataset.dataset_id == "fema_nfhl_local":
            return {
                "layer_id": properties.get("FLD_AR_ID")
                or properties.get("SOURCE_CIT")
                or "fema_flood_layer",
                "name": f"FEMA flood zone {properties.get('FLD_ZONE', 'unknown')}",
                "zone": properties.get("FLD_ZONE", "Unknown"),
                "source_citation": properties.get("SOURCE_CIT", ""),
                "sfha": properties.get("SFHA_TF", ""),
            }
        if dataset.dataset_id == "calfire_fhsz_local":
            hazard_class = (
                properties.get("HAZ_CLASS")
                or properties.get("VH_REC")
                or "Unknown"
            )
            return {
                "layer_id": f"calfire_fhsz_{properties.get('HAZ_CODE', 'unknown')}",
                "name": f"CAL FIRE Fire Hazard Severity Zone: {hazard_class}",
                "hazard_class": hazard_class,
                "state_responsibility_area": properties.get("SRA", ""),
                "incorporated": properties.get("INCORP", ""),
            }
        return {"name": dataset.dataset_name}

    @staticmethod
    def _is_publishable_match(
        dataset: DatasetProvenance,
        feature: Dict[str, Any],
    ) -> bool:
        if dataset.dataset_id != "calfire_fhsz_local":
            return True
        properties = feature.get("properties") or {}
        hazard_class = str(
            properties.get("HAZ_CLASS")
            or properties.get("VH_REC")
            or ""
        ).strip().casefold()
        return hazard_class in CAL_FIRE_SEVERITY_CLASSES

    def check_point(
        self,
        dataset: DatasetProvenance,
        point: GeoPoint,
    ) -> GeospatialEvidence:
        checked_at = datetime.now(timezone.utc)
        if dataset.status in {"invalid", "retired", "data_unavailable"}:
            validation = self._validation(
                dataset,
                status="data_unavailable",
                checked_at=checked_at,
                errors=[f"Dataset registry status is {dataset.status}."],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The registered dataset is not available for an address check.",
            )

        path = self._path(dataset)
        try:
            stat = path.stat()
        except OSError:
            validation = self._validation(
                dataset,
                status="data_unavailable",
                checked_at=checked_at,
                errors=["Local dataset file is missing or unreadable."],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "Official layer unavailable — not checked.",
            )

        layer = _load_validated_layer(
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            dataset.sha256,
            dataset.record_count,
            dataset.source_crs,
        )
        error = layer.get("error")
        if error:
            error_messages = {
                "checksum": "Local dataset checksum does not match registered provenance.",
                "json": "Local dataset is not valid JSON.",
                "schema": "Local dataset is not a non-empty GeoJSON FeatureCollection.",
                "record_count": "Feature count does not match registered provenance.",
                "crs": "GeoJSON CRS does not match registered provenance.",
                "geometry": "No valid polygon geometry was available.",
            }
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=False if error == "checksum" else True,
                record_count_matches=False if error == "record_count" else None,
                crs_matches=False if error == "crs" else None,
                errors=[error_messages.get(error, "Local dataset validation failed.")],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer failed schema or provenance validation and was not checked.",
            )

        scan = scan_polygon_features_at_point(
            path,
            latitude=point.lat,
            longitude=point.lon,
        )
        scan_error = scan.get("error")
        if not scan_error and scan.get("record_count") != dataset.record_count:
            scan_error = "record_count"
        if scan_error:
            error_messages = {
                "json": "Local dataset is not valid JSON.",
                "schema": "Local dataset is not a non-empty GeoJSON FeatureCollection.",
                "record_count": "Feature count does not match registered provenance.",
                "geometry": "No valid polygon geometry was available.",
            }
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=True,
                record_count_matches=False if scan_error == "record_count" else None,
                crs_matches=True,
                errors=[error_messages.get(scan_error, "Local dataset validation failed.")],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer failed schema or provenance validation and was not checked.",
            )

        matched_features = []
        for properties in scan["matched_properties"]:
            feature = {"properties": properties}
            if self._is_publishable_match(dataset, feature):
                matched_features.append(self._feature_payload(dataset, feature))

        warnings = []
        if dataset.status == "provisional":
            warnings.append("Dataset version requires human verification.")
        if scan["invalid_geometry_count"]:
            warnings.append(
                f"{scan['invalid_geometry_count']} records did not contain valid polygon geometry."
            )
        validation = self._validation(
            dataset,
            status="valid",
            checked_at=checked_at,
            checksum_matches=True,
            record_count_matches=True,
            crs_matches=True,
            valid_geometry_count=scan["valid_geometry_count"],
            warnings=warnings,
        )
        public_status = (
            "official_verified"
            if dataset.status == "verified"
            else "official_provisional"
        )
        limitations = list(dataset.prohibited_claims)
        if dataset.status == "provisional":
            limitations.insert(0, PROVISIONAL_LIMIT)
        return GeospatialEvidence(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            hazard_type=dataset.hazard_type,
            evidence_status="checked",
            matched=bool(matched_features),
            checked_at=checked_at,
            effective_date=dataset.effective_date,
            claim_type=self._claim_type(dataset),
            precision="address_point",
            public_claim_status=public_status,
            source_agency=dataset.agency,
            source_url=dataset.authoritative_landing_url,
            matched_features=matched_features,
            limitations=limitations,
            provenance=dataset,
            validation=validation,
        )
