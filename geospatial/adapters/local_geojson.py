import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from shapely.geometry import Point, shape

from geospatial.adapters.base import GeospatialAdapter
from geospatial.models import (
    DatasetProvenance,
    DatasetValidation,
    GeoPoint,
    GeospatialEvidence,
)


PROVISIONAL_LIMIT = (
    "This official-source snapshot is provisional until a named human reviewer "
    "compares it with the agency's official viewer and approves this dataset version."
)


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
            raw_bytes = path.read_bytes()
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

        checksum_matches = hashlib.sha256(raw_bytes).hexdigest() == dataset.sha256
        if not checksum_matches:
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=False,
                errors=["Local dataset checksum does not match registered provenance."],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer failed provenance validation and was not checked.",
            )

        try:
            payload = json.loads(raw_bytes)
        except (UnicodeError, json.JSONDecodeError):
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=True,
                errors=["Local dataset is not valid JSON."],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer is invalid and was not checked.",
            )

        features = payload.get("features") if isinstance(payload, dict) else None
        if payload.get("type") != "FeatureCollection" or not isinstance(features, list) or not features:
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=True,
                errors=["Local dataset is not a non-empty GeoJSON FeatureCollection."],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer is empty or invalid and was not checked.",
            )

        record_count_matches = len(features) == dataset.record_count
        source_crs = self._crs_name(payload)
        crs_matches = bool(source_crs and source_crs == dataset.source_crs)
        if not record_count_matches or not crs_matches:
            errors = []
            if not record_count_matches:
                errors.append("Feature count does not match registered provenance.")
            if not crs_matches:
                errors.append("GeoJSON CRS does not match registered provenance.")
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=True,
                record_count_matches=record_count_matches,
                crs_matches=crs_matches,
                errors=errors,
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer failed schema or provenance validation and was not checked.",
            )

        address_point = Point(point.lon, point.lat)
        matched_features = []
        valid_geometry_count = 0
        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue
            try:
                geometry_shape = shape(geometry)
            except Exception:
                continue
            if (
                geometry_shape.is_empty
                or not geometry_shape.is_valid
                or geometry_shape.geom_type not in {"Polygon", "MultiPolygon"}
            ):
                continue
            valid_geometry_count += 1
            if geometry_shape.contains(address_point) or geometry_shape.touches(address_point):
                matched_features.append(self._feature_payload(dataset, feature))

        if valid_geometry_count == 0:
            validation = self._validation(
                dataset,
                status="invalid",
                checked_at=checked_at,
                checksum_matches=True,
                record_count_matches=True,
                crs_matches=True,
                errors=["No valid polygon geometry was available."],
            )
            return self._unavailable(
                dataset,
                checked_at,
                validation,
                "The local layer contained no valid polygon geometry and was not checked.",
            )

        warnings = []
        if dataset.status == "provisional":
            warnings.append("Dataset version requires human verification.")
        if valid_geometry_count < len(features):
            warnings.append(
                f"{len(features) - valid_geometry_count} records did not contain valid polygon geometry."
            )
        validation = self._validation(
            dataset,
            status="valid",
            checked_at=checked_at,
            checksum_matches=True,
            record_count_matches=True,
            crs_matches=True,
            valid_geometry_count=valid_geometry_count,
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
