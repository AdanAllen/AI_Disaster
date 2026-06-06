from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from geospatial.adapters.local_geojson import LocalGeoJSONAdapter
from geospatial.models import (
    DatasetProvenance,
    DatasetValidation,
    GeoPoint,
    GeospatialEvidence,
)
from geospatial.registry import DatasetRegistry, DatasetRegistryError, get_default_registry


class GeospatialEvidenceService:
    def __init__(
        self,
        *,
        project_root: Path,
        registry: Optional[DatasetRegistry] = None,
    ):
        self.project_root = Path(project_root)
        self.registry = registry or get_default_registry()
        self.local_adapter = LocalGeoJSONAdapter(self.project_root)

    def check_point(self, dataset_id: str, lat: float, lon: float) -> GeospatialEvidence:
        dataset = self.registry.get(dataset_id)
        if dataset is None:
            raise DatasetRegistryError(f"Dataset is not registered: {dataset_id}")
        point = GeoPoint(lat=lat, lon=lon)
        if dataset.coverage_bbox and not self._inside_coverage(point, dataset.coverage_bbox):
            return self._not_covered(dataset)
        if dataset.source_type != "local_snapshot":
            return self._not_checked(
                dataset,
                "This registered remote dataset has not been enabled for automated checks.",
            )
        return self.local_adapter.check_point(dataset, point)

    @staticmethod
    def _inside_coverage(point: GeoPoint, bounds) -> bool:
        return (
            bounds["min_lat"] <= point.lat <= bounds["max_lat"]
            and bounds["min_lon"] <= point.lon <= bounds["max_lon"]
        )

    @staticmethod
    def _not_covered(dataset: DatasetProvenance) -> GeospatialEvidence:
        checked_at = datetime.now(timezone.utc)
        message = (
            "The address point is outside this dataset version's documented coverage; "
            "the location was not evaluated by this layer."
        )
        validation = DatasetValidation(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            status="valid",
            checked_at=checked_at,
            warnings=[message],
        )
        return GeospatialEvidence(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            hazard_type=dataset.hazard_type,
            evidence_status="not_covered",
            matched=None,
            checked_at=checked_at,
            effective_date=dataset.effective_date,
            claim_type=dataset.claim_type,
            precision="address_point",
            public_claim_status="not_evaluated",
            source_agency=dataset.agency,
            source_url=dataset.authoritative_landing_url,
            matched_features=[],
            limitations=[message, *dataset.prohibited_claims],
            provenance=dataset,
            validation=validation,
        )

    @staticmethod
    def _not_checked(
        dataset: DatasetProvenance,
        message: str,
    ) -> GeospatialEvidence:
        checked_at = datetime.now(timezone.utc)
        validation = DatasetValidation(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            status="data_unavailable",
            checked_at=checked_at,
            errors=[message],
        )
        return GeospatialEvidence(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            hazard_type=dataset.hazard_type,
            evidence_status="not_checked",
            matched=None,
            checked_at=checked_at,
            effective_date=dataset.effective_date,
            claim_type=dataset.claim_type,
            precision="address_point",
            public_claim_status="not_evaluated",
            source_agency=dataset.agency,
            source_url=dataset.authoritative_landing_url,
            matched_features=[],
            limitations=[message, *dataset.prohibited_claims],
            provenance=dataset,
            validation=validation,
        )
