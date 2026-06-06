from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from geospatial.adapters.local_geojson import LocalGeoJSONAdapter
from geospatial.adapters.arcgis_feature_service import ArcGISFeatureServiceAdapter
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
        self.arcgis_adapter = ArcGISFeatureServiceAdapter()

    def check_point(self, dataset_id: str, lat: float, lon: float) -> GeospatialEvidence:
        dataset = self.registry.get(dataset_id)
        if dataset is None:
            raise DatasetRegistryError(f"Dataset is not registered: {dataset_id}")
        point = GeoPoint(lat=lat, lon=lon)
        if dataset.coverage_bbox and not self._inside_coverage(point, dataset.coverage_bbox):
            return self._not_covered(dataset)
        if dataset.source_type == "local_snapshot":
            return self.local_adapter.check_point(dataset, point)
        if dataset.source_type == "remote_service":
            return self.arcgis_adapter.check_point(dataset, point)
        return self._not_checked(dataset, "This registered data source cannot perform point checks.")

    def map_geojson(
        self,
        dataset_id: str,
        *,
        lat: float,
        lon: float,
        radius_degrees: float = 0.12,
    ) -> dict:
        dataset = self.registry.get(dataset_id)
        if dataset is None:
            raise DatasetRegistryError(f"Dataset is not registered: {dataset_id}")
        if dataset.source_type != "remote_service":
            raise DatasetRegistryError("Map GeoJSON is only supported for remote services.")
        point = GeoPoint(lat=lat, lon=lon)
        if dataset.coverage_bbox and not self._inside_coverage(point, dataset.coverage_bbox):
            raise DatasetRegistryError("The point is outside the registered dataset coverage.")
        return self.arcgis_adapter.query_geojson(
            dataset,
            min_lat=point.lat - radius_degrees,
            max_lat=point.lat + radius_degrees,
            min_lon=point.lon - radius_degrees,
            max_lon=point.lon + radius_degrees,
        )

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
