"""Read-only point checks against registered public ArcGIS feature layers."""

from datetime import datetime, timezone
from functools import lru_cache

import requests

from geospatial.adapters.base import GeospatialAdapter
from geospatial.models import (
    DatasetProvenance,
    DatasetValidation,
    GeoPoint,
    GeospatialEvidence,
)


PROVISIONAL_LIMIT = (
    "This official remote dataset is provisional until a named human reviewer "
    "compares known locations with the agency's official viewer."
)


@lru_cache(maxsize=32)
def _remote_record_count(service_url: str, timeout_seconds: int) -> int:
    response = requests.get(
        f"{service_url.rstrip('/')}/query",
        params={"f": "json", "where": "1=1", "returnCountOnly": "true"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error") or not isinstance(payload.get("count"), int):
        raise ValueError("ArcGIS record-count response is invalid.")
    return payload["count"]


class ArcGISFeatureServiceAdapter(GeospatialAdapter):
    timeout_seconds = 8

    @staticmethod
    def _unavailable(
        dataset: DatasetProvenance,
        checked_at: datetime,
        message: str,
    ) -> GeospatialEvidence:
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
            evidence_status="data_unavailable",
            matched=None,
            checked_at=checked_at,
            effective_date=dataset.effective_date,
            claim_type=dataset.claim_type,
            precision="address_point",
            public_claim_status="official_unavailable",
            source_agency=dataset.agency,
            source_url=dataset.authoritative_landing_url,
            matched_features=[],
            limitations=[message, *dataset.prohibited_claims],
            provenance=dataset,
            validation=validation,
        )

    @staticmethod
    def _feature_payload(dataset: DatasetProvenance, attributes) -> dict:
        return {
            "layer_id": dataset.dataset_id,
            "name": dataset.result_label or dataset.dataset_name,
            "dataset_name": dataset.dataset_name,
            "attributes": {
                key: value
                for key, value in (attributes or {}).items()
                if key.lower() not in {"shape", "shape__area", "shape__length"}
            },
        }

    def check_point(
        self,
        dataset: DatasetProvenance,
        point: GeoPoint,
    ) -> GeospatialEvidence:
        checked_at = datetime.now(timezone.utc)
        if dataset.status in {"invalid", "retired", "data_unavailable"}:
            return self._unavailable(
                dataset,
                checked_at,
                "The registered CGS dataset is not available for an address check.",
            )
        if not dataset.exact_service_or_download_url:
            return self._unavailable(
                dataset,
                checked_at,
                "The registered CGS service URL is missing, so the layer was not checked.",
            )

        params = {
            "f": "json",
            "where": "1=1",
            "geometry": f"{point.lon},{point.lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "false",
        }
        try:
            record_count = _remote_record_count(
                dataset.exact_service_or_download_url,
                self.timeout_seconds,
            )
            if record_count <= 0:
                return self._unavailable(
                    dataset,
                    checked_at,
                    "The official CGS layer is empty, so the address was not checked.",
                )
            response = requests.get(
                f"{dataset.exact_service_or_download_url.rstrip('/')}/query",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            return self._unavailable(
                dataset,
                checked_at,
                "The official CGS service was unavailable or unreadable, so the layer was not checked.",
            )

        if not isinstance(payload, dict) or payload.get("error") or not isinstance(payload.get("features"), list):
            return self._unavailable(
                dataset,
                checked_at,
                "The official CGS service returned an invalid response, so the layer was not checked.",
            )

        matching_features = []
        accepted_values = {value.casefold() for value in dataset.match_values}
        for feature in payload["features"]:
            attributes = feature.get("attributes") or {}
            if dataset.match_field and accepted_values:
                value = str(attributes.get(dataset.match_field, "")).casefold()
                if value not in accepted_values:
                    continue
            matching_features.append(self._feature_payload(dataset, attributes))

        warnings = []
        limitations = list(dataset.prohibited_claims)
        if dataset.status == "provisional":
            warnings.append("Dataset version requires human verification.")
            limitations.insert(0, PROVISIONAL_LIMIT)
        validation = DatasetValidation(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            status="valid",
            checked_at=checked_at,
            valid_geometry_count=len(payload["features"]),
            warnings=warnings,
        )
        return GeospatialEvidence(
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            hazard_type=dataset.hazard_type,
            evidence_status="checked",
            matched=bool(matching_features),
            checked_at=checked_at,
            effective_date=dataset.effective_date,
            claim_type=dataset.claim_type,
            precision="address_point",
            public_claim_status=(
                "official_verified"
                if dataset.status == "verified"
                else "official_provisional"
            ),
            source_agency=dataset.agency,
            source_url=dataset.authoritative_landing_url,
            matched_features=matching_features,
            limitations=limitations,
            provenance=dataset,
            validation=validation,
        )
