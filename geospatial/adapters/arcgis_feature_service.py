"""Reserved adapter boundary for future reviewed CGS ArcGIS services.

No remote service is queried in Geospatial Evidence Engine v1. Exact service
URLs and schemas must be recorded and human-reviewed before implementation.
"""

from geospatial.adapters.base import GeospatialAdapter
from geospatial.models import DatasetProvenance, GeoPoint, GeospatialEvidence


class ArcGISFeatureServiceAdapter(GeospatialAdapter):
    def check_point(
        self,
        dataset: DatasetProvenance,
        point: GeoPoint,
    ) -> GeospatialEvidence:
        raise NotImplementedError(
            "Remote ArcGIS checks are not enabled until dataset provenance is reviewed."
        )
