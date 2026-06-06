from abc import ABC, abstractmethod

from geospatial.models import DatasetProvenance, GeoPoint, GeospatialEvidence


class GeospatialAdapter(ABC):
    @abstractmethod
    def check_point(
        self,
        dataset: DatasetProvenance,
        point: GeoPoint,
    ) -> GeospatialEvidence:
        """Check one address point against one registered dataset."""
