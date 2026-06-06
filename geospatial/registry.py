import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from geospatial.models import DatasetProvenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "geospatial" / "datasets.json"


class DatasetRegistryError(RuntimeError):
    pass


class DatasetRegistry:
    def __init__(self, registry_path: Path = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self._datasets = self._load()

    def _load(self) -> List[DatasetProvenance]:
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DatasetRegistryError("Geospatial dataset registry is unavailable.") from exc
        if not isinstance(payload, list):
            raise DatasetRegistryError("Geospatial dataset registry must contain a list.")

        datasets = []
        seen = set()
        for index, item in enumerate(payload):
            try:
                dataset = DatasetProvenance(**item)
            except ValidationError as exc:
                raise DatasetRegistryError(
                    f"Geospatial dataset registry item {index} is invalid."
                ) from exc
            if dataset.dataset_id in seen:
                raise DatasetRegistryError(
                    f"Duplicate geospatial dataset_id: {dataset.dataset_id}"
                )
            seen.add(dataset.dataset_id)
            datasets.append(dataset)
        return datasets

    def all(self) -> List[DatasetProvenance]:
        return list(self._datasets)

    def get(self, dataset_id: str) -> Optional[DatasetProvenance]:
        return next(
            (item for item in self._datasets if item.dataset_id == dataset_id),
            None,
        )


@lru_cache(maxsize=1)
def get_default_registry() -> DatasetRegistry:
    return DatasetRegistry()
