from datetime import date, datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DatasetStatus = Literal["verified", "provisional", "invalid", "retired", "data_unavailable"]
EvidenceStatus = Literal["checked", "not_checked", "data_unavailable", "not_covered"]
PublicClaimStatus = Literal[
    "official_verified",
    "official_provisional",
    "official_unavailable",
    "not_evaluated",
    "retired",
]
ClaimType = Literal[
    "regulatory_zone",
    "hazard_zone",
    "proximity",
    "scenario",
    "live_alert",
    "observation",
    "guidance",
]
Precision = Literal["address_point", "bounded_area", "city", "county", "general"]


class GeoPoint(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class DatasetProvenance(BaseModel):
    dataset_id: NonEmptyText
    dataset_version: NonEmptyText
    hazard_type: NonEmptyText
    agency: NonEmptyText
    authoritative_landing_url: NonEmptyText
    exact_service_or_download_url: str = ""
    dataset_name: NonEmptyText
    claim_type: ClaimType
    source_type: Literal["local_snapshot", "remote_service", "live_feed"]
    license_terms_notes: NonEmptyText
    coverage_area: NonEmptyText
    coverage_bbox: Optional[Dict[str, float]] = None
    intended_claim: NonEmptyText
    prohibited_claims: List[NonEmptyText] = Field(min_length=1)
    retrieved_at: Optional[date] = None
    effective_date: Optional[str] = None
    local_path: str = ""
    sha256: str = ""
    record_count: Optional[int] = Field(default=None, ge=0)
    source_crs: str = ""
    converted_crs: str = ""
    status: DatasetStatus = "provisional"
    human_reviewer: str = ""
    human_reviewed_at: Optional[date] = None
    official_viewer_url: str = ""
    notes: str = ""

    @field_validator("sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        checksum = value.strip().lower()
        if checksum and (
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal checksum")
        return checksum

    @field_validator("coverage_bbox")
    @classmethod
    def validate_coverage_bbox(
        cls,
        value: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        if value is None:
            return None
        required = {"min_lat", "max_lat", "min_lon", "max_lon"}
        if set(value) != required:
            raise ValueError("coverage_bbox requires complete latitude and longitude bounds")
        if not (-90 <= value["min_lat"] < value["max_lat"] <= 90):
            raise ValueError("coverage_bbox latitude bounds are invalid")
        if not (-180 <= value["min_lon"] < value["max_lon"] <= 180):
            raise ValueError("coverage_bbox longitude bounds are invalid")
        return value

    @model_validator(mode="after")
    def enforce_human_verification(self):
        if self.source_type == "local_snapshot":
            if not self.local_path:
                raise ValueError("local snapshots require local_path")
            if not self.sha256:
                raise ValueError("local snapshots require sha256")
            if self.record_count is None:
                raise ValueError("local snapshots require record_count")
        if self.status == "verified" and (
            not self.human_reviewer or self.human_reviewed_at is None
        ):
            raise ValueError("verified datasets require a named human reviewer and review date")
        return self


class DatasetValidation(BaseModel):
    dataset_id: NonEmptyText
    dataset_version: NonEmptyText
    status: Literal["valid", "invalid", "data_unavailable"]
    checked_at: datetime
    checksum_matches: Optional[bool] = None
    record_count_matches: Optional[bool] = None
    crs_matches: Optional[bool] = None
    valid_geometry_count: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class LayerCheckResult(BaseModel):
    matched: bool
    matched_features: List[Dict[str, Any]] = Field(default_factory=list)
    valid_feature_count: int = Field(default=0, ge=0)


class GeospatialEvidence(BaseModel):
    dataset_id: NonEmptyText
    dataset_version: NonEmptyText
    hazard_type: NonEmptyText
    evidence_status: EvidenceStatus
    matched: Optional[bool] = None
    checked_at: datetime
    effective_date: Optional[str] = None
    claim_type: ClaimType
    precision: Precision
    public_claim_status: PublicClaimStatus
    source_agency: NonEmptyText
    source_url: NonEmptyText
    matched_features: List[Dict[str, Any]] = Field(default_factory=list)
    limitations: List[NonEmptyText] = Field(min_length=1)
    provenance: DatasetProvenance
    validation: DatasetValidation

    @model_validator(mode="after")
    def validate_evidence_state(self):
        if self.evidence_status == "checked" and self.matched is None:
            raise ValueError("checked evidence requires matched true or false")
        if self.evidence_status != "checked" and self.matched is not None:
            raise ValueError("unchecked evidence cannot claim a match or non-match")
        return self
