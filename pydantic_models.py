from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Scope = Literal["address_level", "jurisdiction_level", "zip_estimate", "county_fallback"]
Basis = Literal["gis_overlay", "official_registry", "zip_csv_heuristic", "county_guidance"]
LocationPrecision = Literal["address_point", "city", "neighborhood", "census_tract", "zip", "county", "unknown"]
DataStatus = Literal["checked", "not_checked", "not_in_layer", "fallback_used", "needs_review"]
MatchType = Literal["inside", "near", "intersects", "jurisdiction_match", "zip_match", "fallback", "none"]
ExposureLevel = Literal["low", "medium", "high", "unknown"]
ConfidenceLabel = Literal["source_backed", "mixed_support", "needs_review"]
ReviewStatus = Literal["reviewed", "draft", "insufficient_source_support"]


class LocationResult(BaseModel):
    input_address: str = ""
    formatted_address: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    city: str = ""
    county: str = ""
    zip_code: str = ""
    neighborhood: str = ""
    census_tract: str = ""
    geocoder: str = "nominatim"
    geocode_confidence: ConfidenceLabel = "needs_review"
    limitations: List[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    source_id: str
    name: str
    agency: str
    source_type: str
    hazards: List[str] = Field(default_factory=list)
    geographic_scope: str = ""
    claim_type: str = ""
    use_in_app: str = ""
    confidence: ConfidenceLabel = "needs_review"
    review_status: ReviewStatus = "draft"
    url: str = ""
    notes: str = ""
    last_verified: str = ""


class RAGChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_name: str
    agency: str
    hazard_type: str
    jurisdiction: str
    geographic_scope: str
    claim_type: str
    page_number: Optional[int] = None
    text: str
    url: str = ""
    confidence: ConfidenceLabel = "needs_review"
    review_status: ReviewStatus = "draft"


class PreparednessAction(BaseModel):
    id: str
    label: str
    source_id: str = ""


class RecoveryQuestion(BaseModel):
    id: str
    question: str
    source_id: str = ""


class HazardResult(BaseModel):
    hazard_id: str
    hazard_type: str
    label: str
    scope: Scope
    basis: Basis
    location_precision: LocationPrecision
    data_status: DataStatus
    exposure_level: ExposureLevel = "unknown"
    is_in_hazard_zone: Optional[bool] = None
    match_type: MatchType = "none"
    matched_layers: List[Dict[str, Any]] = Field(default_factory=list)
    source_url: str = ""
    confidence: ConfidenceLabel = "needs_review"
    review_status: ReviewStatus = "draft"
    why_shown: str = ""
    limitations: List[str] = Field(default_factory=list)
    recommended_actions: List[PreparednessAction] = Field(default_factory=list)
    recovery_questions: List[RecoveryQuestion] = Field(default_factory=list)
    sources: List[SourceRecord] = Field(default_factory=list)
    legacy_score: Optional[float] = None
    legacy_priority_score: Optional[int] = None


class HazardExplanation(BaseModel):
    plain_english_explanation: str
    why_shown: str
    recommended_actions: List[PreparednessAction] = Field(default_factory=list)
    recovery_questions: List[RecoveryQuestion] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    citations: List[SourceRecord] = Field(default_factory=list)
    confidence: ConfidenceLabel = "needs_review"
    review_status: ReviewStatus = "draft"
