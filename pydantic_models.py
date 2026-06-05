from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Scope = Literal["address_level", "jurisdiction_level", "zip_estimate", "county_fallback"]
Basis = Literal["gis_overlay", "official_registry", "zip_csv_heuristic", "county_guidance"]
LocationPrecision = Literal["address_point", "city", "neighborhood", "census_tract", "zip", "county", "unknown"]
DataStatus = Literal["checked", "not_checked", "not_in_layer", "fallback_used", "needs_review"]
MatchType = Literal["inside", "near", "intersects", "jurisdiction_match", "zip_match", "fallback", "none"]
ExposureLevel = Literal["low", "medium", "high", "unknown"]
ConfidenceLabel = Literal["source_backed", "mixed_support", "needs_review"]
ReviewStatus = Literal["reviewed", "draft_reviewed", "draft", "needs_source_review", "insufficient_source_support"]


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


class ResidentGuidanceItem(BaseModel):
    chunk_id: str
    source_id: str
    jurisdiction: str
    hazard_type: str
    resident_phase: Literal["hazard_priority", "local_context", "before", "during", "after", "recovery", "limitations"]
    guidance_type: str = ""
    plain_language: str
    recommended_action: str = ""
    recovery_question: str = ""
    source_detail: str = ""
    source_url: str = ""
    review_status: ReviewStatus = "draft"


class SpecializedGuidance(BaseModel):
    location_specific_context: List[str] = Field(default_factory=list)
    city_context: List[str] = Field(default_factory=list)
    household_factors: List[str] = Field(default_factory=list)
    access_functional_needs: List[str] = Field(default_factory=list)
    recovery_needs: List[str] = Field(default_factory=list)
    resident_guidance: Dict[str, List[ResidentGuidanceItem]] = Field(default_factory=dict)
    guidance_source_status: str = "county_fallback"
    source_ids: List[str] = Field(default_factory=list)


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
    local_plan_match: Optional[Dict[str, Any]] = None
    specialized_guidance: SpecializedGuidance = Field(default_factory=SpecializedGuidance)
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
