from datetime import date
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import AnyHttpUrl, BaseModel, Field, StrictInt, StringConstraints, field_validator, model_validator


Scope = Literal["address_level", "jurisdiction_level", "zip_estimate", "county_fallback"]
Basis = Literal["gis_overlay", "official_registry", "zip_csv_heuristic", "county_guidance"]
LocationPrecision = Literal["address_point", "city", "neighborhood", "census_tract", "zip", "county", "unknown"]
DataStatus = Literal["checked", "not_checked", "data_unavailable", "not_in_layer", "fallback_used", "needs_review"]
MatchType = Literal[
    "inside",
    "near",
    "near_fault",
    "fault_proximity_context",
    "intersects",
    "jurisdiction_match",
    "zip_match",
    "fallback",
    "none",
]
ExposureLevel = Literal["low", "medium", "high", "unknown"]
ConfidenceLabel = Literal["source_backed", "mixed_support", "needs_review"]
ReviewStatus = Literal["reviewed", "draft_reviewed", "draft", "needs_source_review", "insufficient_source_support"]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PageReference = Union[
    Annotated[StrictInt, Field(gt=0)],
    Annotated[
        str,
        StringConstraints(
            strict=True,
            strip_whitespace=True,
            min_length=1,
            pattern=r"^\d+(?:\s*[-,;]\s*\d+)*$",
        ),
    ],
]


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


class ActionCitation(BaseModel):
    source_id: NonEmptyText
    source_name: NonEmptyText
    source_url: AnyHttpUrl
    source_document: str = ""
    source_page: Optional[PageReference] = None
    source_section: str = ""
    source_summary: NonEmptyText


class ActionRecord(BaseModel):
    action_id: NonEmptyText
    title: NonEmptyText
    instruction: NonEmptyText
    hazards: List[NonEmptyText] = Field(min_length=1)
    household_factors: List[str] = Field(default_factory=list)
    time_buckets: List[
        Literal["today", "this_week", "this_month", "before", "during", "after", "recovery"]
    ] = Field(min_length=1)
    citation: ActionCitation
    confidence: Literal["official_direct", "official_paraphrase", "expert_reviewed", "needs_source"]
    review_status: Literal["reviewed", "draft_reviewed", "draft", "needs_source", "retired"]
    authority_scope: Literal["national", "state", "county", "city"]
    guidance_scope: Literal["general", "hazard_specific"]
    trigger_type: Literal["general", "hazard_result", "location", "household", "live_event"]
    applicable_jurisdictions: List[str] = Field(default_factory=list)
    required_household_factors: List[str] = Field(default_factory=list)
    excluded_household_factors: List[str] = Field(default_factory=list)
    required_evidence: Dict[str, Any] = Field(default_factory=dict)
    priority_category: Literal[
        "life_safety",
        "official_alerts",
        "evacuation",
        "medical",
        "communication",
        "supplies",
        "property",
        "recovery",
    ]
    last_source_verified: date
    notes: str = ""

    @field_validator(
        "hazards",
        "household_factors",
        "applicable_jurisdictions",
        "required_household_factors",
        "excluded_household_factors",
    )
    @classmethod
    def normalize_action_lists(cls, values: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for value in values:
            text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def validate_publishability(self):
        if self.trigger_type == "household" and not self.required_household_factors:
            raise ValueError("household actions require at least one required household factor")
        if self.review_status in {"reviewed", "draft_reviewed"} and self.confidence == "needs_source":
            raise ValueError("reviewed actions cannot have needs_source confidence")
        return self

    @property
    def displayable(self) -> bool:
        return (
            self.review_status in {"reviewed", "draft_reviewed"}
            and self.confidence != "needs_source"
            and bool(self.citation.source_url)
            and bool(self.citation.source_summary)
        )


class PreparednessAction(ActionRecord):
    why_shown: NonEmptyText
    matched_hazards: List[str] = Field(default_factory=list)
    matched_household_factors: List[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return self.action_id

    @property
    def label(self) -> str:
        return self.instruction


class RecoveryQuestion(BaseModel):
    id: str
    question: str
    source_id: str = ""
    source_name: str = ""
    source_url: str = ""
    source_summary: str = ""


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


class LHMPLocationFact(BaseModel):
    id: NonEmptyText
    jurisdiction: NonEmptyText
    applies_to_jurisdictions: List[str] = Field(default_factory=list)
    hazard: NonEmptyText
    evidence_tier: Literal["area_based", "citywide", "general"]
    geography_type: NonEmptyText
    named_areas: List[NonEmptyText] = Field(min_length=1)
    location_aliases: List[str] = Field(default_factory=list)
    coordinate_rule: Dict[str, float] = Field(default_factory=dict)
    location_cue: NonEmptyText
    resident_meaning: NonEmptyText
    before_actions: List[NonEmptyText] = Field(min_length=1)
    during_actions: List[NonEmptyText] = Field(min_length=1)
    after_actions: List[NonEmptyText] = Field(min_length=1)
    recovery_steps: List[NonEmptyText] = Field(min_length=1)
    resident_impacts: List[NonEmptyText] = Field(min_length=1)
    household_factors: List[str] = Field(default_factory=list)
    infrastructure_dependencies: List[NonEmptyText] = Field(min_length=1)
    requires_gis_confirmation: bool = True
    precision_limitations: List[NonEmptyText] = Field(min_length=1)
    source_document: NonEmptyText
    source_page: PageReference
    source_excerpt_summary: NonEmptyText
    source_name: NonEmptyText
    source_url: NonEmptyText
    review_status: ReviewStatus

    @field_validator("applies_to_jurisdictions", "location_aliases", "household_factors")
    @classmethod
    def clean_optional_lists(cls, values: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            cleaned.append(text)
        return cleaned

    @field_validator("coordinate_rule")
    @classmethod
    def validate_coordinate_rule(cls, rule: Dict[str, float]) -> Dict[str, float]:
        if not rule:
            return {}
        required = {"min_lat", "max_lat", "min_lon", "max_lon"}
        if set(rule) != required:
            raise ValueError("coordinate_rule must be empty or contain a complete reviewed bounding box")
        if not (-90 <= rule["min_lat"] < rule["max_lat"] <= 90):
            raise ValueError("coordinate_rule latitude bounds are invalid")
        if not (-180 <= rule["min_lon"] < rule["max_lon"] <= 180):
            raise ValueError("coordinate_rule longitude bounds are invalid")
        return rule

    @model_validator(mode="after")
    def validate_matchability(self):
        if self.evidence_tier == "area_based" and not self.location_aliases and not self.coordinate_rule:
            raise ValueError("area_based facts require reviewed aliases or a complete coordinate bound")
        return self


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
