"""Strict schemas for the offline LHMP extraction and review boundary."""

from datetime import datetime
from typing import Annotated, List, Literal, Optional

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CitySlug = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
DisplayLocation = Literal[
    "risk_summary",
    "city_page",
    "city_hazard_page",
    "hazard_page",
    "map_explanation",
]
CandidateStatus = Literal["candidate"]
ReviewedStatus = Literal["reviewed"]
CanonicalHazard = Literal[
    "earthquake",
    "flood",
    "wildfire",
    "landslide",
    "tsunami",
    "dam_failure",
    "drought",
    "extreme_heat",
    "poor_air_quality",
    "utility_disruption",
    "severe_weather",
    "sea_level_rise",
]


class PlanRegistryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: CitySlug
    city_display_name: NonEmptyText
    plan_title: Optional[NonEmptyText] = None
    plan_year: Optional[Annotated[int, Field(ge=1900, le=2100)]] = None
    local_pdf_path: NonEmptyText
    source_url: Optional[AnyHttpUrl] = None
    notes: str = ""
    enabled: bool = True
    markdown_path: Optional[NonEmptyText] = None
    max_render_pages: Annotated[int, Field(ge=0, le=100)] = 12
    max_evidence_candidates: Annotated[int, Field(ge=1, le=5000)] = 300


class CandidatePlanManifest(BaseModel):
    schema_version: Literal[1] = 1
    city: CitySlug
    source_document: NonEmptyText
    source_pdf_path: NonEmptyText
    markdown_path: str = ""
    source_url: str = ""
    source_pdf_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    source_pdf_size_bytes: Annotated[int, Field(gt=0)]
    pdf_page_count: Annotated[int, Field(gt=0)]
    extracted_at: datetime
    extractor_version: NonEmptyText
    pymupdf_version: NonEmptyText
    hazard_filter: Optional[CanonicalHazard] = None
    rendered_pdf_pages: List[Annotated[int, Field(gt=0)]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    review_status: CandidateStatus = "candidate"


class ReviewedPlanManifest(BaseModel):
    schema_version: Literal[1] = 1
    city: CitySlug
    jurisdiction_name: NonEmptyText
    plan_title: NonEmptyText
    source_document: NonEmptyText
    source_url: AnyHttpUrl
    source_pdf_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    edition: str = ""
    effective_date: str = ""
    document_status: NonEmptyText
    reviewer_id: NonEmptyText
    reviewed_at: datetime
    review_notes: NonEmptyText
    review_status: ReviewedStatus = "reviewed"


class SourceLinkedCandidate(BaseModel):
    id: NonEmptyText
    city: CitySlug
    source_document: NonEmptyText
    pdf_page: Annotated[int, Field(gt=0)]
    page_label: str = ""
    section_heading: str = ""
    suggested_hazard: Optional[CanonicalHazard] = None
    extraction_reason: NonEmptyText
    review_status: CandidateStatus = "candidate"


class EvidenceCandidate(SourceLinkedCandidate):
    original_text: str = ""
    extracted_snippet: str = ""

    @model_validator(mode="after")
    def require_source_text(self):
        if not self.original_text.strip() and not self.extracted_snippet.strip():
            raise ValueError("candidate evidence requires original_text or extracted_snippet")
        return self


class VisualCandidate(SourceLinkedCandidate):
    visual_type: Literal["map", "figure", "chart", "diagram", "other"]
    caption: str = ""
    page_image_path: NonEmptyText
    address_specific: Literal[False] = False


class TableCandidate(SourceLinkedCandidate):
    caption: str = ""
    extracted_text: NonEmptyText
    page_image_path: str = ""


class DataSourceCandidate(SourceLinkedCandidate):
    source_text: NonEmptyText
    suggested_name: str = ""
    suggested_agency: str = ""
    suggested_url: str = ""


class ReviewedSourceLinkedRecord(BaseModel):
    id: NonEmptyText
    city: CitySlug
    hazard: CanonicalHazard
    source_document: NonEmptyText
    pdf_page: Annotated[int, Field(gt=0)]
    page_label: str = ""
    section_heading: str = ""
    reviewer_id: NonEmptyText
    reviewed_at: datetime
    review_notes: NonEmptyText
    review_status: ReviewedStatus = "reviewed"


class ReviewedFact(ReviewedSourceLinkedRecord):
    user_facing_text: NonEmptyText
    plain_english_summary: NonEmptyText
    source_snippet: str = ""
    evidence_note: str = ""
    display_locations: List[DisplayLocation] = Field(min_length=1)
    geographic_scope: Literal["citywide", "area_context", "general"]
    address_specific: Literal[False] = False

    @model_validator(mode="after")
    def require_evidence_note(self):
        if not self.source_snippet.strip() and not self.evidence_note.strip():
            raise ValueError("reviewed facts require a source_snippet or evidence_note")
        return self

    @field_validator("display_locations")
    @classmethod
    def dedupe_display_locations(cls, values):
        return list(dict.fromkeys(values))


class ReviewedVisual(ReviewedSourceLinkedRecord):
    title: NonEmptyText
    caption: NonEmptyText
    visual_type: Literal["map", "figure", "chart", "diagram", "table", "other"]
    asset_path: NonEmptyText
    display_locations: List[DisplayLocation] = Field(min_length=1)
    address_specific: bool = False
    gis_dataset_id: str = ""
    official_gis_url: Optional[AnyHttpUrl] = None
    gis_linkage_reviewed: bool = False

    @model_validator(mode="after")
    def validate_public_asset_and_address_claim(self):
        expected_prefix = f"lhmp/{self.city}/figures/"
        normalized_path = self.asset_path.lstrip("/")
        if normalized_path.startswith("static/"):
            normalized_path = normalized_path[len("static/"):]
        if not normalized_path.startswith(expected_prefix):
            raise ValueError("reviewed visual asset_path must be inside its city static figure directory")
        if self.address_specific and not (
            self.gis_dataset_id.strip()
            and self.official_gis_url
            and self.gis_linkage_reviewed
        ):
            raise ValueError(
                "address-specific visuals require a reviewed official GIS linkage"
            )
        return self

    @field_validator("display_locations")
    @classmethod
    def dedupe_display_locations(cls, values):
        return list(dict.fromkeys(values))


class ReviewedDataSource(ReviewedSourceLinkedRecord):
    name: NonEmptyText
    agency: NonEmptyText
    official_url: AnyHttpUrl
    supported_claim: NonEmptyText
    dataset_id: str = ""
