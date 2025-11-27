"""Job Profile models for job postings and their enriched representations (Pydantic only)."""

from typing import Any

from pydantic import Field

from .base import (
    IdentifiedSchema,
    RSASBaseModel,
    TimestampSchema,
    VersionedSchema,
)
from .enums import EmploymentType, LocationType, SeniorityLevel


# =============================================================================
# Pydantic Schemas (for validation and API)
# =============================================================================


class Skill(RSASBaseModel):
    """A required or preferred skill."""

    name: str = Field(..., description="Skill name")
    category: str | None = Field(None, description="Skill category (e.g., 'programming_language')")
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Importance weight")
    is_hard_requirement: bool = Field(
        False, description="If true, absence creates a ceiling on score"
    )
    min_years: float | None = Field(None, ge=0, description="Minimum years of experience")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")


class ExperienceRequirement(RSASBaseModel):
    """Experience requirements for a job."""

    min_years_overall: float | None = Field(None, ge=0, description="Minimum years overall")
    min_years_relevant: float | None = Field(
        None, ge=0, description="Minimum years in relevant domain"
    )
    max_years: float | None = Field(None, ge=0, description="Maximum years (seniority ceiling)")
    seniority_level: SeniorityLevel | None = Field(None, description="Expected seniority level")


class EducationRequirement(RSASBaseModel):
    """Education requirements for a job."""

    min_degree: str | None = Field(None, description="Minimum degree level")
    preferred_fields: list[str] = Field(
        default_factory=list, description="Preferred fields of study"
    )
    required_certifications: list[str] = Field(
        default_factory=list, description="Required certifications"
    )


class LocationRequirement(RSASBaseModel):
    """Location requirements for a job."""

    location_type: LocationType = Field(LocationType.FLEXIBLE, description="Work arrangement")
    accepted_locations: list[str] = Field(
        default_factory=list, description="Accepted countries/regions"
    )
    timezone_preferences: list[str] = Field(
        default_factory=list, description="Preferred timezones"
    )


class JobMetadata(RSASBaseModel):
    """Additional job metadata."""

    department: str | None = Field(None, description="Department or team")
    role_family: str | None = Field(None, description="Role family (e.g., 'Engineering')")
    employment_type: EmploymentType | None = Field(None, description="Employment type")
    visa_sponsorship: bool = Field(False, description="Visa sponsorship available")
    security_clearance_required: bool = Field(False, description="Security clearance required")
    work_authorization_constraints: list[str] = Field(
        default_factory=list, description="Work authorization constraints"
    )


class EnrichedJobProfile(IdentifiedSchema, TimestampSchema, VersionedSchema):
    """Enriched job profile created by Job Understanding Agent."""

    job_id: str = Field(..., description="Job identifier (same as id)")
    title: str = Field(..., description="Normalized job title")
    raw_description: str = Field(..., description="Original job description")

    # Requirements
    must_have_skills: list[Skill] = Field(
        default_factory=list, description="Must-have skills and qualifications"
    )
    nice_to_have_skills: list[Skill] = Field(
        default_factory=list, description="Nice-to-have skills and qualifications"
    )

    # Experience and education
    experience_requirements: ExperienceRequirement | None = Field(
        None, description="Experience requirements"
    )
    education_requirements: EducationRequirement | None = Field(
        None, description="Education requirements"
    )

    # Domain and context
    domain_tags: dict[str, float] = Field(
        default_factory=dict, description="Domain tags with confidence scores"
    )
    soft_skills: list[str] = Field(default_factory=list, description="Required soft skills")

    # Location
    location_requirements: LocationRequirement | None = Field(
        None, description="Location requirements"
    )

    # Metadata
    job_metadata: JobMetadata | None = Field(None, description="Additional job metadata")

    # Search and matching
    keywords: list[str] = Field(default_factory=list, description="Keywords for candidate matching")

    # Quality indicators
    ambiguity_notes: list[str] = Field(
        default_factory=list, description="Fields where JD was ambiguous"
    )
    extraction_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Overall confidence")

    # Processing metadata
    embedding_id: str | None = Field(None, description="Semantic embedding vector ID")
    model_used: str | None = Field(None, description="Model variant used for extraction")


class JobInput(RSASBaseModel):
    """Input for creating a new job."""

    title: str = Field(..., min_length=1, max_length=500, description="Job title")
    description: str = Field(..., min_length=10, description="Job description text")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")
