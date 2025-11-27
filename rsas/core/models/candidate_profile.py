"""Candidate Profile models for resumes and their parsed representations (Pydantic only)."""

from datetime import datetime
from typing import Any

from pydantic import Field

from .base import (
    IdentifiedSchema,
    RSASBaseModel,
    TimestampSchema,
    VersionedSchema,
)


# =============================================================================
# Pydantic Schemas
# =============================================================================


class EducationEntry(RSASBaseModel):
    """Single education entry."""

    degree: str | None = Field(None, description="Degree type")
    field: str | None = Field(None, description="Field of study")
    institution: str | None = Field(None, description="Institution name")
    start_date: datetime | None = Field(None, description="Start date")
    end_date: datetime | None = Field(None, description="End date")
    honors: list[str] = Field(default_factory=list, description="Honors and awards")


class WorkExperienceEntry(RSASBaseModel):
    """Single work experience entry."""

    job_title: str | None = Field(None, description="Job title")
    company: str | None = Field(None, description="Company name")
    location: str | None = Field(None, description="Location")
    start_date: datetime | None = Field(None, description="Start date")
    end_date: datetime | None = Field(None, description="End date (None if current)")
    is_current: bool = Field(False, description="Currently employed")
    responsibilities: list[str] = Field(default_factory=list, description="Responsibilities")


class ContactInfo(RSASBaseModel):
    """Candidate contact information (PII)."""

    name: str | None = Field(None, description="Full name")
    email: str | None = Field(None, description="Email address")
    phone: str | None = Field(None, description="Phone number")
    location: str | None = Field(None, description="Location text")
    linkedin: str | None = Field(None, description="LinkedIn URL")
    github: str | None = Field(None, description="GitHub URL")
    portfolio: str | None = Field(None, description="Portfolio URL")
    website: str | None = Field(None, description="Personal website")


class ParsedResume(IdentifiedSchema, TimestampSchema, VersionedSchema):
    """Parsed resume structure created by Parser Agent."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Associated job ID")

    # Contact (PII - access controlled)
    contact: ContactInfo | None = Field(None, description="Contact information")

    # Resume sections
    education: list[EducationEntry] = Field(default_factory=list, description="Education history")
    experience: list[WorkExperienceEntry] = Field(
        default_factory=list, description="Work experience"
    )
    explicit_skills: list[str] = Field(default_factory=list, description="Listed skills")
    projects: list[dict[str, Any]] = Field(default_factory=list, description="Projects")
    publications: list[dict[str, Any]] = Field(default_factory=list, description="Publications")
    certifications: list[str] = Field(default_factory=list, description="Certifications")

    # Raw text
    raw_text: str = Field("", description="Raw extracted text")

    # Parsing quality
    parse_confidence: float = Field(1.0, ge=0.0, le=1.0, description="Parsing confidence")
    missing_sections: list[str] = Field(
        default_factory=list, description="Sections not detected"
    )

    # Provenance
    model_used: str | None = Field(None, description="Model used for parsing")
    pdf_library: str | None = Field(None, description="PDF library used")


class SkillEvidence(RSASBaseModel):
    """Evidence for a skill from resume."""

    section_type: str = Field(..., description="Section type (e.g., 'experience')")
    job_index: int | None = Field(None, description="Job index if from experience")
    bullet_index: int | None = Field(None, description="Bullet index if from responsibilities")
    snippet: str = Field(..., description="Text snippet showing evidence")


class CandidateSkill(RSASBaseModel):
    """Single skill with evidence and metadata."""

    skill_name: str = Field(..., description="Skill name (normalized)")
    taxonomy_id: str | None = Field(None, description="Normalized taxonomy ID")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")
    years_experience: float | None = Field(None, ge=0, description="Estimated years")
    last_used_date: datetime | None = Field(None, description="Last usage date")
    recency_score: float = Field(1.0, ge=0.0, le=1.0, description="Recency score")
    evidence: list[SkillEvidence] = Field(default_factory=list, description="Evidence spans")


class CandidateProfile(IdentifiedSchema, TimestampSchema, VersionedSchema):
    """Complete candidate profile created by Skills Extraction Agent."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Associated job ID")
    resume_id: str = Field(..., description="Parsed resume ID")

    # Skills with evidence
    skills: list[CandidateSkill] = Field(default_factory=list, description="Extracted skills")

    # Aggregate metrics
    total_years_experience: float | None = Field(None, ge=0, description="Total years")
    relevant_years_experience: float | None = Field(
        None, ge=0, description="Years in relevant domain"
    )
    recency_index: float = Field(1.0, ge=0.0, le=1.0, description="Aggregate recency score")

    # Domain alignment
    domain_tags: dict[str, float] = Field(
        default_factory=dict, description="Domain tags with confidence"
    )

    # Quality flags
    employment_gaps: list[dict[str, Any]] = Field(
        default_factory=list, description="Employment gaps"
    )
    quality_flags: list[str] = Field(default_factory=list, description="Quality issues")


# =============================================================================
# SQLAlchemy ORM Models
# =============================================================================


# SQL DB model removed
