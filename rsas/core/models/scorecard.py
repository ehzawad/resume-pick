"""ScoreCard models for candidate scoring and evaluation (Pydantic only)."""

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


class ScoreDimensions(RSASBaseModel):
    """Multi-dimensional scoring breakdown."""

    technical_skills: float = Field(0.0, ge=0.0, le=100.0, description="Technical skills score")
    experience: float = Field(0.0, ge=0.0, le=100.0, description="Experience score")
    education: float = Field(0.0, ge=0.0, le=100.0, description="Education score")
    culture_fit: float = Field(0.0, ge=0.0, le=100.0, description="Culture fit score")
    career_trajectory: float = Field(0.0, ge=0.0, le=100.0, description="Career trajectory score")


class MustHaveDetail(RSASBaseModel):
    """Detail about a must-have requirement."""

    requirement: str = Field(..., description="Requirement name")
    met: bool = Field(..., description="Whether requirement is met")
    evidence: str | None = Field(None, description="Evidence summary")
    penalty: float = Field(0.0, ge=0.0, description="Penalty if missing")


class RedFlag(RSASBaseModel):
    """A concern or warning about the candidate."""

    code: str = Field(..., description="Machine-readable code")
    severity: str = Field(..., description="Severity level (high, medium, low)")
    description: str = Field(..., description="Human-readable description")


class ScoreCard(IdentifiedSchema, TimestampSchema, VersionedSchema):
    """Complete scorecard for a candidate-job pairing."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")

    # Scores
    dimensions: ScoreDimensions = Field(..., description="Dimensional scores")
    total_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Score confidence")

    # Must-have analysis
    must_have_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Must-have coverage %")
    must_have_details: list[MustHaveDetail] = Field(
        default_factory=list, description="Must-have coverage details"
    )
    missing_must_haves: list[str] = Field(
        default_factory=list, description="Missing must-haves"
    )

    # Nice-to-have analysis
    nice_to_have_coverage: float = Field(
        0.0, ge=0.0, le=1.0, description="Nice-to-have coverage %"
    )

    # Flags and highlights
    red_flags: list[RedFlag] = Field(default_factory=list, description="Red flags")
    standout_areas: list[str] = Field(default_factory=list, description="Standout strengths")

    # Explanations
    justifications: dict[str, str] = Field(
        default_factory=dict, description="Justifications per dimension"
    )
    summary: str = Field("", description="Overall summary")

    # Configuration used
    scoring_config_id: str | None = Field(None, description="Scoring configuration ID")
