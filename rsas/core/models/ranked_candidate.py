"""RankedCandidate models for final rankings and tiers (Pydantic only)."""

from typing import Any

from pydantic import Field

from .base import (
    IdentifiedSchema,
    RSASBaseModel,
    TimestampSchema,
    VersionedSchema,
)
from .enums import Tier


# =============================================================================
# Pydantic Schemas
# =============================================================================


class RankedCandidate(IdentifiedSchema, TimestampSchema, VersionedSchema):
    """Ranked candidate with tier and summary."""

    job_id: str = Field(..., description="Job identifier")
    candidate_id: str = Field(..., description="Candidate identifier")

    # Ranking
    rank: int = Field(..., ge=1, description="Rank (1 = best)")
    percentile: float = Field(..., ge=0.0, le=100.0, description="Percentile")
    tier: Tier = Field(..., description="Tier classification")

    # Scores (denormalized for convenience)
    total_score: float = Field(..., ge=0.0, le=100.0, description="Total score")
    normalized_score: float | None = Field(None, ge=0.0, le=1.0, description="Normalized score")

    # Recruiter-facing summary
    summary: str = Field("", description="1-3 sentence rationale")
    strengths: list[str] = Field(default_factory=list, description="Key strengths")
    concerns: list[str] = Field(default_factory=list, description="Key concerns")

    # Metadata
    ranking_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional ranking metadata"
    )


class RankedList(RSASBaseModel):
    """Complete ranked list for a job."""

    job_id: str = Field(..., description="Job identifier")
    rankings: list[RankedCandidate] = Field(..., description="Ranked candidates")
    total_candidates: int = Field(..., ge=0, description="Total number of candidates")

    # Statistics
    tier_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count per tier"
    )
    score_statistics: dict[str, float] = Field(
        default_factory=dict, description="Score statistics (mean, median, std)"
    )


# =============================================================================
# SQLAlchemy ORM Models
# =============================================================================


# SQL DB model removed
