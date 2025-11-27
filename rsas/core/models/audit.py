"""Audit models for tracing, logging, and checkpoint management (Pydantic only)."""

from typing import Any

from pydantic import Field

from .base import (
    IdentifiedSchema,
    RSASBaseModel,
    TimestampSchema,
    VersionedSchema,
)
from .enums import AgentType, JobStatus, ProcessingStage


# =============================================================================
# Pydantic Schemas
# =============================================================================


class AgentTrace(IdentifiedSchema, TimestampSchema):
    """Agent execution trace for idempotency and debugging."""

    job_id: str = Field(..., description="Job identifier")
    agent_type: AgentType = Field(..., description="Agent type")

    # Idempotency
    input_hash: str = Field(..., description="SHA-256 hash of input for idempotency")

    # Execution data
    output_data: dict[str, Any] | None = Field(None, description="Agent output")
    success: bool = Field(True, description="Whether execution succeeded")
    error: str | None = Field(None, description="Error message if failed")

    # Performance metrics
    duration_ms: int = Field(0, ge=0, description="Execution duration in milliseconds")
    tokens_used: int = Field(0, ge=0, description="Tokens consumed")
    tokens_input: int = Field(0, ge=0, description="Input tokens")
    tokens_output: int = Field(0, ge=0, description="Output tokens")

    # Model info
    model_used: str | None = Field(None, description="Model used")
    reasoning_effort: str | None = Field(None, description="Reasoning effort level")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PipelineState(IdentifiedSchema, TimestampSchema, VersionedSchema):
    """Pipeline execution state for checkpointing and resume."""

    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    current_stage: ProcessingStage = Field(..., description="Current processing stage")

    # Progress tracking
    completed_resumes: list[str] = Field(
        default_factory=list, description="Completed resume IDs"
    )
    failed_resumes: list[tuple[str, str]] = Field(
        default_factory=list, description="Failed resume IDs with errors"
    )

    # Timing
    started_at: Any = Field(..., description="Pipeline start time")
    last_checkpoint_at: Any | None = Field(None, description="Last checkpoint time")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConfigVersion(IdentifiedSchema, TimestampSchema):
    """Configuration snapshot for reproducibility."""

    job_id: str | None = Field(None, description="Job ID (None for global configs)")
    tenant_id: str | None = Field(None, description="Tenant ID (None for global)")

    # Configuration data
    config_data: dict[str, Any] = Field(..., description="Configuration snapshot")
    version: str = Field(..., description="Configuration version")

    # Description
    description: str | None = Field(None, description="Configuration description")


# =============================================================================
# SQLAlchemy ORM Models
# =============================================================================


# SQL-free; DB models removed.
