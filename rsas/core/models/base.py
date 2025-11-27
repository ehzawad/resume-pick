"""Base Pydantic schemas and helpers for RSAS models (SQL-free)."""

import uuid
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Type variable for generic result types
T = TypeVar('T')


# =============================================================================
# Pydantic Base Classes
# =============================================================================


class RSASBaseModel(BaseModel):
    """Base Pydantic model for all schemas with common configuration."""

    model_config = ConfigDict(
        # Allow ORM model conversion
        from_attributes=True,
        # Validate on assignment
        validate_assignment=True,
        # Use enum values instead of enum members
        use_enum_values=True,
        # Strict type checking
        str_strip_whitespace=True,
    )


class TimestampSchema(RSASBaseModel):
    """Schema with timestamp fields."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VersionedSchema(RSASBaseModel):
    """Schema with versioning."""

    schema_version: str = Field(default="1.0.0", description="Schema version")


class IdentifiedSchema(RSASBaseModel):
    """Schema with UUID identifier."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")


# =============================================================================
# Common Response Models
# =============================================================================


class AgentContext(RSASBaseModel):
    """Context passed to all agent executions."""

    job_id: str = Field(..., description="Job identifier")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Trace ID")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentResult(RSASBaseModel, Generic[T]):
    """Standardized result wrapper for agent executions."""

    success: bool = Field(..., description="Whether execution succeeded")
    data: T | None = Field(None, description="Result data")
    error: str | None = Field(None, description="Error message if failed")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score")
    tokens_used: int = Field(0, ge=0, description="Tokens consumed")
    duration_ms: int = Field(0, ge=0, description="Execution duration in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PipelineResult(RSASBaseModel):
    """Result of pipeline execution."""

    success: bool = Field(..., description="Whether pipeline succeeded")
    output: Any | None = Field(None, description="Pipeline output")
    error: str | None = Field(None, description="Error message if failed")
    stats: dict[str, Any] = Field(default_factory=dict, description="Pipeline statistics")


# =============================================================================
# Utility Functions
# =============================================================================


def generate_id(prefix: str = "") -> str:
    """Generate a prefixed UUID.

    Args:
        prefix: Optional prefix for the ID (e.g., "job_", "cand_")

    Returns:
        Prefixed UUID string
    """
    uid = str(uuid.uuid4())
    return f"{prefix}{uid}" if prefix else uid


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        Current datetime in UTC timezone
    """
    return datetime.now(timezone.utc)
