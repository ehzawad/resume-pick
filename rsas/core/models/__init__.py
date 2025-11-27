"""RSAS data models for jobs, candidates, scoring, and ranking."""

from .audit import AgentTrace, ConfigVersion, PipelineState
from .base import (
    AgentContext,
    AgentResult,
    IdentifiedSchema,
    PipelineResult,
    RSASBaseModel,
    TimestampSchema,
    VersionedSchema,
    generate_id,
    utc_now,
)
from .candidate_profile import (
    CandidateProfile,
    CandidateSkill,
    ContactInfo,
    EducationEntry,
    ParsedResume,
    SkillEvidence,
    WorkExperienceEntry,
)
from .enums import (
    AgentType,
    EducationLevel,
    EmploymentType,
    JobStatus,
    LocationType,
    ProcessingStage,
    SeniorityLevel,
    Tier,
)
from .job_profile import (
    EducationRequirement,
    EnrichedJobProfile,
    ExperienceRequirement,
    JobInput,
    JobMetadata,
    LocationRequirement,
    Skill,
)
from .ranked_candidate import RankedCandidate, RankedList
from .scorecard import MustHaveDetail, RedFlag, ScoreCard, ScoreDimensions

__all__ = [
    # Base
    "RSASBaseModel",
    "IdentifiedSchema",
    "TimestampSchema",
    "VersionedSchema",
    "AgentContext",
    "AgentResult",
    "PipelineResult",
    "generate_id",
    "utc_now",
    # Enums
    "JobStatus",
    "AgentType",
    "ProcessingStage",
    "Tier",
    "SeniorityLevel",
    "EducationLevel",
    "LocationType",
    "EmploymentType",
    # Job Profile
    "EnrichedJobProfile",
    "JobInput",
    "Skill",
    "ExperienceRequirement",
    "EducationRequirement",
    "LocationRequirement",
    "JobMetadata",
    # Candidate Profile
    "CandidateProfile",
    "ParsedResume",
    "ContactInfo",
    "EducationEntry",
    "WorkExperienceEntry",
    "CandidateSkill",
    "SkillEvidence",
    # ScoreCard
    "ScoreCard",
    "ScoreDimensions",
    "MustHaveDetail",
    "RedFlag",
    # Ranked Candidate
    "RankedCandidate",
    "RankedList",
    # Audit
    "AgentTrace",
    "PipelineState",
    "ConfigVersion",
]
