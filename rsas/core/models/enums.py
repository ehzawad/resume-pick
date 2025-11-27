"""Enumeration types for RSAS models."""

from enum import Enum


class JobStatus(str, Enum):
    """Job processing status."""

    DRAFT = "draft"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class AgentType(str, Enum):
    """Agent type identifiers."""

    JOB_UNDERSTANDING = "job_understanding"
    PARSER = "parser"
    SKILLS_EXTRACTION = "skills_extraction"
    MATCHING = "matching"
    SCORING = "scoring"
    RANKING = "ranking"
    BIAS_CHECK = "bias_check"
    OUTPUT = "output"


class ProcessingStage(str, Enum):
    """Candidate processing stages."""

    PENDING = "pending"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    MATCHING = "matching"
    SCORING = "scoring"
    RANKING = "ranking"
    BIAS_CHECKING = "bias_checking"
    COMPLETED = "completed"
    FAILED = "failed"


class Tier(str, Enum):
    """Candidate ranking tiers."""

    TOP_10 = "top_10"
    TOP_25 = "top_25"
    TOP_50 = "top_50"
    BOTTOM_50 = "bottom_50"
    REJECTED = "rejected"


class SeniorityLevel(str, Enum):
    """Seniority levels for job and candidates."""

    INTERN = "intern"
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    LEAD = "lead"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


class EducationLevel(str, Enum):
    """Education levels."""

    NONE = "none"
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    POST_DOC = "post_doc"


class LocationType(str, Enum):
    """Location/work arrangement types."""

    REMOTE = "remote"
    ONSITE = "onsite"
    HYBRID = "hybrid"
    FLEXIBLE = "flexible"


class EmploymentType(str, Enum):
    """Employment types."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERN = "intern"
    TEMPORARY = "temporary"
    FREELANCE = "freelance"
