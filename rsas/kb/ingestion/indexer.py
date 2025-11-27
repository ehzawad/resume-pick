"""Metadata indexer for fast Tier 1 filtering.

Extracts and indexes searchable metadata:
- Skills (for skill-based filtering)
- Companies (for company filtering)
- Job titles (for title filtering)
- Years of experience (for range filtering)
- Last job info (for recency)
"""

from typing import Any

from ...core.models.candidate_profile import CandidateProfile, ParsedResume
from ...observability.logger import get_logger
from .summarizer import CandidateSummary

logger = get_logger(__name__)


class IndexedMetadata:
    """Structured metadata for Tier 1 search indexing."""

    def __init__(
        self,
        indexed_skills: list[str],
        indexed_companies: list[str],
        indexed_titles: list[str],
        years_experience_total: float,
        last_job_title: str | None,
        last_company: str | None,
    ):
        """Initialize indexed metadata.

        Args:
            indexed_skills: List of normalized skills
            indexed_companies: List of companies worked at
            indexed_titles: List of job titles held
            years_experience_total: Total years of experience
            last_job_title: Most recent job title
            last_company: Most recent company
        """
        self.indexed_skills = indexed_skills
        self.indexed_companies = indexed_companies
        self.indexed_titles = indexed_titles
        self.years_experience_total = years_experience_total
        self.last_job_title = last_job_title
        self.last_company = last_company

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "indexed_skills": self.indexed_skills,
            "indexed_companies": self.indexed_companies,
            "indexed_titles": self.indexed_titles,
            "years_experience_total": self.years_experience_total,
            "last_job_title": self.last_job_title,
            "last_company": self.last_company,
        }


class MetadataIndexer:
    """Extract and normalize metadata for fast filtering."""

    def __init__(self):
        """Initialize metadata indexer."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def index_candidate(
        self,
        candidate_profile: CandidateProfile,
        parsed_resume: ParsedResume,
        summary: CandidateSummary,
    ) -> IndexedMetadata:
        """Extract indexed metadata from candidate data.

        Args:
            candidate_profile: Candidate skills profile
            parsed_resume: Parsed resume data
            summary: Generated summary

        Returns:
            IndexedMetadata object
        """
        self.logger.info("indexing_candidate", candidate_id=candidate_profile.candidate_id)

        # Extract skills (from profile + summary)
        profile_skills = [s.skill_name for s in candidate_profile.skills]
        summary_skills = summary.key_skills
        indexed_skills = list(set(profile_skills + summary_skills))  # Deduplicate

        # Extract companies
        indexed_companies = []
        if parsed_resume.experience:
            for exp in parsed_resume.experience:
                if exp.company:
                    indexed_companies.append(self._normalize_company(exp.company))
        indexed_companies = list(set(indexed_companies))  # Deduplicate

        # Extract job titles
        indexed_titles = []
        if parsed_resume.experience:
            for exp in parsed_resume.experience:
                if exp.job_title:
                    indexed_titles.append(self._normalize_title(exp.job_title))
        indexed_titles = list(set(indexed_titles))  # Deduplicate

        # Years of experience
        years_experience_total = candidate_profile.total_years_experience or summary.years_experience or 0

        # Last job info
        last_job_title = None
        last_company = None
        if parsed_resume.experience:
            most_recent = parsed_resume.experience[0]  # Assuming sorted by recency
            last_job_title = most_recent.job_title
            last_company = most_recent.company

        indexed_metadata = IndexedMetadata(
            indexed_skills=indexed_skills[:50],  # Limit to 50 skills
            indexed_companies=indexed_companies[:20],  # Limit to 20 companies
            indexed_titles=indexed_titles[:15],  # Limit to 15 titles
            years_experience_total=years_experience_total,
            last_job_title=last_job_title,
            last_company=last_company,
        )

        self.logger.info(
            "candidate_indexed",
            candidate_id=candidate_profile.candidate_id,
            skills_count=len(indexed_skills),
            companies_count=len(indexed_companies),
            titles_count=len(indexed_titles),
        )

        return indexed_metadata

    def _normalize_company(self, company: str) -> str:
        """Normalize company name for better matching.

        Args:
            company: Raw company name

        Returns:
            Normalized company name
        """
        # Remove common suffixes
        suffixes = [" Inc.", " Inc", " LLC", " Ltd.", " Ltd", " Corporation", " Corp.", " Corp"]
        normalized = company.strip()
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                break

        return normalized

    def _normalize_title(self, title: str) -> str:
        """Normalize job title for better matching.

        Args:
            title: Raw job title

        Returns:
            Normalized title
        """
        # Basic normalization: lowercase, strip
        normalized = title.strip()

        # Common abbreviations
        replacements = {
            "Sr.": "Senior",
            "Sr": "Senior",
            "Jr.": "Junior",
            "Jr": "Junior",
            "Mgr": "Manager",
            "Mgr.": "Manager",
            "Eng": "Engineer",
            "Dev": "Developer",
        }

        for abbr, full in replacements.items():
            if abbr in normalized:
                normalized = normalized.replace(abbr, full)

        return normalized

    def index_batch(
        self,
        candidates: list[tuple[CandidateProfile, ParsedResume, CandidateSummary]],
    ) -> list[IndexedMetadata]:
        """Index metadata for multiple candidates.

        Args:
            candidates: List of (profile, resume, summary) tuples

        Returns:
            List of IndexedMetadata objects
        """
        self.logger.info("indexing_batch", count=len(candidates))

        indexed_metadata = []
        for candidate_profile, parsed_resume, summary in candidates:
            try:
                metadata = self.index_candidate(candidate_profile, parsed_resume, summary)
                indexed_metadata.append(metadata)
            except Exception as e:
                self.logger.error(
                    "indexing_failed",
                    candidate_id=candidate_profile.candidate_id,
                    error=str(e),
                    exc_info=True,
                )
                # Add empty metadata as placeholder
                indexed_metadata.append(
                    IndexedMetadata(
                        indexed_skills=[],
                        indexed_companies=[],
                        indexed_titles=[],
                        years_experience_total=0,
                        last_job_title=None,
                        last_company=None,
                    )
                )

        self.logger.info("batch_indexing_complete", total=len(indexed_metadata))
        return indexed_metadata
