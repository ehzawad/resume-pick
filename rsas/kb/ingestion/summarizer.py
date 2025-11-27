"""Candidate resume summarizer for searchable metadata generation using gpt-5.1."""

from typing import Any

from pydantic import BaseModel, Field

from ...core.models.candidate_profile import CandidateProfile, ParsedResume
from ...integrations.openai_client import OpenAIClient
from ...observability.logger import get_logger

logger = get_logger(__name__)


class CandidateSummary(BaseModel):
    """Structured candidate summary for knowledge base."""

    summary_text: str = Field(..., description="2-3 sentence professional summary")
    key_skills: list[str] = Field(..., description="Top 5-7 key skills")
    experience_highlights: list[str] = Field(..., description="2-3 notable achievements/roles")
    domain_expertise: list[str] = Field(..., description="Industry domains/specializations")
    years_experience: float | None = Field(None, description="Total years of experience")
    education_summary: str | None = Field(None, description="Highest degree and field")


class ResumeSummarizer:
    """Generate searchable summaries from candidate profiles with gpt-5.1."""

    def __init__(self, openai_client: OpenAIClient | None = None):
        """Initialize summarizer.

        Args:
            openai_client: OpenAI client instance
        """
        self.openai_client = openai_client or OpenAIClient(model="gpt-5.1")
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def summarize_candidate(
        self,
        candidate_profile: CandidateProfile,
        parsed_resume: ParsedResume,
    ) -> tuple[CandidateSummary, dict[str, Any]]:
        """Generate searchable summary from candidate profile and parsed resume.

        Args:
            candidate_profile: Candidate skills profile
            parsed_resume: Parsed resume data

        Returns:
            Tuple of (candidate summary, metadata with cost/tokens)
        """
        self.logger.info(
            "summarizing_candidate",
            candidate_id=candidate_profile.candidate_id,
        )

        # Fast path for test/offline mode
        if getattr(self.openai_client, "test_mode", False):
            summary = CandidateSummary(
                summary_text="Mock candidate summary",
                key_skills=[s.skill_name for s in candidate_profile.skills[:5]],
                experience_highlights=[],
                domain_expertise=[],
                years_experience=candidate_profile.total_years_experience,
                education_summary=None,
            )
            return summary, {"tokens_total": 0, "mock": True}

        # Build prompt from available data
        prompt = self._build_summary_prompt(candidate_profile, parsed_resume)

        summary, metadata = await self.openai_client.create_response(
            input_text=[
                {
                    "role": "system",
                    "content": "You are a professional recruiter summarizing candidate profiles for search. "
                    "Create concise, keyword-rich summaries under 300 tokens.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=CandidateSummary,
            model="gpt-5.1",
            max_output_tokens=500,
        )

        self.logger.info(
            "candidate_summarized",
            candidate_id=candidate_profile.candidate_id,
            tokens_used=metadata.get("tokens_total", 0),
        )

        return summary, metadata

    def _build_summary_prompt(
        self, candidate_profile: CandidateProfile, parsed_resume: ParsedResume
    ) -> str:
        """Build prompt for summary generation.

        Args:
            candidate_profile: Candidate skills profile
            parsed_resume: Parsed resume data

        Returns:
            Prompt string
        """
        # Extract key information
        skills = [s.skill_name for s in candidate_profile.skills[:15]]  # Top 15 skills
        total_years = candidate_profile.total_years_experience or 0

        # Work experience
        experience_text = ""
        if parsed_resume.experience:
            for i, exp in enumerate(parsed_resume.experience[:3], 1):  # Top 3 roles
                experience_text += f"\n{i}. {exp.job_title or 'N/A'} at {exp.company or 'N/A'}"
                if exp.responsibilities:
                    # First 2 responsibilities
                    for resp in exp.responsibilities[:2]:
                        experience_text += f"\n   - {resp}"

        # Education
        education_text = ""
        if parsed_resume.education:
            edu = parsed_resume.education[0]  # Highest/most recent
            education_text = f"{edu.degree or 'N/A'} in {edu.field or 'N/A'}"
            if edu.institution:
                education_text += f" from {edu.institution}"

        # Build prompt
        prompt = f"""Summarize this candidate profile for recruiter search:

**SKILLS:** {', '.join(skills)}

**EXPERIENCE:** {total_years:.1f} years total
{experience_text}

**EDUCATION:** {education_text or 'Not specified'}

**CERTIFICATIONS:** {', '.join(parsed_resume.certifications[:5]) if parsed_resume.certifications else 'None listed'}

Generate a structured summary optimized for search:
1. Professional summary (2-3 sentences highlighting key strengths and experience)
2. Top 5-7 key technical/domain skills
3. 2-3 notable achievements or roles
4. Industry domains/specializations
5. Total years of experience
6. Education summary (highest degree and field)

Keep it concise and keyword-rich for semantic search."""

        return prompt

    async def summarize_batch(
        self,
        candidates: list[tuple[CandidateProfile, ParsedResume]],
    ) -> list[tuple[CandidateSummary, dict[str, Any]]]:
        """Generate summaries for multiple candidates.

        Args:
            candidates: List of (candidate_profile, parsed_resume) tuples

        Returns:
            List of (summary, metadata) tuples
        """
        self.logger.info("summarizing_batch", count=len(candidates))

        summaries = []
        for candidate_profile, parsed_resume in candidates:
            try:
                summary, metadata = await self.summarize_candidate(
                    candidate_profile, parsed_resume
                )
                summaries.append((summary, metadata))
            except Exception as e:
                self.logger.error(
                    "summarization_failed",
                    candidate_id=candidate_profile.candidate_id,
                    error=str(e),
                    exc_info=True,
                )
                # Add empty summary as placeholder
                empty_summary = CandidateSummary(
                    summary_text=f"Error summarizing candidate: {str(e)}",
                    key_skills=[],
                    experience_highlights=[],
                    domain_expertise=[],
                )
                summaries.append((empty_summary, {"tokens_total": 0, "error": str(e)}))

        self.logger.info("batch_summarization_complete", total=len(summaries))
        return summaries
