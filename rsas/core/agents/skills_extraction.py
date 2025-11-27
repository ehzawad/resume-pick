"""Skills Extraction Agent - extracts skills with evidence from parsed resumes."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.candidate_profile import CandidateProfile, ParsedResume
from ..models.enums import AgentType
from ..models.job_profile import EnrichedJobProfile
from .base import BaseAgent


class SkillsExtractionInput(RSASBaseModel):
    """Input for Skills Extraction Agent."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")
    parsed_resume: ParsedResume = Field(..., description="Parsed resume")
    job_profile: EnrichedJobProfile = Field(..., description="Job profile for context")


class SkillsExtractionAgent(BaseAgent[SkillsExtractionInput, CandidateProfile]):
    """Agent that extracts skills with evidence and computes experience metrics."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SKILLS_EXTRACTION

    @property
    def output_schema(self) -> Type[CandidateProfile]:
        return CandidateProfile

    async def process(
        self, input_data: SkillsExtractionInput, context: AgentContext
    ) -> AgentResult[CandidateProfile]:
        """Extract skills and build candidate profile.

        Args:
            input_data: Skills extraction input
            context: Execution context

        Returns:
            AgentResult with CandidateProfile
        """
        if os.getenv("RSAS_TEST_MODE"):
            output = CandidateProfile(
                candidate_id=input_data.candidate_id,
                job_id=input_data.job_id,
                resume_id=input_data.parsed_resume.id,
                skills=[],
                total_years_experience=1.0,
                relevant_years_experience=1.0,
                recency_index=1.0,
            )
            return AgentResult(success=True, data=output, confidence=1.0, metadata={"mock": True})

        prompt = self._build_prompt(input_data, context)
        output, metadata = await self._call_response_api(prompt, context)

        # Set required fields
        output.candidate_id = input_data.candidate_id
        output.job_id = input_data.job_id
        output.resume_id = input_data.parsed_resume.id

        return AgentResult(
            success=True,
            data=output,
            confidence=output.recency_index,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: SkillsExtractionInput, context: AgentContext) -> str:
        """Build prompt for skills extraction."""
        job_skills = "\n".join(
            f"- {skill.name} ({skill.category or 'general'})"
            for skill in (
                input_data.job_profile.must_have_skills
                + input_data.job_profile.nice_to_have_skills
            )
        )

        experience_text = "\n\n".join(
            f"**{exp.job_title} at {exp.company}** ({exp.start_date} - {exp.end_date or 'Present'})\n"
            + "\n".join(f"- {resp}" for resp in exp.responsibilities)
            for exp in input_data.parsed_resume.experience
        )

        return f"""Extract and analyze skills from this candidate's resume, focusing on relevance to the job.

**JOB-RELEVANT SKILLS** (to prioritize):
{job_skills}

**CANDIDATE'S EXPLICIT SKILLS**:
{', '.join(input_data.parsed_resume.explicit_skills)}

**CANDIDATE'S WORK EXPERIENCE**:
{experience_text}

Extract:

1. **All technical skills** (both explicit and inferred from experience):
   - Normalize names (e.g., "React.js" → "React", "Python 3" → "Python")
   - Assign confidence (0.0-1.0)
   - Estimate years of experience based on job durations
   - Determine last used date (most recent job mentioning it)
   - Provide evidence: specific bullet points or sections mentioning the skill (≤20 words)

2. **Skills categorization**:
   - Match to job requirements where possible
   - Categorize: programming_languages, frameworks, tools, cloud, databases, methodologies

3. **Aggregate experience metrics**:
   - Total years of professional experience
   - Years in roles relevant to this job
   - Recency index (0.0-1.0, weighted by how recent skills were used)

4. **Domain alignment**:
   - Extract domain tags from companies, projects, technologies
   - Assign confidence scores

5. **Employment analysis**:
   - Identify employment gaps (> 6 months with no job listed)
   - Flag quality issues (overlapping jobs, inconsistent dates)

Prioritize skills relevant to the job requirements. Keep output concise:
- Max 25 skills total (prioritize job-relevant).
- Evidence snippets ≤20 words.
- Use null/empty lists when unknown.
- Ensure well-formed JSON that matches the schema and fits comfortably within the token limit."""
