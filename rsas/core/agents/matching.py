"""Matching Agent - matches candidate profiles to job requirements."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.candidate_profile import CandidateProfile
from ..models.enums import AgentType
from ..models.job_profile import EnrichedJobProfile
from .base import BaseAgent


class MatchingInput(RSASBaseModel):
    """Input for Matching Agent."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")
    candidate_profile: CandidateProfile = Field(..., description="Candidate profile")
    job_profile: EnrichedJobProfile = Field(..., description="Job profile")


class MatchReport(RSASBaseModel):
    """Match report for candidate-job pairing."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")
    must_have_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Must-have coverage")
    nice_to_have_coverage: float = Field(0.0, ge=0.0, le=1.0, description="Nice-to-have coverage")
    skill_matches: list[dict[str, str]] = Field(default_factory=list, description="Skill matches")
    skill_gaps: list[str] = Field(default_factory=list, description="Missing skills")
    experience_alignment: float = Field(0.0, ge=0.0, le=1.0, description="Experience alignment")
    education_match: bool = Field(False, description="Education requirements met")
    overall_match_score: float = Field(0.0, ge=0.0, le=100.0, description="Overall match 0-100")
    match_details: dict[str, str] = Field(default_factory=dict, description="Detailed matching")


class MatchingAgent(BaseAgent[MatchingInput, MatchReport]):
    """Agent that matches candidates to job requirements."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.MATCHING

    @property
    def output_schema(self) -> Type[MatchReport]:
        return MatchReport

    async def process(
        self, input_data: MatchingInput, context: AgentContext
    ) -> AgentResult[MatchReport]:
        """Match candidate to job requirements."""
        if os.getenv("RSAS_TEST_MODE"):
            output = MatchReport(
                candidate_id=input_data.candidate_id,
                job_id=input_data.job_id,
                must_have_coverage=0.5,
                nice_to_have_coverage=0.5,
                skill_matches=[],
                skill_gaps=[],
                experience_alignment=0.5,
                education_match=True,
                overall_match_score=50.0,
                match_details={},
            )
            return AgentResult(success=True, data=output, confidence=0.5, metadata={"mock": True})

        prompt = self._build_prompt(input_data, context)
        output, metadata = await self._call_response_api(prompt, context)

        output.candidate_id = input_data.candidate_id
        output.job_id = input_data.job_id

        return AgentResult(
            success=True,
            data=output,
            confidence=output.overall_match_score / 100.0,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: MatchingInput, context: AgentContext) -> str:
        """Build matching prompt."""
        job = input_data.job_profile
        cand = input_data.candidate_profile

        must_haves = "\n".join(f"- {s.name} ({s.category or 'general'})" for s in job.must_have_skills)
        nice_to_haves = "\n".join(f"- {s.name}" for s in job.nice_to_have_skills)
        cand_skills = "\n".join(
            f"- {s.skill_name} ({s.years_experience or 0} years)"
            for s in cand.skills
        )

        return f"""Match this candidate to the job requirements with precision.

**JOB REQUIREMENTS:**

Must-Have Skills:
{must_haves}

Nice-to-Have Skills:
{nice_to_haves}

Experience Required: {job.experience_requirements.min_years_overall if job.experience_requirements else 'Not specified'} years
Education Required: {job.education_requirements.min_degree if job.education_requirements else 'Not specified'}

**CANDIDATE PROFILE:**

Skills:
{cand_skills}

Total Experience: {cand.total_years_experience or 0} years
Relevant Experience: {cand.relevant_years_experience or 0} years

Analyze:

1. **Must-have coverage** (0.0-1.0): What fraction of must-haves does the candidate have?
2. **Nice-to-have coverage** (0.0-1.0): What fraction of nice-to-haves?
3. **Skill matches**: List matching skills with evidence
4. **Skill gaps**: List critical missing must-haves
5. **Experience alignment** (0.0-1.0): How well does experience match requirements?
6. **Education match**: Boolean - meets minimum education requirement
7. **Overall match score** (0-100): Weighted combination considering must-haves heavily
8. **Match details**: Explain key matches and gaps

Be precise and evidence-based."""
