"""Scoring Agent - computes multi-dimensional scores for candidates."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.candidate_profile import CandidateProfile
from ..models.enums import AgentType
from ..models.job_profile import EnrichedJobProfile
from ..models.scorecard import ScoreCard, ScoreDimensions
from .matching import MatchReport
from .base import BaseAgent


class ScoringInput(RSASBaseModel):
    """Input for Scoring Agent."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")
    match_report: MatchReport = Field(..., description="Match report")
    candidate_profile: CandidateProfile = Field(..., description="Candidate profile")
    job_profile: EnrichedJobProfile = Field(..., description="Job profile")


class ScoringAgent(BaseAgent[ScoringInput, ScoreCard]):
    """Agent that computes detailed scorecards."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SCORING

    @property
    def output_schema(self) -> Type[ScoreCard]:
        return ScoreCard

    async def process(
        self, input_data: ScoringInput, context: AgentContext
    ) -> AgentResult[ScoreCard]:
        """Compute scorecard for candidate."""
        if os.getenv("RSAS_TEST_MODE"):
            scorecard = ScoreCard(
                candidate_id=input_data.candidate_id,
                job_id=input_data.job_id,
                dimensions=ScoreDimensions(
                    technical_skills=70.0,
                    experience=70.0,
                    education=70.0,
                    culture_fit=70.0,
                    career_trajectory=70.0,
                ),
                total_score=70.0,
                confidence=0.8,
                must_have_coverage=input_data.match_report.must_have_coverage,
                missing_must_haves=input_data.match_report.skill_gaps,
                red_flags=[],
                standout_areas=[],
                justifications={},
                summary="mock score",
            )
            return AgentResult(success=True, data=scorecard, confidence=0.8, metadata={"mock": True})

        prompt = self._build_prompt(input_data, context)
        output, metadata = await self._call_response_api(prompt, context)

        output.candidate_id = input_data.candidate_id
        output.job_id = input_data.job_id

        return AgentResult(
            success=True,
            data=output,
            confidence=output.confidence,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: ScoringInput, context: AgentContext) -> str:
        """Build scoring prompt.

        Note: Scoring weights from config are applied by the LLM following prompt instructions.
        The LLM computes dimension scores (0-100) and then calculates the weighted average
        using the weights specified in the prompt. This is by design for LLM-based scoring.
        """
        weights = context.config.get("agents", {}).get("scoring", {}).get("weights", {})

        return f"""Compute a detailed scorecard for this candidate using multi-dimensional scoring.

**MATCH ANALYSIS:**
- Must-have coverage: {input_data.match_report.must_have_coverage:.1%}
- Nice-to-have coverage: {input_data.match_report.nice_to_have_coverage:.1%}
- Overall match: {input_data.match_report.overall_match_score:.1f}/100
- Skill gaps: {', '.join(input_data.match_report.skill_gaps) or 'None'}

**SCORING WEIGHTS:**
- Technical Skills: {weights.get('technical_skills', 0.40):.0%}
- Experience: {weights.get('experience', 0.30):.0%}
- Education: {weights.get('education', 0.15):.0%}
- Culture Fit: {weights.get('culture_fit', 0.10):.0%}
- Career Trajectory: {weights.get('career_trajectory', 0.05):.0%}

Compute scores (0-100) for each dimension:

1. **Technical Skills** (0-100): Based on must-have coverage, skill depth, recency
2. **Experience** (0-100): Based on years, relevance, progression
3. **Education** (0-100): Based on degree level, field match
4. **Culture Fit** (0-100): Based on soft skills, domain alignment
5. **Career Trajectory** (0-100): Based on progression, stability

Then:
- **Total Score** (0-100): Weighted average of dimensions
- **Confidence** (0.0-1.0): How confident in this scoring
- **Must-have coverage**: {input_data.match_report.must_have_coverage}
- **Missing must-haves**: List critical gaps
- **Red flags**: Any concerns (experience mismatch, gaps, etc.)
- **Standout areas**: Key strengths
- **Justifications**: Explain each dimension score

Be rigorous and evidence-based."""
