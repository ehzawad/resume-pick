"""Bias Check Agent - checks for potential biases in rankings."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.enums import AgentType
from ..models.ranked_candidate import RankedList
from .base import BaseAgent


class BiasCheckInput(RSASBaseModel):
    """Input for Bias Check Agent."""

    job_id: str = Field(..., description="Job identifier")
    ranked_list: RankedList = Field(..., description="Ranked candidates")


class BiasReport(RSASBaseModel):
    """Bias check report."""

    job_id: str = Field(..., description="Job identifier")
    biases_detected: list[dict[str, str]] = Field(default_factory=list, description="Detected biases")
    fairness_score: float = Field(1.0, ge=0.0, le=1.0, description="Overall fairness 0-1")
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    flags_for_review: list[str] = Field(default_factory=list, description="Candidate IDs to review")


class BiasCheckAgent(BaseAgent[BiasCheckInput, BiasReport]):
    """Agent that checks for potential biases."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.BIAS_CHECK

    @property
    def output_schema(self) -> Type[BiasReport]:
        return BiasReport

    async def process(
        self, input_data: BiasCheckInput, context: AgentContext
    ) -> AgentResult[BiasReport]:
        """Check for biases in rankings."""
        if os.getenv("RSAS_TEST_MODE"):
            report = BiasReport(
                job_id=input_data.job_id,
                biases_detected=[],
                fairness_score=1.0,
                recommendations=[],
                flags_for_review=[],
            )
            return AgentResult(success=True, data=report, confidence=1.0, metadata={"mock": True})

        prompt = self._build_prompt(input_data, context)
        output, metadata = await self._call_response_api(prompt, context)

        output.job_id = input_data.job_id

        return AgentResult(
            success=True,
            data=output,
            confidence=output.fairness_score,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: BiasCheckInput, context: AgentContext) -> str:
        """Build bias check prompt."""
        tier_dist = input_data.ranked_list.tier_distribution

        return f"""Analyze this ranking for potential biases and fairness issues.

**RANKING DISTRIBUTION:**
- Total candidates: {input_data.ranked_list.total_candidates}
- Tier distribution: {tier_dist}

**CHECK FOR:**

1. **Concentration biases**: Are top candidates too homogeneous?
2. **Score clustering**: Suspicious patterns in scores
3. **Tier imbalances**: Unusual tier distributions
4. **Must-have penalties**: Over-penalizing for single missing skills
5. **Recency bias**: Over-weighting recent experience

**OUTPUT:**

1. **Biases detected**: List any potential biases found with severity (low/medium/high)
2. **Fairness score** (0.0-1.0): Overall fairness assessment
3. **Recommendations**: Suggestions to improve fairness
4. **Flags for review**: Candidate IDs that should be manually reviewed

Be thorough but not overly conservative. Focus on significant issues."""
