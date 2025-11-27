"""Output Agent - generates formatted outputs for different audiences."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.enums import AgentType
from ..models.ranked_candidate import RankedList
from .bias_check import BiasReport
from .base import BaseAgent


class OutputInput(RSASBaseModel):
    """Input for Output Agent."""

    job_id: str = Field(..., description="Job identifier")
    ranked_list: RankedList = Field(..., description="Ranked candidates")
    bias_report: BiasReport | None = Field(None, description="Bias report")


class FormattedOutput(RSASBaseModel):
    """Formatted output for various audiences."""

    job_id: str = Field(..., description="Job identifier")
    executive_summary: str = Field(..., description="Executive summary")
    top_candidates_summary: list[dict[str, object]] = Field(
        ..., description="Top N summaries (lenient types from model output)"
    )
    full_report_json: dict = Field(..., description="Complete data as JSON")
    recruiter_notes: list[str] = Field(default_factory=list, description="Notes for recruiters")


class OutputAgent(BaseAgent[OutputInput, FormattedOutput]):
    """Agent that generates formatted outputs."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.OUTPUT

    @property
    def output_schema(self) -> Type[FormattedOutput]:
        return FormattedOutput

    async def process(
        self, input_data: OutputInput, context: AgentContext
    ) -> AgentResult[FormattedOutput]:
        """Generate formatted outputs."""
        if os.getenv("RSAS_TEST_MODE"):
            mock_output = FormattedOutput(
                job_id=input_data.job_id,
                executive_summary="Mock executive summary",
                top_candidates_summary=[
                    {"name": "Candidate 1", "rank": "1", "score": "70.0", "summary": "Mock summary"}
                ],
                full_report_json={},
                recruiter_notes=["Mock note"],
            )
            return AgentResult(success=True, data=mock_output, metadata={"mock": True})

        prompt = self._build_prompt(input_data, context)
        output, metadata = await self._call_response_api(prompt, context)

        output.job_id = input_data.job_id

        return AgentResult(
            success=True,
            data=output,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: OutputInput, context: AgentContext) -> str:
        """Build output generation prompt."""
        top_10 = [r for r in input_data.ranked_list.rankings if r.rank <= 10]

        return f"""Generate formatted output for this job's candidate rankings.

**PROCESSED:** {input_data.ranked_list.total_candidates} candidates
**TOP TIER:** {len([r for r in input_data.ranked_list.rankings if r.tier == 'top_10'])} candidates

**GENERATE:**

1. **Executive Summary** (2-3 paragraphs):
   - Overview of candidate pool quality
   - Top tier highlights
   - Key trends observed
   - Any concerns from bias report

2. **Top Candidates Summary** (top 10):
   For each: {{
     "name": "Candidate {{rank}}",
     "rank": {{rank}},
     "score": {{score}},
     "summary": "2-3 sentence summary",
     "key_strengths": ["strength1", "strength2"],
     "considerations": ["any concerns"]
   }}

3. **Full Report JSON**: Complete structured data

4. **Recruiter Notes**:
   - Action items
   - Interview recommendations
   - Diversity considerations
   - Screening suggestions

Be professional, concise, and actionable."""
