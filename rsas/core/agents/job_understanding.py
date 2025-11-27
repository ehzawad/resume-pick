"""Job Understanding Agent - extracts structured requirements from job descriptions."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.enums import AgentType
from ..models.job_profile import EnrichedJobProfile
from .base import BaseAgent


class JobDescriptionInput(RSASBaseModel):
    """Input for Job Understanding Agent."""

    job_id: str = Field(..., description="Job identifier")
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Raw job description text")


class JobUnderstandingAgent(BaseAgent[JobDescriptionInput, EnrichedJobProfile]):
    """Agent that analyzes job descriptions and extracts structured requirements."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.JOB_UNDERSTANDING

    @property
    def output_schema(self) -> Type[EnrichedJobProfile]:
        return EnrichedJobProfile

    async def process(
        self, input_data: JobDescriptionInput, context: AgentContext
    ) -> AgentResult[EnrichedJobProfile]:
        """Process job description and extract structured profile.

        Args:
            input_data: Job description input
            context: Execution context

        Returns:
            AgentResult with EnrichedJobProfile
        """
        if os.getenv("RSAS_TEST_MODE"):
            profile = EnrichedJobProfile(
                job_id=input_data.job_id,
                title=input_data.title,
                raw_description=input_data.description,
                must_have_skills=[],
                nice_to_have_skills=[],
            )
            return AgentResult(success=True, data=profile, confidence=1.0, metadata={"mock": True})

        # Build prompt
        prompt = self._build_prompt(input_data, context)

        # Call Response API
        output, metadata = await self._call_response_api(prompt, context)

        # Set job_id explicitly
        output.job_id = input_data.job_id
        output.title = input_data.title
        output.raw_description = input_data.description

        return AgentResult(
            success=True,
            data=output,
            confidence=output.extraction_confidence,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: JobDescriptionInput, context: AgentContext) -> str:
        """Build prompt for job understanding.

        Args:
            input_data: Input data
            context: Execution context

        Returns:
            Prompt string
        """
        return f"""Analyze this job description and extract structured requirements with high precision.

JOB TITLE: {input_data.title}

JOB DESCRIPTION:
{input_data.description}

Extract the following with confidence scores:

1. **Must-have skills and qualifications** (critical requirements):
   - Identify technical skills, tools, frameworks, languages
   - Categorize each skill (e.g., "programming_language", "framework", "tool", "methodology")
   - Assign importance weights (0.0-1.0)
   - Mark if it's a hard requirement (absence creates ceiling on score)
   - Estimate minimum years of experience if mentioned

2. **Nice-to-have skills and qualifications** (preferred but not required):
   - Similar structure to must-haves
   - Lower weights overall

3. **Experience requirements**:
   - Minimum overall years of experience
   - Minimum years in relevant domain/role
   - Maximum years (seniority ceiling) if mentioned
   - Expected seniority level

4. **Education requirements**:
   - Minimum degree level
   - Preferred fields of study
   - Required certifications

5. **Location requirements**:
   - Work arrangement (remote/onsite/hybrid/flexible)
   - Accepted locations (countries/regions)
   - Timezone preferences

6. **Domain tags** (with confidence):
   - Industry domains (e.g., "healthcare", "fintech", "e-commerce")
   - Technical domains (e.g., "machine_learning", "web_development")

7. **Soft skills**:
   - Communication, leadership, teamwork, etc.

8. **Job metadata**:
   - Department/team
   - Role family (e.g., "Engineering", "Product")
   - Employment type
   - Visa sponsorship availability
   - Security clearance requirements

9. **Search keywords** for candidate matching

10. **Ambiguity notes**:
    - List any fields where the JD was ambiguous or unclear
    - Note assumptions made

Provide confidence scores (0.0-1.0) for each extraction.
Be precise and conservative - only extract what is clearly stated or strongly implied."""
