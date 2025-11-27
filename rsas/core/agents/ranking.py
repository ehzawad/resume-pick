"""Ranking Agent - ranks candidates and assigns tiers."""

import os
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.enums import AgentType, Tier
from ..models.ranked_candidate import RankedCandidate, RankedList
from ..models.scorecard import ScoreCard
from .base import BaseAgent


class RankingInput(RSASBaseModel):
    """Input for Ranking Agent."""

    job_id: str = Field(..., description="Job identifier")
    scorecards: list[ScoreCard] = Field(..., description="All scorecards for job")


class RankingAgent(BaseAgent[RankingInput, RankedList]):
    """Agent that ranks candidates and assigns tiers."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.RANKING

    @property
    def output_schema(self) -> Type[RankedList]:
        return RankedList

    async def process(
        self, input_data: RankingInput, context: AgentContext
    ) -> AgentResult[RankedList]:
        """Rank candidates and assign tiers."""
        if os.getenv("RSAS_TEST_MODE"):
            sorted_cards = sorted(input_data.scorecards, key=lambda sc: sc.total_score, reverse=True)
            rankings = []
            for idx, sc in enumerate(sorted_cards, start=1):
                rankings.append(
                    RankedCandidate(
                        job_id=input_data.job_id,
                        candidate_id=sc.candidate_id,
                        rank=idx,
                        percentile=100 - (idx - 1) * 10,
                        tier=Tier.TOP_10 if idx == 1 else Tier.TOP_25,
                        total_score=sc.total_score,
                        summary="mock ranking",
                    )
                )
            ranked_list = RankedList(
                job_id=input_data.job_id,
                rankings=rankings,
                total_candidates=len(rankings),
                tier_distribution={"top_10": 1},
                score_statistics={},
            )
            return AgentResult(success=True, data=ranked_list, metadata={"mock": True})

        prompt = self._build_prompt(input_data, context)
        output, metadata = await self._call_response_api(prompt, context)

        output.job_id = input_data.job_id

        return AgentResult(
            success=True,
            data=output,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: RankingInput, context: AgentContext) -> str:
        """Build ranking prompt."""
        scorecard_summary = "\n".join(
            f"- Candidate {i+1}: Score {sc.total_score:.1f}, Must-haves {sc.must_have_coverage:.1%}"
            for i, sc in enumerate(sorted(input_data.scorecards, key=lambda x: x.total_score, reverse=True))
        )

        return f"""Rank {len(input_data.scorecards)} candidates and assign tiers.

**CANDIDATES (pre-sorted by score):**
{scorecard_summary}

**TASKS:**

1. **Rank**: Assign ranks 1-{len(input_data.scorecards)} (1 = best)
2. **Calculate percentiles**: 0-100 based on rank
3. **Assign tiers**:
   - top_10: Top 10% (90th percentile+)
   - top_25: Top 25% (75th percentile+)
   - top_50: Top 50% (50th percentile+)
   - bottom_50: Bottom 50%

4. **Generate summaries**: 1-3 sentence rationale per candidate explaining:
   - Why they're in this tier
   - Top 2-3 strengths
   - Key concerns (if any)

5. **Compute statistics**:
   - Tier distribution (count per tier)
   - Score statistics (mean, median, std dev)

Return comprehensive ranked list."""
