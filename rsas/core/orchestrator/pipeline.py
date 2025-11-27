"""JobPipeline - orchestrates all agents for resume ranking."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.bias_check import BiasCheckAgent, BiasCheckInput, BiasReport
from ..agents.job_understanding import JobDescriptionInput, JobUnderstandingAgent
from ..agents.matching import MatchingAgent, MatchingInput, MatchReport
from ..agents.output import FormattedOutput, OutputAgent, OutputInput
from ..agents.parser import ParserAgent, ParsedResume, ResumeParseInput
from ..agents.ranking import RankingAgent, RankingInput
from ..agents.scoring import ScoringAgent, ScoringInput
from ..agents.skills_extraction import SkillsExtractionAgent, SkillsExtractionInput
from ..models.audit import PipelineState
from ..models.base import AgentContext, PipelineResult
from ..models.candidate_profile import CandidateProfile
from ..models.enums import JobStatus
from ..models.job_profile import EnrichedJobProfile
from ..models.ranked_candidate import RankedList
from ..models.scorecard import ScoreCard
from ..storage.object_store import ObjectStore
from ...observability.logger import get_logger

logger = get_logger(__name__)


class JobPipeline:
    """Orchestrates all agents to process resumes for a job."""

    def __init__(
        self,
        store: ObjectStore,
        context: AgentContext,
    ):
        self.store = store
        self.context = context

        # Initialize all agents
        self.job_understanding = JobUnderstandingAgent(store)
        self.parser = ParserAgent(store)
        self.skills_extraction = SkillsExtractionAgent(store)
        self.matching = MatchingAgent(store)
        self.scoring = ScoringAgent(store)
        self.ranking = RankingAgent(store)
        self.bias_check = BiasCheckAgent(store)
        self.output_agent = OutputAgent(store)

        # Get concurrency limit from config
        self.max_concurrent = context.config.get("pipeline", {}).get("max_concurrent_resumes", 10)
        self.checkpoint_interval = context.config.get("pipeline", {}).get("checkpoint_interval", 10)

    async def run(
        self,
        job_id: str,
        job_description: str,
        resume_dir: Path,
    ) -> PipelineResult:
        """Run complete pipeline for a job.

        Args:
            job_id: Unique job identifier
            job_description: Raw job description text
            resume_dir: Directory containing resume PDFs

        Returns:
            PipelineResult with final output and metadata
        """
        logger.info("pipeline_started", job_id=job_id, resume_dir=str(resume_dir))

        start_time = datetime.now(timezone.utc)

        try:
            # Create or load pipeline state
            state = await self._get_or_create_state(job_id)

            # Step 1: Job Understanding
            job_profile = await self._process_job_understanding(job_id, job_description, state)

            # Step 2-5: Process all resumes (parse → extract → match → score)
            scorecards = await self._process_all_resumes(
                job_id, job_profile, resume_dir, state
            )

            # Step 6: Ranking
            ranked_list = await self._process_ranking(job_id, scorecards, state)

            # Step 7: Bias Check
            bias_report = await self._process_bias_check(job_id, ranked_list, state)

            # Step 8: Output Generation
            final_output = await self._process_output(job_id, ranked_list, bias_report, state)

            # Mark pipeline as completed
            await self._update_state(
                state,
                status=JobStatus.COMPLETED,
                metadata={"completed_at": datetime.now(timezone.utc).isoformat()},
            )

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            logger.info(
                "pipeline_completed",
                job_id=job_id,
                duration_seconds=duration,
                total_candidates=len(scorecards),
            )

            return PipelineResult(
                success=True,
                output=final_output,
                stats={
                    "job_id": job_id,
                    "duration_seconds": duration,
                    "total_candidates": len(scorecards),
                    "pipeline_state": state.model_dump(mode="json"),
                },
            )

        except Exception as e:
            logger.exception("pipeline_failed", job_id=job_id, error=str(e))

            # Mark pipeline as failed
            if 'state' in locals():
                await self._update_state(
                    state,
                    status=JobStatus.FAILED,
                    metadata={"error": str(e), "failed_at": datetime.now(timezone.utc).isoformat()},
                )

            return PipelineResult(
                success=False,
                output=None,
                error=str(e),
                stats={"job_id": job_id, "failed_at": datetime.now(timezone.utc).isoformat()},
            )

    async def _get_or_create_state(self, job_id: str) -> PipelineState:
        """Get existing pipeline state or create new one."""
        existing = self.store.load_pipeline_state(job_id)
        if existing:
            return existing

        from ..models.enums import ProcessingStage

        state = PipelineState(
            job_id=job_id,
            status=JobStatus.PROCESSING,
            current_stage=ProcessingStage.PENDING,
            completed_resumes=[],
            failed_resumes=[],
            started_at=datetime.now(timezone.utc),
            last_checkpoint_at=None,
            metadata={},
        )
        self.store.save_pipeline_state(state)
        return state

    async def _update_state(
        self,
        state: PipelineState,
        status: JobStatus | None = None,
        current_stage: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update pipeline state in database."""
        if status:
            state.status = status
        if current_stage:
            from ..models.enums import ProcessingStage

            if isinstance(current_stage, str):
                try:
                    state.current_stage = ProcessingStage(current_stage)
                except Exception:
                    state.current_stage = current_stage  # type: ignore
            else:
                state.current_stage = current_stage
        if metadata:
            current_metadata = state.metadata or {}
            state.metadata = {**current_metadata, **metadata}
            state.last_checkpoint_at = datetime.now(timezone.utc)

        self.store.save_pipeline_state(state)

    async def _process_job_understanding(
        self,
        job_id: str,
        job_description: str,
        state: PipelineState,
    ) -> EnrichedJobProfile:
        """Process job description to extract requirements."""
        logger.info("stage_started", stage="job_understanding", job_id=job_id)

        from ..models.enums import ProcessingStage

        await self._update_state(state, current_stage=ProcessingStage.PARSING)

        # Extract title from first line of job description
        lines = job_description.strip().split('\n')
        title = lines[0] if lines else "Untitled Position"

        input_data = JobDescriptionInput(
            job_id=job_id,
            title=title,
            description=job_description,
        )

        result = await self.job_understanding.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Job understanding failed: {result.error}")

        # Store job profile
        self.store.save_job_profile(result.data)

        logger.info("stage_completed", stage="job_understanding", job_id=job_id)
        return result.data

    async def _process_all_resumes(
        self,
        job_id: str,
        job_profile: EnrichedJobProfile,
        resume_dir: Path,
        state: PipelineState,
    ) -> list[ScoreCard]:
        """Process all resumes through parse → extract → match → score pipeline."""
        # Find all PDF files
        pdf_files = list(resume_dir.glob("*.pdf"))
        total_resumes = len(pdf_files)

        logger.info(
            "resume_processing_started",
            job_id=job_id,
            total_resumes=total_resumes,
            resume_dir=str(resume_dir),
        )

        from ..models.enums import ProcessingStage

        await self._update_state(
            state,
            current_stage=ProcessingStage.PARSING,
            metadata={"total_resumes": total_resumes},
        )

        # Process resumes with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        scorecards: list[ScoreCard] = []
        processed_count = 0

        async def process_resume(pdf_path: Path, index: int) -> ScoreCard | None:
            nonlocal processed_count

            async with semaphore:
                try:
                    # Generate candidate ID from filename
                    candidate_id = f"{job_id}_{pdf_path.stem}"

                    # Step 2: Parse PDF
                    parsed = await self._parse_resume(job_id, candidate_id, pdf_path)

                    # Step 3: Extract skills
                    profile = await self._extract_skills(candidate_id, job_id, parsed, job_profile)

                    # Step 4: Match to job
                    match_report = await self._match_candidate(
                        candidate_id, job_id, profile, job_profile
                    )

                    # Step 5: Score candidate
                    scorecard = await self._score_candidate(
                        candidate_id, job_id, match_report, profile, job_profile
                    )

                    processed_count += 1
                    state.completed_resumes.append(candidate_id)

                    # Checkpoint every N resumes
                    if processed_count % self.checkpoint_interval == 0:
                        await self._update_state(
                            state,
                            metadata={
                                "last_checkpoint": datetime.now(timezone.utc).isoformat(),
                                "processed_count": processed_count,
                            },
                        )
                        logger.info(
                            "checkpoint_saved",
                            job_id=job_id,
                            processed=processed_count,
                            total=total_resumes,
                        )

                    await self._update_state(
                        state,
                        metadata={"processed_count": processed_count},
                    )

                    return scorecard

                except Exception as e:
                    logger.exception(
                        "resume_processing_failed",
                        job_id=job_id,
                        pdf_path=str(pdf_path),
                        error=str(e),
                    )
                    state.failed_resumes.append((str(pdf_path), str(e)))
                    await self._update_state(
                        state,
                        metadata={"processed_count": processed_count},
                    )
                    return None

        # Process all resumes concurrently
        results = await asyncio.gather(
            *[process_resume(pdf, i) for i, pdf in enumerate(pdf_files)],
            return_exceptions=False,
        )

        # Filter out None results (failed resumes)
        scorecards = [sc for sc in results if sc is not None]

        logger.info(
            "resume_processing_completed",
            job_id=job_id,
            total_resumes=total_resumes,
            successful=len(scorecards),
            failed=total_resumes - len(scorecards),
        )

        return scorecards

    async def _parse_resume(self, job_id: str, candidate_id: str, pdf_path: Path) -> ParsedResume:
        """Parse PDF resume."""
        input_data = ResumeParseInput(
            job_id=job_id,
            candidate_id=candidate_id,
            resume_path=str(pdf_path),
        )
        result = await self.parser.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Resume parsing failed: {result.error}")

        self.store.save_parsed_resume(result.data)
        return result.data

    async def _extract_skills(
        self,
        candidate_id: str,
        job_id: str,
        parsed_resume: ParsedResume,
        job_profile: EnrichedJobProfile,
    ) -> CandidateProfile:
        """Extract skills from parsed resume."""
        input_data = SkillsExtractionInput(
            candidate_id=candidate_id,
            job_id=job_id,
            parsed_resume=parsed_resume,
            job_profile=job_profile,
        )
        result = await self.skills_extraction.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Skills extraction failed: {result.error}")

        # Persist parsed + profile
        self.store.save_candidate_profile(result.data)

        return result.data

    async def _match_candidate(
        self,
        candidate_id: str,
        job_id: str,
        candidate_profile: CandidateProfile,
        job_profile: EnrichedJobProfile,
    ) -> MatchReport:
        """Match candidate to job requirements."""
        input_data = MatchingInput(
            candidate_id=candidate_id,
            job_id=job_id,
            candidate_profile=candidate_profile,
            job_profile=job_profile,
        )
        result = await self.matching.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Matching failed: {result.error}")

        return result.data

    async def _score_candidate(
        self,
        candidate_id: str,
        job_id: str,
        match_report: MatchReport,
        candidate_profile: CandidateProfile,
        job_profile: EnrichedJobProfile,
    ) -> ScoreCard:
        """Score candidate on multiple dimensions."""
        input_data = ScoringInput(
            candidate_id=candidate_id,
            job_id=job_id,
            match_report=match_report,
            candidate_profile=candidate_profile,
            job_profile=job_profile,
        )
        result = await self.scoring.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Scoring failed: {result.error}")

        self.store.save_scorecard(result.data)

        return result.data

    async def _process_ranking(
        self,
        job_id: str,
        scorecards: list[ScoreCard],
        state: PipelineState,
    ) -> RankedList:
        """Rank all candidates and assign tiers."""
        from ..models.enums import ProcessingStage

        logger.info("stage_started", stage="ranking", job_id=job_id, total_scorecards=len(scorecards))

        await self._update_state(state, current_stage=ProcessingStage.RANKING)

        input_data = RankingInput(
            job_id=job_id,
            scorecards=scorecards,
        )
        result = await self.ranking.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Ranking failed: {result.error}")

        self.store.save_rankings(result.data)
        logger.info("stage_completed", stage="ranking", job_id=job_id)
        return result.data

    async def _process_bias_check(
        self,
        job_id: str,
        ranked_list: RankedList,
        state: PipelineState,
    ) -> BiasReport:
        """Check for potential biases in rankings."""
        from ..models.enums import ProcessingStage

        logger.info("stage_started", stage="bias_check", job_id=job_id)

        await self._update_state(state, current_stage=ProcessingStage.BIAS_CHECKING)

        input_data = BiasCheckInput(
            job_id=job_id,
            ranked_list=ranked_list,
        )
        result = await self.bias_check.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Bias check failed: {result.error}")

        logger.info("stage_completed", stage="bias_check", job_id=job_id)
        return result.data

    async def _process_output(
        self,
        job_id: str,
        ranked_list: RankedList,
        bias_report: BiasReport,
        state: PipelineState,
    ) -> FormattedOutput:
        """Generate formatted outputs."""
        from ..models.enums import ProcessingStage

        logger.info("stage_started", stage="output", job_id=job_id)

        await self._update_state(state, current_stage=ProcessingStage.COMPLETED)

        input_data = OutputInput(
            job_id=job_id,
            ranked_list=ranked_list,
            bias_report=bias_report,
        )
        result = await self.output_agent.execute(input_data, self.context)

        if not result.success:
            raise RuntimeError(f"Output generation failed: {result.error}")

        # Persist final output for later inspection
        output_path = self.store._job_dir(job_id) / "output.json"
        output_path.write_text(result.data.model_dump_json(indent=2), encoding="utf-8")
        logger.info("stage_completed", stage="output", job_id=job_id)
        return result.data
