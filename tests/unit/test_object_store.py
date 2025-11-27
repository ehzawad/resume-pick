"""Sanity checks for the SQL-free object store."""

from datetime import datetime, timezone

from rsas.core.storage.object_store import ObjectStore
from rsas.core.models.audit import AgentTrace, PipelineState
from rsas.core.models.enums import AgentType, JobStatus, ProcessingStage, Tier
from rsas.core.models.job_profile import EnrichedJobProfile
from rsas.core.models.candidate_profile import CandidateProfile, ParsedResume
from rsas.core.models.scorecard import ScoreCard, ScoreDimensions
from rsas.core.models.ranked_candidate import RankedCandidate, RankedList


def test_object_store_roundtrip(tmp_path):
    store = ObjectStore(tmp_path / "store")

    # Job profile
    job_profile = EnrichedJobProfile(
        job_id="job1",
        title="Test Role",
        raw_description="A test job",
    )
    store.save_job_profile(job_profile)
    loaded_job = store.load_job_profile("job1")
    assert loaded_job and loaded_job.title == "Test Role"

    # Pipeline state
    state = PipelineState(
        job_id="job1",
        status=JobStatus.PROCESSING,
        current_stage=ProcessingStage.PARSING,
        completed_resumes=[],
        failed_resumes=[],
        started_at=datetime.now(timezone.utc),
    )
    store.save_pipeline_state(state)
    loaded_state = store.load_pipeline_state("job1")
    assert loaded_state and loaded_state.current_stage == ProcessingStage.PARSING

    # Parsed resume + candidate profile
    parsed = ParsedResume(candidate_id="cand1", job_id="job1")
    profile = CandidateProfile(candidate_id="cand1", job_id="job1", resume_id="resume1")
    store.save_parsed_resume(parsed)
    store.save_candidate_profile(profile)
    assert store.load_parsed_resume("job1", "cand1")
    assert store.load_candidate_profile("job1", "cand1")
    assert "cand1" in store.list_candidate_ids("job1")

    # Scorecard
    scorecard = ScoreCard(
        candidate_id="cand1",
        job_id="job1",
        dimensions=ScoreDimensions(),
        total_score=88.0,
        must_have_coverage=0.8,
        confidence=0.9,
    )
    store.save_scorecard(scorecard)
    loaded_sc = store.load_scorecard("job1", "cand1")
    assert loaded_sc and loaded_sc.total_score == 88.0
    assert store.list_scorecards("job1")

    # Rankings
    ranked = RankedCandidate(
        job_id="job1",
        candidate_id="cand1",
        rank=1,
        percentile=100,
        tier=Tier.TOP_10,
        total_score=88.0,
    )
    ranked_list = RankedList(job_id="job1", rankings=[ranked], total_candidates=1)
    store.save_rankings(ranked_list)
    loaded_rankings = store.load_rankings("job1")
    assert loaded_rankings and loaded_rankings.rankings[0].candidate_id == "cand1"

    # Agent trace
    trace = AgentTrace(
        job_id="job1",
        agent_type=AgentType.PARSER,
        input_hash="abc123",
        output_data={"ok": True},
    )
    store.save_agent_trace("job1", AgentType.PARSER.value, "abc123", trace)
    cached = store.get_agent_trace("job1", AgentType.PARSER.value, "abc123")
    assert cached and cached["output_data"]["ok"] is True

    # KB record
    store.save_kb_record("job1", "cand1", {"summary_text": "hello", "embedding_id": "emb1"})
    kb = store.load_kb_record("job1", "cand1")
    assert kb and kb["embedding_id"] == "emb1"
    assert any(rec[1] == "cand1" for rec in store.list_kb_records("job1"))
