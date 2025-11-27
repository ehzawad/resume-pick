"""File-based object store for jobs, candidates, and pipeline artifacts.

This replaces the previous SQL-backed storage with simple JSON files plus
directories per job. It is intentionally lightweight to keep the stack
SQL-free while still persisting key artifacts and traces.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models.audit import AgentTrace, PipelineState
from ..models.job_profile import EnrichedJobProfile
from ..models.candidate_profile import CandidateProfile, ParsedResume
from ..models.scorecard import ScoreCard
from ..models.ranked_candidate import RankedList


class ObjectStore:
    """Simple JSON-backed persistence layer."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path("data/processed")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        path = self.base_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dump(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Job profile
    # ------------------------------------------------------------------
    def save_job_profile(self, job_profile: EnrichedJobProfile) -> None:
        path = self._job_dir(job_profile.job_id) / "job_profile.json"
        self._dump(path, job_profile.model_dump(mode="json"))

    def load_job_profile(self, job_id: str) -> EnrichedJobProfile | None:
        path = self._job_dir(job_id) / "job_profile.json"
        data = self._load(path)
        return EnrichedJobProfile(**data) if data else None

    # ------------------------------------------------------------------
    # Pipeline state
    # ------------------------------------------------------------------
    def save_pipeline_state(self, state: PipelineState) -> None:
        path = self._job_dir(state.job_id) / "pipeline_state.json"
        self._dump(path, state.model_dump(mode="json"))

    def load_pipeline_state(self, job_id: str) -> PipelineState | None:
        path = self._job_dir(job_id) / "pipeline_state.json"
        data = self._load(path)
        return PipelineState(**data) if data else None

    # ------------------------------------------------------------------
    # Parsed resumes and candidate profiles
    # ------------------------------------------------------------------
    def save_parsed_resume(self, parsed: ParsedResume) -> None:
        path = self._job_dir(parsed.job_id) / "parsed_resumes" / f"{parsed.candidate_id}.json"
        self._dump(path, parsed.model_dump(mode="json"))

    def load_parsed_resume(self, job_id: str, candidate_id: str) -> ParsedResume | None:
        path = self._job_dir(job_id) / "parsed_resumes" / f"{candidate_id}.json"
        data = self._load(path)
        return ParsedResume(**data) if data else None

    def save_candidate_profile(self, profile: CandidateProfile) -> None:
        path = self._job_dir(profile.job_id) / "candidate_profiles" / f"{profile.candidate_id}.json"
        self._dump(path, profile.model_dump(mode="json"))

    def load_candidate_profile(self, job_id: str, candidate_id: str) -> CandidateProfile | None:
        path = self._job_dir(job_id) / "candidate_profiles" / f"{candidate_id}.json"
        data = self._load(path)
        return CandidateProfile(**data) if data else None

    def list_candidate_ids(self, job_id: str) -> list[str]:
        profiles_dir = self._job_dir(job_id) / "candidate_profiles"
        if not profiles_dir.exists():
            return []
        return [p.stem for p in profiles_dir.glob("*.json")]

    # ------------------------------------------------------------------
    # Scorecards and rankings
    # ------------------------------------------------------------------
    def save_scorecard(self, scorecard: ScoreCard) -> None:
        path = self._job_dir(scorecard.job_id) / "scorecards" / f"{scorecard.candidate_id}.json"
        self._dump(path, scorecard.model_dump(mode="json"))

    def load_scorecard(self, job_id: str, candidate_id: str) -> ScoreCard | None:
        path = self._job_dir(job_id) / "scorecards" / f"{candidate_id}.json"
        data = self._load(path)
        return ScoreCard(**data) if data else None

    def list_scorecards(self, job_id: str) -> list[ScoreCard]:
        score_dir = self._job_dir(job_id) / "scorecards"
        if not score_dir.exists():
            return []
        cards: list[ScoreCard] = []
        for path in score_dir.glob("*.json"):
            data = self._load(path)
            if data:
                cards.append(ScoreCard(**data))
        return cards

    def save_rankings(self, ranked_list: RankedList) -> None:
        path = self._job_dir(ranked_list.job_id) / "ranking.json"
        self._dump(path, ranked_list.model_dump(mode="json"))

    def load_rankings(self, job_id: str) -> RankedList | None:
        path = self._job_dir(job_id) / "ranking.json"
        data = self._load(path)
        return RankedList(**data) if data else None

    # ------------------------------------------------------------------
    # Agent traces for idempotency
    # ------------------------------------------------------------------
    def get_agent_trace(self, job_id: str, agent_type: str, input_hash: str) -> dict[str, Any] | None:
        path = self._job_dir(job_id) / "traces" / agent_type / f"{input_hash}.json"
        return self._load(path)

    def save_agent_trace(
        self,
        job_id: str,
        agent_type: str,
        input_hash: str,
        trace: AgentTrace,
    ) -> None:
        path = self._job_dir(job_id) / "traces" / agent_type / f"{input_hash}.json"
        self._dump(path, trace.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # KB metadata (summary, embedding info, indexed fields)
    # ------------------------------------------------------------------
    def save_kb_record(self, job_id: str, candidate_id: str, record: dict[str, Any]) -> None:
        path = self._job_dir(job_id) / "kb" / f"{candidate_id}.json"
        self._dump(path, record)

    def load_kb_record(self, job_id: str, candidate_id: str) -> dict[str, Any] | None:
        path = self._job_dir(job_id) / "kb" / f"{candidate_id}.json"
        return self._load(path)

    def list_kb_records(self, job_id: str | None = None) -> list[tuple[str, str, dict[str, Any]]]:
        records: list[tuple[str, str, dict[str, Any]]] = []
        job_dirs = [self._job_dir(job_id)] if job_id else list(self.base_dir.glob("*"))
        for job_dir in job_dirs:
            job_id_val = job_dir.name
            kb_dir = job_dir / "kb"
            if not kb_dir.exists():
                continue
            for path in kb_dir.glob("*.json"):
                data = self._load(path)
                if data:
                    records.append((job_id_val, path.stem, data))
        return records
