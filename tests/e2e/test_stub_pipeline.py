"""End-to-end smoke test with mocked OpenAI and file-backed storage."""

import asyncio
from pathlib import Path

from rsas.core.config.loader import load_config
from rsas.core.models.base import AgentContext
from rsas.core.orchestrator.pipeline import JobPipeline
from rsas.core.storage.object_store import ObjectStore


def test_pipeline_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("RSAS_TEST_MODE", "1")

    store = ObjectStore(tmp_path / "store")
    context = AgentContext(job_id="job-test", config=load_config())

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    (resumes_dir / "candidate.pdf").write_bytes(b"%PDF-1.4 mock")

    pipeline = JobPipeline(store, context)

    async def run():
        return await pipeline.run(
            job_id="job-test",
            job_description="Test role\nResponsibilities: testing things.",
            resume_dir=resumes_dir,
        )

    result = asyncio.run(run())
    assert result.success

    job_dir = store._job_dir("job-test")
    assert (job_dir / "ranking.json").exists()
    assert (job_dir / "output.json").exists()
