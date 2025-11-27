import asyncio
from pathlib import Path

import pytest

try:
    from rsas.agents_sdk.pipeline_agent import run_with_agents_sdk
except ImportError:
    run_with_agents_sdk = None

from rsas.core.storage.object_store import ObjectStore


@pytest.mark.skipif(run_with_agents_sdk is None, reason="agents SDK not installed")
def test_agents_sdk_stub(tmp_path, monkeypatch):
    monkeypatch.setenv("RSAS_TEST_MODE", "1")
    store = ObjectStore(tmp_path / "store")

    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    (resumes_dir / "candidate.pdf").write_bytes(b"%PDF-1.4 mock")

    jd_text = "Test role\nResponsibilities: testing things."

    async def run():
        return await run_with_agents_sdk("job-agents", jd_text, str(resumes_dir), store)  # type: ignore

    result = asyncio.run(run())
    assert result.success
