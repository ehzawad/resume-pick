"""Agents SDK orchestration for RSAS stages."""

from __future__ import annotations

import os
from typing import Any

from openai_agents import Agent, Runner

from ..core.storage.object_store import ObjectStore
from ..core.orchestrator.pipeline import JobPipeline
from ..core.config.loader import load_config
from ..core.models.base import AgentContext


def _make_stage_tool(store: ObjectStore, ctx: AgentContext, stage: str):
    async def tool(job_id: str, job_description: str | None = None, resumes_dir: str | None = None):
        pipeline = JobPipeline(store, ctx)
        ctx.job_id = job_id
        if stage == "full":
            return await pipeline.run(job_id, job_description or "", resumes_dir)  # type: ignore
        raise ValueError("Unknown stage")

    tool.__name__ = "run_pipeline"
    tool.__doc__ = "Run the full RSAS pipeline"
    return tool


def build_rsas_orchestrator(store: ObjectStore, config: dict[str, Any] | None = None) -> Agent:
    """Create an Agents SDK orchestrator that runs the full pipeline via a tool."""
    cfg = config or load_config()
    ctx = AgentContext(job_id="", config=cfg)

    full_tool = _make_stage_tool(store, ctx, "full")

    orchestrator = Agent(
        name="rsas_orchestrator",
        instructions=(
            "You run the RSAS resume pipeline. "
            "Call the provided tool `run_pipeline` with job_id, job_description, and resumes_dir. "
            "Return the pipeline result as JSON."
        ),
        tools=[
            full_tool,
        ],
    )

    return orchestrator


async def run_with_agents_sdk(job_id: str, job_description: str, resumes_dir: str, store: ObjectStore):
    """Entry point to run the pipeline via Agents SDK orchestrator."""
    # In test mode or missing key, bypass LLM and call tool directly.
    if os.getenv("RSAS_TEST_MODE") or not os.getenv("OPENAI_API_KEY"):
        ctx = AgentContext(job_id=job_id, config=load_config())
        tool = _make_stage_tool(store, ctx, "full")
        return await tool(job_id=job_id, job_description=job_description, resumes_dir=resumes_dir)

    orchestrator = build_rsas_orchestrator(store)
    result = await Runner.run(
        orchestrator,
        {
            "job_id": job_id,
            "job_description": job_description,
            "resumes_dir": resumes_dir,
        },
    )
    return result.final_output
