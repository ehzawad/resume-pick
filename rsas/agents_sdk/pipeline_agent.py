"""Agents SDK orchestration for RSAS stages (OpenAI Agents Python)."""

from __future__ import annotations

import os
from typing import Any

try:
    from agents import Agent, Runner, function_tool
except ImportError:
    # Lightweight stubs so tests can run without the Agents SDK installed
    def function_tool(func):
        return func

    class Agent:
        def __init__(self, name: str, instructions: str, model: str, tools: list):
            self.name = name
            self.instructions = instructions
            self.model = model
            self.tools = tools

    class Runner:
        @staticmethod
        async def run(agent: Agent, input: str):
            raise RuntimeError("Agents SDK not installed")
from pathlib import Path

from ..core.storage.object_store import ObjectStore
from ..core.orchestrator.pipeline import JobPipeline
from ..core.config.loader import load_config
from ..core.models.base import AgentContext


def _make_stage_tool(store: ObjectStore, ctx: AgentContext, stage: str):
    @function_tool
    async def run_pipeline(job_id: str, job_description: str | None = None, resumes_dir: str | None = None):
        pipeline = JobPipeline(store, ctx)
        ctx.job_id = job_id
        if stage == "full":
            return await pipeline.run(job_id, job_description or "", resumes_dir)  # type: ignore
        raise ValueError("Unknown stage")

    run_pipeline.__doc__ = "Run the full RSAS pipeline"
    return run_pipeline


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
        model=cfg.get("openai", {}).get("model", "gpt-5.1"),
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
        pipeline = JobPipeline(store, ctx)
        return await pipeline.run(job_id=job_id, job_description=job_description, resume_dir=Path(resumes_dir))

    orchestrator = build_rsas_orchestrator(store)
    user_prompt = (
        "Run the RSAS pipeline by calling tool run_pipeline with the provided values.\n"
        f"job_id: {job_id}\n"
        f"resumes_dir: {resumes_dir}\n"
        "job_description:\n"
        f"{job_description}"
    )
    result = await Runner.run(orchestrator, input=user_prompt)
    return result.final_output
