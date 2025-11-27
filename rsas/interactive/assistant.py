"""Interactive candidate Q&A assistant with tool-calling.

This module exposes a small, agentic helper that can:
- List top candidates for a job
- Fetch detailed profile/scorecard context for a candidate
- Run semantic search across candidates (if Chroma is available)

It uses OpenAI chat completions with tool-calling for simplicity, and
falls back to a deterministic offline mode when RSAS_TEST_MODE is set
or no API key is present.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable

from openai import OpenAI

from ..core.config.loader import load_config
from ..core.models.scorecard import ScoreCard
from ..core.storage.object_store import ObjectStore
from ..observability.logger import get_logger
from ..search.service import SearchService

logger = get_logger(__name__)


@dataclass
class CandidateOverview:
    candidate_id: str
    name: str | None
    rank: int | None
    total_score: float | None
    must_have_coverage: float | None
    summary: str | None
    standout_areas: list[str]
    red_flags: list[dict[str, Any]]
    links: dict[str, str | None]


class CandidateAssistant:
    """Agentic helper to answer questions about candidates."""

    def __init__(self, store: ObjectStore, job_id: str, model: str | None = None):
        self.store = store
        self.job_id = job_id
        cfg = load_config()
        self.model = model or cfg.get("openai", {}).get("model", "gpt-5.1")
        self.client: OpenAI | None = None

        # Respect offline/test mode
        self.test_mode = bool(os.getenv("RSAS_TEST_MODE") or not os.getenv("OPENAI_API_KEY"))
        if not self.test_mode:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Preload ranking for fast lookup
        self.ranking = self.store.load_rankings(job_id)

        # Optional semantic search
        self.search_service = self._init_search_service()

    # ------------------------------------------------------------------ #
    # Tool dispatchers
    # ------------------------------------------------------------------ #
    def _build_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_top_candidates",
                    "description": "List top candidates for a job with scores and rank.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of candidates to return (default 5)",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_candidate_overview",
                    "description": "Fetch profile, scorecard, and ranking context for a specific candidate.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                        },
                        "required": ["candidate_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_candidates",
                    "description": "Semantic search across stored candidates to find relevant profiles.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_n": {
                                "type": "integer",
                                "description": "How many matches to return (default 5)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "list_top_candidates":
                top_n = int(arguments.get("top_n") or 5)
                return json.dumps(self._list_top_candidates(top_n), ensure_ascii=False)
            if name == "get_candidate_overview":
                return json.dumps(
                    asdict(self._candidate_overview(arguments["candidate_id"])),
                    ensure_ascii=False,
                )
            if name == "search_candidates":
                return json.dumps(
                    self._search(arguments["query"], int(arguments.get("top_n") or 5)),
                    ensure_ascii=False,
                )
        except Exception as exc:  # Defensive: never crash tool calling
            logger.exception("tool_call_failed", tool=name, error=str(exc))
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"Unknown tool {name}"})

    # ------------------------------------------------------------------ #
    # Data access helpers
    # ------------------------------------------------------------------ #
    def _list_top_candidates(self, top_n: int) -> list[dict[str, Any]]:
        if self.ranking:
            data = self.ranking.rankings[:top_n]
            result: list[dict[str, Any]] = []
            for entry in data:
                scorecard = self.store.load_scorecard(self.job_id, entry.candidate_id)
                result.append(
                    {
                        "rank": entry.rank,
                        "candidate_id": entry.candidate_id,
                        "total_score": entry.total_score,
                        "must_have_coverage": scorecard.must_have_coverage if scorecard else None,
                        "confidence": scorecard.confidence if scorecard else None,
                        "tier": entry.tier,
                        "summary": entry.summary,
                    }
                )
            return result

        # Fallback to scorecards if ranking not present
        cards = self.store.list_scorecards(self.job_id)
        cards = sorted(cards, key=lambda sc: sc.total_score, reverse=True)[:top_n]
        return [
            {
                "rank": idx + 1,
                "candidate_id": card.candidate_id,
                "total_score": card.total_score,
                "must_have_coverage": card.must_have_coverage,
                "tier": None,
                "summary": card.summary,
            }
            for idx, card in enumerate(cards)
        ]

    def _candidate_overview(self, candidate_id: str) -> CandidateOverview:
        profile = self.store.load_candidate_profile(self.job_id, candidate_id)
        parsed = self.store.load_parsed_resume(self.job_id, candidate_id)
        scorecard: ScoreCard | None = self.store.load_scorecard(self.job_id, candidate_id)

        # Determine rank if available
        rank = None
        if self.ranking:
            for entry in self.ranking.rankings:
                if entry.candidate_id == candidate_id:
                    rank = entry.rank
                    break

        links = {
            "linkedin": getattr(parsed.contact, "linkedin", None) if parsed else None,
            "github": getattr(parsed.contact, "github", None) if parsed else None,
            "portfolio": getattr(parsed.contact, "portfolio", None) if parsed else None,
        }

        return CandidateOverview(
            candidate_id=candidate_id,
            name=getattr(parsed.contact, "name", None) if parsed else None,
            rank=rank,
            total_score=scorecard.total_score if scorecard else None,
            must_have_coverage=scorecard.must_have_coverage if scorecard else None,
            summary=scorecard.summary if scorecard else None,
            standout_areas=scorecard.standout_areas if scorecard else [],
            red_flags=[rf.model_dump(mode="json") for rf in scorecard.red_flags] if scorecard else [],
            links=links,
        )

    def _search(self, query: str, top_n: int) -> list[dict[str, Any]]:
        if not self.search_service:
            return [{"info": "Semantic search not configured; enable Chroma to use this tool."}]

        async def _run():
            return await self.search_service.search(query=query, job_id=self.job_id, top_k=top_n)

        results = asyncio.run(_run())
        payload = []
        for res in results:
            payload.append(
                {
                    "candidate_id": res.candidate_id,
                    "job_id": res.job_id,
                    "score": res.score,
                    "summary": res.summary_text,
                }
            )
        return payload

    def _init_search_service(self) -> SearchService | None:
        try:
            from ..kb.storage.chroma_client import get_chroma_client

            chroma = get_chroma_client(mode="memory")
        except Exception:
            chroma = None

        try:
            return SearchService(store=self.store, chroma_client=chroma)
        except Exception as exc:
            logger.warning("search_service_init_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------ #
    # Conversation loop
    # ------------------------------------------------------------------ #
    def chat_once(self, user_input: str, history: list[dict[str, Any]] | None = None) -> str:
        """Handle a single turn, returning assistant text."""
        history = history or []

        # Offline/test fallback: deterministic answer using local data only
        if self.test_mode or not self.client:
            return self._offline_answer(user_input)

        messages = history + [
            {
                "role": "system",
                "content": (
                    "You are RSAS Candidate Navigator. "
                    "Answer questions about candidates using the tools. "
                    "Keep answers concise and cite evidence from scorecards when relevant."
                ),
            },
            {"role": "user", "content": user_input},
        ]

        tools = self._build_tools()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        message = response.choices[0].message
        if not message.tool_calls:
            return message.content or "No answer generated."

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in message.tool_calls],
            }
        )

        # Execute tool calls
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments or "{}")
            result = self._dispatch_tool(tool_call.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # Ask model to respond with tool outputs
        follow_up = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        return follow_up.choices[0].message.content or "No answer generated."

    def _offline_answer(self, user_input: str) -> str:
        """Simple deterministic answer that does not hit the API."""
        top = self._list_top_candidates(top_n=3)
        lead = top[0] if top else None
        parts = [f"(offline) I cannot call the model, but here is stored context for job {self.job_id}."]
        if lead:
            parts.append(
                f"Top candidate: {lead.get('candidate_id')} "
                f"with score {lead.get('total_score')} and coverage {lead.get('must_have_coverage')}."
            )
        parts.append("Ask again with OPENAI_API_KEY set to enable interactive answers.")
        return " ".join(parts)


def run_chat_session(job_id: str, model: str | None = None, prompt: str | None = None) -> None:
    """Launch an interactive or one-off chat session."""
    store = ObjectStore(load_config().get("storage", {}).get("object_store_dir", "data/processed"))
    assistant = CandidateAssistant(store=store, job_id=job_id, model=model)

    console_print: Callable[[str], None]
    try:
        from rich.console import Console

        console = Console()

        def console_print(msg: str) -> None:
            console.print(msg)

    except Exception:
        console_print = print

    if prompt:
        answer = assistant.chat_once(prompt)
        console_print(f"\n[bold]Assistant:[/bold] {answer}")
        return

    console_print("[bold blue]RSAS Candidate Navigator[/bold blue] (type 'exit' to quit)")
    history: list[dict[str, Any]] = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            console_print("\n[dim]Exiting chat.[/dim]")
            break
        if user_input.lower() in {"exit", "quit"}:
            console_print("[dim]Goodbye![/dim]")
            break

        answer = assistant.chat_once(user_input, history=history)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        console_print(f"[bold]Assistant:[/bold] {answer}")
