"""CLI interface for RSAS using Typer."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ..core.config.loader import load_config
from ..core.models.base import AgentContext
from ..core.orchestrator.pipeline import JobPipeline
from ..core.storage.object_store import ObjectStore
from ..search.service import SearchService
from ..observability.logger import get_logger
from ..interactive.assistant import run_chat_session

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="rsas",
    help="Resume Sorting Agent System - AI-powered resume ranking",
    add_completion=False,
)


def _get_store() -> ObjectStore:
    """Get file-based object store from config."""
    config = load_config()
    base_dir = config.get("storage", {}).get("object_store_dir", "data/processed")
    return ObjectStore(base_dir)


def _get_agent_context() -> AgentContext:
    """Get agent context from config."""
    config = load_config()
    return AgentContext(
        config=config,
        job_id="",  # Will be overridden
        agent_type="",  # Will be overridden
    )


@app.command()
def process(
    job_id: Annotated[str, typer.Option("--job-id", "-j", help="Unique job identifier")],
    job_description: Annotated[
        Path,
        typer.Option(
            "--description",
            "-d",
            help="Path to job description file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    resumes_dir: Annotated[
        Path,
        typer.Option(
            "--resumes",
            "-r",
            help="Directory containing resume PDFs",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path to save output JSON"),
    ] = None,
    limit_resumes: Annotated[
        int | None,
        typer.Option(
            "--limit-resumes",
            "-l",
            help="Max number of resumes to process (for cost control). Processes all if not set.",
        ),
    ] = None,
):
    """Process resumes for a job opening.

    Runs the complete pipeline:
    1. Understand job requirements
    2. Parse all resumes
    3. Extract skills
    4. Match candidates to job
    5. Score candidates
    6. Rank candidates
    7. Check for biases
    8. Generate formatted output
    """
    console.print(f"\n[bold blue]Starting RSAS pipeline for job:[/bold blue] {job_id}")
    console.print(f"[dim]Job description:[/dim] {job_description}")
    console.print(f"[dim]Resumes directory:[/dim] {resumes_dir}")

    # Validate inputs
    if limit_resumes is not None and limit_resumes <= 0:
        console.print("[red]! Error:[/red] --limit-resumes must be greater than 0")
        raise typer.Exit(code=1)

    # Check that resume directory is not empty
    pdf_files = list(resumes_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]! Error:[/red] No PDF files found in {resumes_dir}")
        raise typer.Exit(code=1)

    console.print(f"[dim]Found {len(pdf_files)} PDF files[/dim]")

    # Load job description with error handling
    try:
        jd_text = job_description.read_text(encoding="utf-8")
    except (IOError, UnicodeDecodeError) as e:
        console.print(f"[red]! Error reading job description:[/red] {e}")
        raise typer.Exit(code=1)

    # Initialize components
    store = _get_store()
    context = _get_agent_context()
    context.job_id = job_id

    # Run pipeline
    async def run_pipeline():
        pipeline = JobPipeline(store, context)
        result = await pipeline.run(job_id, jd_text, resumes_dir, resume_limit=limit_resumes)

        return result

    # Execute async pipeline
    result = asyncio.run(run_pipeline())

    # Display results
    if result.success:
        console.print("\n[green]> Pipeline completed successfully![/green]")

        # Display summary
        output = result.output
        console.print(f"\n[bold]Executive Summary:[/bold]")
        console.print(output.executive_summary)

        # Display top candidates
        console.print(f"\n[bold]Top {len(output.top_candidates_summary)} Candidates:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Name")
        table.add_column("Score", justify="right")
        table.add_column("Summary")

        for candidate in output.top_candidates_summary:
            # Cache summary to avoid multiple lookups
            summary = candidate.get("summary", "")
            summary_display = summary[:60] + "..." if len(summary) > 60 else summary

            table.add_row(
                str(candidate.get("rank", "N/A")),
                candidate.get("name", "Unknown"),
                str(candidate.get("score", "N/A")),
                summary_display,
            )

        console.print(table)

        # Save output if requested
        if output_file:
            output_data = {
                "job_id": job_id,
                "executive_summary": output.executive_summary,
                "top_candidates": output.top_candidates_summary,
                "full_report": output.full_report_json,
                "recruiter_notes": output.recruiter_notes,
                "stats": result.stats,
            }
            try:
                # Create parent directory if it doesn't exist
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
                console.print(f"\n[green]Output saved to:[/green] {output_file}")
            except (IOError, TypeError) as e:
                console.print(f"\n[red]! Error saving output:[/red] {e}")

    else:
        console.print(f"\n[red]! Pipeline failed:[/red] {result.error}")
        raise typer.Exit(code=1)


@app.command()
def process_agents(
    job_id: Annotated[str, typer.Option("--job-id", "-j", help="Unique job identifier")],
    job_description: Annotated[
        Path,
        typer.Option(
            "--description",
            "-d",
            help="Path to job description file",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ],
    resumes_dir: Annotated[
        Path,
        typer.Option(
            "--resumes",
            "-r",
            help="Directory containing resume PDFs",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
):
    """Run pipeline via Agents SDK orchestrator."""
    from ..agents_sdk.pipeline_agent import run_with_agents_sdk

    store = _get_store()
    jd_text = job_description.read_text(encoding="utf-8")

    async def run():
        return await run_with_agents_sdk(job_id, jd_text, str(resumes_dir), store)

    result = asyncio.run(run())

    console.print(result)


@app.command()
def status(
    job_id: Annotated[str, typer.Option("--job-id", "-j", help="Job identifier")],
):
    """Check status of a job pipeline."""
    console.print(f"\n[bold blue]Checking status for job:[/bold blue] {job_id}")

    store = _get_store()
    state = store.load_pipeline_state(job_id)

    if not state:
        console.print(f"[yellow]No pipeline state found for job:[/yellow] {job_id}")
        return

    # Display status
    table = Table(show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    total_resumes = state.metadata.get("total_resumes") if state.metadata else None
    processed = state.metadata.get("processed_count") if state.metadata else None

    table.add_row("Job ID", state.job_id)
    table.add_row("Status", f"[{'green' if state.status == 'completed' else 'yellow'}]{state.status.value if hasattr(state.status, 'value') else state.status}[/]")
    table.add_row("Current Stage", state.current_stage if isinstance(state.current_stage, str) else state.current_stage.value)
    if total_resumes is not None and processed is not None:
        table.add_row("Progress", f"{processed}/{total_resumes}")
    table.add_row("Started", str(state.created_at))

    console.print(table)


@app.command()
def ranking(
    job_id: Annotated[str, typer.Option("--job-id", "-j", help="Job identifier")],
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of top candidates to show")] = 10,
    export: Annotated[
        Path | None,
        typer.Option("--export", "-e", help="Export rankings to CSV"),
    ] = None,
):
    """Display candidate rankings for a job."""
    # Validate inputs
    if top_n <= 0:
        console.print("[red]! Error:[/red] --top-n must be greater than 0")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]Rankings for job:[/bold blue] {job_id}")

    store = _get_store()
    ranking = store.load_rankings(job_id)

    if ranking:
        display_list = ranking.rankings[:top_n]
    else:
        scorecards = store.list_scorecards(job_id)
        if not scorecards:
            console.print(f"[yellow]No scorecards found for job:[/yellow] {job_id}")
            return
        scorecards = sorted(scorecards, key=lambda sc: sc.total_score, reverse=True)[:top_n]
        display_list = []
        for idx, sc in enumerate(scorecards, start=1):
            display_list.append(
                {
                    "rank": idx,
                    "candidate_id": sc.candidate_id,
                    "total_score": sc.total_score,
                    "must_have_coverage": sc.must_have_coverage,
                    "confidence": sc.confidence,
                }
            )

    # Display rankings
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Candidate ID")
    table.add_column("Total Score", justify="right")
    table.add_column("Must-Have", justify="right")
    table.add_column("Confidence", justify="right")

    for rank, entry in enumerate(display_list, start=1):
        if hasattr(entry, "candidate_id"):
            cand_id = entry.candidate_id
            total_score = entry.total_score
            must_have = entry.must_have_coverage if hasattr(entry, "must_have_coverage") else None
            confidence = entry.confidence if hasattr(entry, "confidence") else None
        else:
            cand_id = entry.get("candidate_id")
            total_score = entry.get("total_score", 0)
            must_have = entry.get("must_have_coverage", 0)
            confidence = entry.get("confidence", 0)

        table.add_row(
            str(entry.rank if hasattr(entry, "rank") else entry.get("rank", rank)),
            cand_id,
            f"{total_score:.1f}",
            f"{must_have:.1%}" if must_have is not None else "N/A",
            f"{confidence:.2f}" if confidence is not None else "N/A",
        )

    console.print(table)

    # Export if requested
    if export:
        import csv

        try:
            # Create parent directory if it doesn't exist
            export.parent.mkdir(parents=True, exist_ok=True)

            with export.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Rank", "Candidate ID", "Total Score", "Must-Have Coverage", "Confidence"])

                for rank, entry in enumerate(display_list, start=1):
                    if hasattr(entry, "candidate_id"):
                        cand_id = entry.candidate_id
                        total_score = entry.total_score
                        must_have = entry.must_have_coverage if hasattr(entry, "must_have_coverage") else 0
                        confidence = entry.confidence if hasattr(entry, "confidence") else 0
                    else:
                        cand_id = entry.get("candidate_id")
                        total_score = entry.get("total_score", 0)
                        must_have = entry.get("must_have_coverage", 0)
                        confidence = entry.get("confidence", 0)

                    writer.writerow([
                        rank,
                        cand_id,
                        f"{total_score:.1f}",
                        f"{must_have:.1%}",
                        f"{confidence:.2f}",
                    ])

            console.print(f"\n[green]Rankings exported to:[/green] {export}")
        except (IOError, ValueError) as e:
            console.print(f"\n[red]! Error exporting rankings:[/red] {e}")


@app.command()
def init_db():
    """Initialize the object store."""
    console.print("[bold blue]Initializing object store...[/bold blue]")
    store = _get_store()
    store.base_dir.mkdir(parents=True, exist_ok=True)
    console.print("[green]Object store ready at[/green] ", store.base_dir)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query text")],
    job_id: Annotated[str | None, typer.Option("--job-id", "-j", help="Job filter")] = None,
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of results")] = 10,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank", help="Use GPT rerank")] = False,
):
    """Semantic search over ingested candidates."""
    store = _get_store()

    # Try to init Chroma if available
    chroma_client = None
    try:
        from ..kb.storage.chroma_client import get_chroma_client

        chroma_client = get_chroma_client(mode="memory")
    except Exception as e:
        logger.warning("chroma_init_failed", error=str(e))
        chroma_client = None

    service = SearchService(store=store, chroma_client=chroma_client)

    async def run_search():
        return await service.search(query=query, job_id=job_id, top_k=top_n, rerank=rerank)

    results = asyncio.run(run_search())

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Candidate ID")
    table.add_column("Job ID")
    table.add_column("Score", justify="right")
    table.add_column("Summary")

    for idx, res in enumerate(results, start=1):
        summary = (res.summary_text or "")[:60]
        table.add_row(str(idx), res.candidate_id, res.job_id, f"{res.score:.3f}", summary)

    console.print(table)


@app.command()
def chat(
    job_id: Annotated[str, typer.Option("--job-id", "-j", help="Job identifier")],
    prompt: Annotated[str | None, typer.Option("--prompt", "-p", help="Single-turn question (omit for interactive mode)")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Override model for chat completions")] = None,
):
    """Interactive agentic chat to answer questions about candidates."""
    console.print(
        "[dim]Launching RSAS Candidate Navigator. Set OPENAI_API_KEY for live answers; "
        "offline mode will use stored data only.[/dim]"
    )
    run_chat_session(job_id=job_id, model=model, prompt=prompt)


if __name__ == "__main__":
    app()
