"""Knowledge Base ingestion pipeline.

Orchestrates the complete ingestion process:
1. Load candidates from database
2. Generate summaries (gpt-4o-mini)
3. Generate embeddings (text-embedding-3-small)
4. Index metadata for Tier 1 filtering
5. Store in ChromaDB for Tier 2 semantic search
6. Update database with enriched data

Cost estimate for 346 resumes:
- Summaries: 346 × $0.001 = $0.35
- Embeddings: 346 × $0.000006 = $0.002
- Total: ~$0.35
"""

import time
from datetime import datetime
from typing import Any

from ...core.models.candidate_profile import CandidateProfile, ParsedResume
from ...core.storage.object_store import ObjectStore
from ...integrations.openai_client import OpenAIClient
from ...observability.logger import get_logger
from .embedder import EmbeddingGenerator
from .indexer import MetadataIndexer
from .summarizer import ResumeSummarizer

logger = get_logger(__name__)


class IngestionResult:
    """Result of knowledge base ingestion."""

    def __init__(
        self,
        success: bool,
        candidates_processed: int,
        candidates_failed: int,
        total_cost: float,
        total_tokens: int,
        duration_seconds: float,
        errors: list[str] | None = None,
    ):
        """Initialize ingestion result.

        Args:
            success: Whether ingestion succeeded overall
            candidates_processed: Number successfully processed
            candidates_failed: Number that failed
            total_cost: Estimated total cost in USD
            total_tokens: Total tokens used
            duration_seconds: Total duration in seconds
            errors: List of error messages
        """
        self.success = success
        self.candidates_processed = candidates_processed
        self.candidates_failed = candidates_failed
        self.total_cost = total_cost
        self.total_tokens = total_tokens
        self.duration_seconds = duration_seconds
        self.errors = errors or []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "candidates_processed": self.candidates_processed,
            "candidates_failed": self.candidates_failed,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors,
        }


class KnowledgeBaseIngestionPipeline:
    """Orchestrates knowledge base ingestion from existing candidate data."""

    def __init__(
        self,
        store: ObjectStore,
        openai_client: OpenAIClient | None = None,
        chroma_client: Any | None = None,  # ChromaDB client
        skip_if_exists: bool = True,
    ):
        """Initialize ingestion pipeline.

        Args:
            store: Object store for persisted artifacts
            openai_client: OpenAI client
            chroma_client: ChromaDB client (optional, will create if None)
            skip_if_exists: Skip candidates that already have embeddings
        """
        self.store = store
        self.openai_client = openai_client or OpenAIClient()
        self.chroma_client = chroma_client
        self.skip_if_exists = skip_if_exists

        # Initialize components
        self.summarizer = ResumeSummarizer(self.openai_client)
        self.embedder = EmbeddingGenerator(self.openai_client)
        self.indexer = MetadataIndexer()

        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def ingest_all_candidates(
        self,
        job_id: str | None = None,
        batch_size: int = 50,
    ) -> IngestionResult:
        """Ingest all candidates from database into knowledge base.

        Args:
            job_id: Optional job ID to filter candidates (None = all candidates)
            batch_size: Batch size for processing

        Returns:
            IngestionResult with stats
        """
        start_time = time.time()

        self.logger.info(
            "ingestion_started",
            job_id=job_id,
            batch_size=batch_size,
            skip_if_exists=self.skip_if_exists,
        )

        try:
            # 1. Load candidates from database
            candidates = await self._load_candidates(job_id)
            total_candidates = len(candidates)

            self.logger.info("candidates_loaded", count=total_candidates)

            if total_candidates == 0:
                return IngestionResult(
                    success=True,
                    candidates_processed=0,
                    candidates_failed=0,
                    total_cost=0,
                    total_tokens=0,
                    duration_seconds=time.time() - start_time,
                )

            # 2. Process in batches
            processed = 0
            failed = 0
            total_cost = 0.0
            total_tokens = 0
            errors = []

            for i in range(0, total_candidates, batch_size):
                batch = candidates[i : i + batch_size]
                batch_num = i // batch_size + 1

                self.logger.info(
                    "processing_batch",
                    batch_num=batch_num,
                    batch_size=len(batch),
                    progress=f"{i}/{total_candidates}",
                )

                try:
                    # Process batch
                    batch_result = await self._process_batch(batch)

                    processed += batch_result["processed"]
                    failed += batch_result["failed"]
                    total_cost += batch_result["cost"]
                    total_tokens += batch_result["tokens"]
                    errors.extend(batch_result["errors"])

                except Exception as e:
                    self.logger.error(
                        "batch_processing_failed",
                        batch_num=batch_num,
                        error=str(e),
                        exc_info=True,
                    )
                    failed += len(batch)
                    errors.append(f"Batch {batch_num} failed: {str(e)}")

            duration = time.time() - start_time

            self.logger.info(
                "ingestion_complete",
                processed=processed,
                failed=failed,
                total_cost=total_cost,
                total_tokens=total_tokens,
                duration_seconds=duration,
            )

            return IngestionResult(
                success=failed < total_candidates,  # Success if at least one succeeded
                candidates_processed=processed,
                candidates_failed=failed,
                total_cost=total_cost,
                total_tokens=total_tokens,
                duration_seconds=duration,
                errors=errors,
            )

        except Exception as e:
            self.logger.error("ingestion_failed", error=str(e), exc_info=True)
            return IngestionResult(
                success=False,
                candidates_processed=0,
                candidates_failed=0,
                total_cost=0,
                total_tokens=0,
                duration_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    async def _load_candidates(self, job_id: str | None = None) -> list[tuple[str, str, CandidateProfile, ParsedResume]]:
        """Load candidates from the object store."""
        job_dirs = [self.store.base_dir / job_id] if job_id else [
            p for p in self.store.base_dir.iterdir() if p.is_dir()
        ]

        candidates: list[tuple[str, str, CandidateProfile, ParsedResume]] = []
        for job_dir in job_dirs:
            jid = job_dir.name
            for cid in self.store.list_candidate_ids(jid):
                profile = self.store.load_candidate_profile(jid, cid)
                parsed = self.store.load_parsed_resume(jid, cid)
                if not profile or not parsed:
                    continue

                if self.skip_if_exists:
                    kb_rec = self.store.load_kb_record(jid, cid)
                    if kb_rec and kb_rec.get("embedding_id"):
                        continue

                candidates.append((jid, cid, profile, parsed))

        return candidates

    async def _process_batch(
        self, candidates: list[tuple[str, str, CandidateProfile, ParsedResume]]
    ) -> dict[str, Any]:
        """Process a batch of candidates.

        Args:
            candidates: List of (job_id, candidate_id, profile, parsed_resume)

        Returns:
            Dict with processing stats
        """
        candidate_data = candidates

        # 1. Generate summaries
        summaries = []
        summary_tokens = 0
        for job_id, candidate_id, candidate_profile, parsed_resume in candidate_data:
            try:
                summary, metadata = await self.summarizer.summarize_candidate(
                    candidate_profile, parsed_resume
                )
                summaries.append((job_id, candidate_id, candidate_profile, parsed_resume, summary))
                summary_tokens += metadata.get("tokens_total", 0)
            except Exception as e:
                self.logger.error(
                    "summarization_failed",
                    candidate_id=candidate_id,
                    error=str(e),
                )

        # 2. Generate embeddings
        embeddings = []
        embedding_tokens = 0
        if summaries:
            embedding_inputs = [(summary, candidate_id) for _, candidate_id, _, _, summary in summaries]
            try:
                embedding_results = await self.embedder.generate_embeddings_batch(
                    embedding_inputs, batch_size=100
                )
                embeddings = embedding_results
                embedding_tokens = sum(m.get("tokens_used", 0) for _, m in embedding_results)
            except Exception as e:
                self.logger.error("batch_embedding_failed", error=str(e))

        # 3. Index metadata
        indexed_metadata = []
        if summaries:
            index_inputs = [
                (candidate_profile, parsed_resume, summary)
                for _, _, candidate_profile, parsed_resume, summary in summaries
            ]
            indexed_metadata = self.indexer.index_batch(index_inputs)

        # 4. Store in ChromaDB (if available)
        if self.chroma_client and embeddings:
            try:
                await self._store_embeddings_in_chroma(embeddings, summaries, indexed_metadata)
            except Exception as e:
                self.logger.error("chroma_storage_failed", error=str(e))

        # 5. Update database
        processed = 0
        failed = 0
        errors = []

        for i, (job_id, candidate_id, _, _, summary) in enumerate(summaries):
            try:
                record: dict[str, Any] = {
                    "summary_text": summary.summary_text,
                    "updated_at": datetime.utcnow().isoformat(),
                }

                if i < len(embeddings):
                    embedding, emb_metadata = embeddings[i]
                    record["embedding_id"] = f"emb_{candidate_id}"
                    record["embedding_generated_at"] = datetime.utcnow().isoformat()
                    record["embedding_dimensions"] = len(embedding)
                    record["embedding_tokens_used"] = emb_metadata.get("tokens_used", 0)

                if i < len(indexed_metadata):
                    metadata_dict = indexed_metadata[i].to_dict()
                    record.update(metadata_dict)

                self.store.save_kb_record(job_id, candidate_id, record)
                processed += 1

            except Exception as e:
                self.logger.error(
                    "database_update_failed",
                    candidate_id=candidate_id,
                    error=str(e),
                )
                failed += 1
                errors.append(f"Candidate {candidate_id}: {str(e)}")

        # Calculate cost (rough estimate)
        # gpt-4o-mini: ~$0.15/1M input, ~$0.60/1M output
        summary_cost = (summary_tokens / 1_000_000) * 0.40  # Average of input/output
        # text-embedding-3-small: $0.02/1M tokens
        embedding_cost = (embedding_tokens / 1_000_000) * 0.02
        total_cost = summary_cost + embedding_cost

        return {
            "processed": processed,
            "failed": failed,
            "cost": total_cost,
            "tokens": summary_tokens + embedding_tokens,
            "errors": errors,
        }

    async def _store_embeddings_in_chroma(
        self,
        embeddings: list[tuple[list[float], dict[str, Any]]],
        summaries: list[tuple[str, str, CandidateProfile, ParsedResume, Any]],
        indexed_metadata: list[Any],
    ) -> None:
        """Store embeddings in ChromaDB.

        Args:
            embeddings: List of (embedding, metadata) tuples
            summaries: List of (job_id, candidate_id, profile, resume, summary) tuples
        """
        if not self.chroma_client:
            self.logger.warning("chroma_client_not_available")
            return

        try:
            # Extract data for ChromaDB
            embedding_vectors = [emb for emb, _ in embeddings]
            candidate_ids = [candidate_id for _, candidate_id, _, _, _ in summaries]

            # Build metadata for each candidate
            metadatas = []
            documents = []  # Summary texts

            for i, (job_id, candidate_id, _, _, summary) in enumerate(summaries):
                metadata = {
                    "candidate_id": candidate_id,
                    "job_id": job_id,
                }
                if i < len(indexed_metadata) and indexed_metadata[i]:
                    md = indexed_metadata[i]
                    metadata.update(
                        {
                            "last_company": md.last_company,
                            "last_title": md.last_job_title,
                            "years_experience_total": md.years_experience_total,
                        }
                    )

                metadatas.append(metadata)
                documents.append(summary.summary_text)

            # Store in ChromaDB
            self.chroma_client.add_embeddings(
                embeddings=embedding_vectors,
                candidate_ids=candidate_ids,
                metadatas=metadatas,
                documents=documents,
            )

            self.logger.info("chroma_embeddings_stored", count=len(embeddings))

        except Exception as e:
            self.logger.error("chroma_storage_failed", error=str(e), exc_info=True)
            raise
