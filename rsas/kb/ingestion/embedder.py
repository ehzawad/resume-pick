"""Embedding generator for candidate summaries using OpenAI text-embedding-3-small.

Cost optimization: Embed only 300-token summaries, not full resumes.
- Full resume: 10,000 tokens × $0.00002 = $0.0002 → $600 for 3,000
- Summary only: 300 tokens × $0.00002 = $0.000006 → $18 for 3,000
97% cost reduction!
"""

from typing import Any

from ...integrations.openai_client import OpenAIClient
from ...observability.logger import get_logger
from .summarizer import CandidateSummary

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for candidate summaries."""

    def __init__(
        self,
        openai_client: OpenAIClient | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        """Initialize embedding generator.

        Args:
            openai_client: OpenAI client instance
            model: Embedding model (default: text-embedding-3-small)
            dimensions: Optional embedding dimensions
        """
        self.openai_client = openai_client or OpenAIClient()
        self.model = model
        self.dimensions = dimensions
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def generate_embedding(
        self, summary: CandidateSummary, candidate_id: str
    ) -> tuple[list[float], dict[str, Any]]:
        """Generate embedding for a candidate summary.

        Args:
            summary: Candidate summary object
            candidate_id: Candidate identifier

        Returns:
            Tuple of (embedding vector, metadata with cost/tokens)
        """
        self.logger.info("generating_embedding", candidate_id=candidate_id)

        # Build embedding text from summary
        embedding_text = self._build_embedding_text(summary)

        # Generate embedding
        embedding, metadata = await self.openai_client.generate_embedding(
            text=embedding_text,
            model=self.model,
            dimensions=self.dimensions,
        )

        metadata["candidate_id"] = candidate_id
        metadata["text_length"] = len(embedding_text)

        self.logger.info(
            "embedding_generated",
            candidate_id=candidate_id,
            dimensions=metadata.get("dimensions", 0),
            tokens=metadata.get("tokens_used", 0),
        )

        return embedding, metadata

    async def generate_embeddings_batch(
        self,
        summaries: list[tuple[CandidateSummary, str]],  # (summary, candidate_id) pairs
        batch_size: int = 100,
    ) -> list[tuple[list[float], dict[str, Any]]]:
        """Generate embeddings for multiple candidate summaries.

        Args:
            summaries: List of (summary, candidate_id) tuples
            batch_size: Batch size for API calls

        Returns:
            List of (embedding vector, metadata) tuples
        """
        self.logger.info("generating_embeddings_batch", count=len(summaries))

        # Build embedding texts
        embedding_texts = [self._build_embedding_text(summary) for summary, _ in summaries]
        candidate_ids = [cid for _, cid in summaries]

        # Generate embeddings in batch
        embeddings, batch_metadata = await self.openai_client.generate_embeddings_batch(
            texts=embedding_texts,
            model=self.model,
            dimensions=self.dimensions,
            batch_size=batch_size,
        )

        # Package results with individual metadata
        results = []
        for i, (embedding, candidate_id) in enumerate(zip(embeddings, candidate_ids)):
            metadata = {
                "candidate_id": candidate_id,
                "tokens_used": batch_metadata["tokens_used"] // len(summaries),  # Approximate
                "dimensions": len(embedding),
                "model": self.model,
            }
            results.append((embedding, metadata))

        self.logger.info(
            "batch_embedding_complete",
            count=len(results),
            total_tokens=batch_metadata["tokens_used"],
        )

        return results

    def _build_embedding_text(self, summary: CandidateSummary) -> str:
        """Build embedding text from candidate summary.

        Combines all structured fields into a single searchable text.

        Args:
            summary: Candidate summary

        Returns:
            Text for embedding
        """
        parts = [
            # Summary text
            summary.summary_text,
            # Skills
            "Skills: " + ", ".join(summary.key_skills),
            # Experience highlights
            "Experience: " + " | ".join(summary.experience_highlights),
            # Domain expertise
            "Domains: " + ", ".join(summary.domain_expertise),
        ]

        # Add education if available
        if summary.education_summary:
            parts.append(f"Education: {summary.education_summary}")

        # Add years if available
        if summary.years_experience:
            parts.append(f"{summary.years_experience:.1f} years experience")

        return " | ".join(parts)


def estimate_embedding_cost(num_candidates: int, avg_summary_tokens: int = 300) -> dict[str, float]:
    """Estimate embedding costs for a batch of candidates.

    Args:
        num_candidates: Number of candidates
        avg_summary_tokens: Average tokens per summary (default: 300)

    Returns:
        Dict with cost breakdown
    """
    # text-embedding-3-small pricing: $0.00002 per 1K tokens
    cost_per_1k_tokens = 0.00002
    total_tokens = num_candidates * avg_summary_tokens
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens

    return {
        "num_candidates": num_candidates,
        "avg_summary_tokens": avg_summary_tokens,
        "total_tokens": total_tokens,
        "cost_per_candidate": total_cost / num_candidates if num_candidates > 0 else 0,
        "total_cost": total_cost,
        "cost_breakdown": {
            "model": "text-embedding-3-small",
            "rate_per_1k_tokens": cost_per_1k_tokens,
        },
    }
