"""Semantic search service using Chroma (if available) plus metadata filters."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from ..core.storage.object_store import ObjectStore
from ..integrations.openai_client import OpenAIClient
from ..observability.logger import get_logger

logger = get_logger(__name__)


class SearchResult(BaseModel):
    candidate_id: str
    job_id: str
    score: float = Field(ge=0.0)
    summary_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RerankOutput(BaseModel):
    candidate_ids: list[str]


class SearchService:
    def __init__(
        self,
        store: ObjectStore,
        chroma_client: Any | None = None,
        openai_client: OpenAIClient | None = None,
    ):
        self.store = store
        self.chroma = chroma_client
        self.openai = openai_client or OpenAIClient(model="gpt-5.1")
        self.test_mode = bool(os.getenv("RSAS_TEST_MODE"))

    async def search(
        self,
        query: str,
        job_id: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Perform semantic search with optional metadata filtering."""
        filters = filters or {}

        if self.chroma and not self.test_mode:
            return await self._search_chroma(query, job_id, top_k, filters, rerank)

        # Fallback: simple keyword score over stored summaries
        return self._search_fallback(query, job_id, top_k, filters)

    def _search_fallback(
        self,
        query: str,
        job_id: str | None,
        top_k: int,
        filters: dict[str, Any],
    ) -> list[SearchResult]:
        query_terms = query.lower().split()
        results: list[SearchResult] = []

        for jid, cid, record in self.store.list_kb_records(job_id):
            summary = record.get("summary_text", "") or ""
            text = summary.lower()
            score = sum(text.count(term) for term in query_terms)

            # Basic metadata filters
            if "last_company" in filters and record.get("last_company") != filters["last_company"]:
                continue
            if "last_title" in filters and record.get("last_title") != filters["last_title"]:
                continue

            results.append(
                SearchResult(
                    candidate_id=cid,
                    job_id=jid,
                    score=float(score),
                    summary_text=summary,
                    metadata=record,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _search_chroma(
        self,
        query: str,
        job_id: str | None,
        top_k: int,
        filters: dict[str, Any],
        rerank: bool,
    ) -> list[SearchResult]:
        embedding, _ = await self.openai.generate_embedding(query)

        where: dict[str, Any] = {}
        if job_id:
            where["job_id"] = job_id
        where.update(filters)

        chroma_results = self.chroma.query(query_embedding=embedding, n_results=top_k, where=where)
        ids = chroma_results.get("ids", [[]])[0] if chroma_results else []
        distances = chroma_results.get("distances", [[]])[0] if chroma_results else []
        metas = chroma_results.get("metadatas", [[]])[0] if chroma_results else []
        docs = chroma_results.get("documents", [[]])[0] if chroma_results else []

        results: list[SearchResult] = []
        for cid, dist, meta, doc in zip(ids, distances, metas, docs):
            if not cid:
                continue
            cid_str = cid
            job_val = (meta or {}).get("job_id") or job_id or ""
            score = float(1.0 - dist) if dist is not None else 0.0
            record = self.store.load_kb_record(job_val, cid_str) or {}
            summary_text = record.get("summary_text") or doc
            merged_meta = {**record, **(meta or {})}
            results.append(
                SearchResult(
                    candidate_id=cid_str,
                    job_id=job_val,
                    score=score,
                    summary_text=summary_text,
                    metadata=merged_meta,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)

        if rerank and len(results) > 1 and not self.test_mode:
            results = await self._rerank_with_gpt(query, results, top_k)

        return results[:top_k]

    async def _rerank_with_gpt(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Optional GPT rerank on top results."""
        items = results[: max(top_k, 5)]
        prompt = "Rerank candidates for query: {q}\n\n".format(q=query)
        for idx, res in enumerate(items, start=1):
            prompt += f"{idx}. {res.candidate_id}: {res.summary_text}\n"
        prompt += "Return JSON with field `candidate_ids` ordered best to worst."

        rerank_output, _ = await self.openai.create_response(
            input_text=prompt,
            response_model=RerankOutput,
            model="gpt-5.1",
        )
        ordered_ids: list[str] = rerank_output.candidate_ids

        id_to_result = {res.candidate_id: res for res in items}
        reranked = [id_to_result[cid] for cid in ordered_ids if cid in id_to_result]

        # Fill any missing
        for res in items:
            if res not in reranked:
                reranked.append(res)
        return reranked
